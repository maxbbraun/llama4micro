#!/usr/bin/env python3
"""Quantize and co-compile the two fixed Amy Small decoder windows for Edge TPU."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from prepare import load_weights


class Decoder(tf.Module):
    def __init__(self, weights, frames, tail=False):
        super().__init__()
        self.weights = weights
        self.tail = tail
        channels = 40 if tail else 192
        self.inference = tf.function(
            self.forward,
            input_signature=[tf.TensorSpec([1, 1, frames, channels], tf.float32, name="latent")],
        )

    def conv(self, x, name, dilation=1):
        weight = self.weights[name + ".weight"].transpose(2, 1, 0)[None]
        y = tf.nn.conv2d(
            x, tf.constant(weight), strides=[1, 1, 1, 1], padding="SAME",
            dilations=[1, 1, dilation, 1], name=name.replace(".", "_") + "_conv",
        )
        return tf.nn.bias_add(y, tf.constant(self.weights[name + ".bias"]))

    @staticmethod
    def act(x, alpha):
        # MAXIMUM(x, alpha*x) makes TF2.15 narrow x to the output range.
        return tf.nn.relu(x) - tf.nn.relu(x * tf.constant(np.float32(-alpha)))

    def forward(self, z):
        x = z if self.tail else self.conv(z, "pre")
        for stage, (stride, padding) in enumerate([(8, 4), (8, 4), (4, 2)]):
            if self.tail and stage < 2:
                continue
            x = self.act(x, .1)
            name = f"up{stage}"
            weight = self.weights[name + ".weight"]
            in_channels, out_channels, kernel = weight.shape
            phase = np.zeros((1, 3, in_channels, stride * out_channels), np.float32)
            # Transpose convolution as low-rate convolution and phase reshape.
            for r in range(stride):
                for delta in [-1, 0, 1]:
                    k = r + padding - delta * stride
                    if 0 <= k < kernel:
                        phase[0, delta + 1, :, r*out_channels:(r+1)*out_channels] = weight[:, :, k]
            x = tf.nn.conv2d(x, tf.constant(phase), strides=[1, 1, 1, 1],
                             padding="SAME", name=name + "_phase_conv")
            x = tf.nn.bias_add(x, tf.constant(np.tile(self.weights[name + ".bias"], stride)))
            x = tf.reshape(x, [1, 1, int(x.shape[2]) * stride, out_channels],
                           name=name + "_phase_reshape")
            branches = []
            for branch, (d1, d2) in enumerate([(1, 2), (2, 6), (3, 12)]):
                base = f"res{stage}.0.blocks.{branch}"
                y = self.conv(self.act(x, .1), base + ".conv1", d1) + x
                y = self.conv(self.act(y, .1), base + ".conv2", d2) + y
                branches.append(y)
            x = (branches[0] + branches[1] + branches[2]) * tf.constant(np.float32(1/3))
            if not self.tail and stage == 1:
                return x
        return tf.identity(self.conv(self.act(x, .01), "post"), name="pre_tanh")


def windows(values, frames):
    """The validated calibration: five evenly spaced full windows per text."""
    result = []
    for value in values:
        value = value.T
        assert len(value) >= frames
        for start in np.linspace(0, len(value) - frames, 5, dtype=int):
            result.append(value[start:start + frames][None, None].astype(np.float32))
    return result


def symmetric_scalars(blob):
    """Preserve scalar real values while avoiding the compiler's zero-point bug."""
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(blob, 0))
    graph = model.subgraphs[0]
    original = [None if b.data is None else np.array(b.data, copy=True) for b in model.buffers]
    changed, seen = [], set()
    for op in graph.operators:
        code = model.operatorCodes[op.opcodeIndex]
        if max(code.builtinCode, code.deprecatedBuiltinCode) != schema.BuiltinOperator.MUL:
            continue
        for index in op.inputs:
            if index in seen:
                continue
            tensor = graph.tensors[index]
            data = original[tensor.buffer]
            if tensor.type != schema.TensorType.INT8 or data is None or len(data) != 1:
                continue
            seen.add(index)
            q = int(data.view(np.int8)[0])
            zero = int(tensor.quantization.zeroPoint[0])
            scale = float(tensor.quantization.scale[0])
            value = (q - zero) * scale
            assert value != 0
            new_q = 127 if value > 0 else -127
            new_scale = np.float32(abs(value) / 127)
            assert abs(new_q * float(new_scale) - value) <= abs(value) * 1e-7
            # Different tensors can share one byte buffer with different scales.
            # Read originals above and give each rewrite its own private buffer.
            buffer = schema.BufferT()
            buffer.data = np.array([new_q], dtype=np.int8).view(np.uint8)
            tensor.buffer = len(model.buffers)
            model.buffers.append(buffer)
            tensor.quantization.scale = np.array([new_scale], dtype=np.float32)
            tensor.quantization.zeroPoint = np.array([0], dtype=np.int64)
            changed.append({"tensor": int(index), "before": value,
                            "after": new_q * float(new_scale)})
    builder = flatbuffers.Builder(0)
    offset = model.Pack(builder)
    builder.Finish(offset, file_identifier=b"TFL3")
    return bytes(builder.Output()), changed


def interface(blob):
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(blob, 0))
    graph = model.subgraphs[0]
    result = []
    for index in [graph.inputs[0], graph.outputs[0]]:
        tensor = graph.tensors[index]
        assert tensor.type == schema.TensorType.INT8
        result.append({"shape": list(map(int, tensor.shape)),
                       "scale": float(tensor.quantization.scale[0]),
                       "zero_point": int(tensor.quantization.zeroPoint[0])})
    return result


def convert(model, calibration):
    concrete = model.inference.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([x] for x in calibration)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    original = converter.convert()
    fixed, changes = symmetric_scalars(original)
    assert interface(original) == interface(fixed)
    # Check the rewrite on every calibration window before invoking the compiler.
    interpreters = [tf.lite.Interpreter(model_content=b, num_threads=1) for b in [original, fixed]]
    for interpreter in interpreters:
        interpreter.allocate_tensors()
    maximum = 0
    for value in calibration:
        results = []
        for interpreter in interpreters:
            input_info = interpreter.get_input_details()[0]
            output_info = interpreter.get_output_details()[0]
            scale, zero = input_info["quantization"]
            q = np.clip(np.rint(value / scale) + zero, -128, 127).astype(np.int8)
            interpreter.set_tensor(input_info["index"], q)
            interpreter.invoke()
            results.append(interpreter.get_tensor(output_info["index"]).astype(np.int16))
        maximum = max(maximum, int(np.max(np.abs(results[0] - results[1]))))
    assert maximum <= 1, f"Scalar rewrite changed calibration output by {maximum} codes"
    return fixed, {"scalar_constants": changes, "rewrite_max_difference_codes": maximum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--work-dir", type=Path, default=repo / "build/amy-small")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--compiler", default="edgetpu_compiler")
    args = parser.parse_args()
    if tf.__version__ != "2.15.0":
        raise ValueError(f"Expected TensorFlow 2.15.0, found {tf.__version__}")
    manifest, arrays = load_weights(args.work_dir / "source")
    weights = {name: value for (component, name), value in arrays.items() if component == "decoder"}
    calibration = json.loads((args.work_dir / "calibration.json").read_text())
    latent = []
    for row in calibration["calibration"]:
        value = np.load(args.work_dir / row["latent"], allow_pickle=False)
        assert hashlib.sha256(value.tobytes()).hexdigest() == row["latent_sha256"]
        latent.append(value)
    mix1 = [Decoder(weights, value.shape[1]).inference(value.T[None, None]).numpy()[0, 0].T
            for value in latent]
    graphs = [("amy_prefix", Decoder(weights, 32), windows(latent, 32)),
              ("amy_tail", Decoder(weights, 666, tail=True), windows(mix1, 666))]
    graph_dir = args.work_dir / "graphs"
    graph_dir.mkdir(exist_ok=True)
    report = {"source_revision": calibration["model_revision"], "tensorflow": tf.__version__,
              "numpy": np.__version__, "calibration_texts": [r["text"] for r in calibration["calibration"]],
              "windows_per_text": 5, "models": []}
    for name, model, values in graphs:
        blob, proof = convert(model, values)
        # Keep compiler input names stable: they affect package/cache metadata.
        stem = ("amy_decoder_latent_f32_audio_mix1_width" if name == "amy_prefix"
                else "amy_decoder_mix1_f666_pre_tanh_audio_width")
        path = graph_dir / (stem + ".tflite")
        path.write_bytes(blob)
        report["models"].append({"name": name, "input": str(path), "stem": stem, "interface": interface(blob), **proof})
    compiler_version = subprocess.check_output([args.compiler, "--version"], text=True).strip()
    if "14.1.317412892" not in compiler_version:
        raise ValueError(f"Expected validated Edge TPU compiler 14.1.317412892: {compiler_version}")
    compiled_dir = args.work_dir / "compiled"
    compiled_dir.mkdir(exist_ok=True)
    command = [args.compiler, "--show_operations", "--out_dir", str(compiled_dir),
               *[row["input"] for row in report["models"]]]
    subprocess.run(command, check=True)
    outputs = []
    for row in report["models"]:
        path = compiled_dir / (row["stem"] + "_edgetpu.tflite")
        blob = path.read_bytes()
        model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(blob, 0))
        graph = model.subgraphs[0]
        assert len(graph.operators) == 1
        code = model.operatorCodes[graph.operators[0].opcodeIndex]
        assert max(code.builtinCode, code.deprecatedBuiltinCode) == schema.BuiltinOperator.CUSTOM
        assert code.customCode == b"edgetpu-custom-op"
        assert interface(blob) == row["interface"]
        row.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        outputs.append((path, row["name"] + "_edgetpu.tflite"))
    report["compiler"] = compiler_version
    (args.work_dir / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    # Publish only after both models passed all conversion and compilation checks.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, name in outputs:
        shutil.copyfile(path, args.output_dir / name)
    shutil.copyfile(args.work_dir / "front_f32.bin", args.output_dir / "front_f32.bin")
    print(f"Wrote three Amy Small model assets to {args.output_dir}")
    print("The compiler generates a fresh cache token. Validate the models before updating "
          "their fingerprints in speech/amy_model.cc and manifest.json.")


if __name__ == "__main__":
    main()
