"""Export the fixed YOLOv8n-224 model used by the Coral Micro."""

import hashlib
from pathlib import Path
import shutil
import types

import cv2
import numpy as np
import onnx
import onnxsim
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build/yolov8'


def raw_heads(self, features):
    outputs = []
    for i in range(self.nl):
        outputs.extend((self.cv2[i](features[i]),
                        self.cv3[i](features[i]).sigmoid()))
    return tuple(outputs)


def export_model():
    BUILD.mkdir(parents=True, exist_ok=True)
    weights = BUILD / 'yolov8n.pt'
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == (
        'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36')
    torch.set_num_threads(4)
    model = YOLO(weights).model.eval().fuse()

    # Explicit channel splits preserve the converter's NHWC layout.
    for module in model.modules():
        if isinstance(module, C2f):
            module.forward = module.forward_split

    # Keep box decoding on the CPU; export logits and class probabilities.
    head = model.model[-1]
    head.export = True
    head.format = 'tflite'
    head.dynamic = False
    head.forward = types.MethodType(raw_heads, head)
    path = BUILD / 'yolov8n.onnx'
    torch.onnx.export(
        model, torch.zeros((1, 3, 224, 224)), path,
        input_names=['images'],
        output_names=['box8', 'cls8', 'box16', 'cls16', 'box32', 'cls32'],
        opset_version=13, do_constant_folding=True, dynamo=False)
    simplified, valid = onnxsim.simplify(onnx.load(path))
    assert valid
    onnx.save(simplified, path)

    # Calibration uses the same RGB nearest-neighbor stretch as the camera.
    images = sorted((BUILD / 'calibration/coco128/images/train2017').glob('*.jpg'))
    assert len(images) == 128
    calibration = []
    for path in images:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
        calibration.append(rgb.astype(np.float32) / 255.)
    np.save(BUILD / 'calibration.npy', np.stack(calibration))

    # onnx2tf also uses small probe images to check tensor layouts.
    np.save(BUILD / 'calibration_image_sample_data_20x128x128x3_float32.npy',
            np.stack([cv2.resize(image, (128, 128)) for image in calibration[:20]]))
    assert set(model.names) == set(range(80))
    (BUILD / 'coco_labels.txt').write_text(
        ''.join(f'{model.names[i]}\n' for i in range(80)))


def install_model():
    """Check the compiled tensor contract before installing model and labels."""
    from tensorflow.lite.python import schema_py_generated as schema

    path = BUILD / 'edgetpu/yolov8n_full_integer_quant_edgetpu.tflite'
    model = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    assert model.SubgraphsLength() == 1
    graph = model.Subgraphs(0)
    assert graph.InputsLength() == 1 and graph.OutputsLength() == 6
    input_tensor = graph.Tensors(graph.Inputs(0))
    assert input_tensor.ShapeAsNumpy().tolist() == [1, 224, 224, 3]
    assert input_tensor.Type() == schema.TensorType.UINT8
    quantization = input_tensor.Quantization()
    assert quantization.ScaleLength() == quantization.ZeroPointLength() == 1
    assert quantization.Scale(0) == np.float32(1 / 255)
    assert quantization.ZeroPoint(0) == 0

    # Output order is part of the decoder's contract in yolov8.h.
    for i, (grid, channels) in enumerate(
            [(7, 64), (28, 80), (7, 80), (14, 80), (28, 64), (14, 64)]):
        tensor = graph.Tensors(graph.Outputs(i))
        assert tensor.ShapeAsNumpy().tolist() == [1, grid, grid, channels]
        assert tensor.Type() == schema.TensorType.UINT8
        quantization = tensor.Quantization()
        assert quantization.ScaleLength() == quantization.ZeroPointLength() == 1
        assert quantization.Scale(0) > 0
    assert graph.OperatorsLength() == 1
    opcode = model.OperatorCodes(graph.Operators(0).OpcodeIndex())
    assert opcode.CustomCode() == b'edgetpu-custom-op'
    labels = (BUILD / 'coco_labels.txt').read_text()
    assert len(labels.splitlines()) == 80

    destination = ROOT / 'models/yolov8'
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination / 'yolov8n-int8_edgetpu.tflite')
    (destination / 'coco_labels.txt').write_text(labels)


if __name__ == '__main__':
    export_model()
