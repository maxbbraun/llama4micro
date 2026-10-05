#!/usr/bin/env python3
"""Fetch pinned Amy Small weights, pack the CPU front, and prepare calibration."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import urllib.request

import numpy as np

REVISION = "c532a5d21c078a16cb633718e9182bfd71a5b760"
RUNTIME_REVISION = "3de9f37cbfedb8a1edcc28f4c8139979c6fcf889"
BASE_URL = f"https://huggingface.co/ampixa/sanoTTS/resolve/{REVISION}/amy-en-1p1m"
SOURCE_HASHES = {
    "manifest.json": "6d4e580326353fdf67d193687714875dace782a0807e108a6858445d2464c62b",
    "piper-phoneme-config.json": "95a23eb4d42909d38df73bb9ac7f45f597dbfcde2d1bf9526fdeaf5466977d77",
    "weights.fp16.bin": "b45240e7c24ac4ca4a1572a7ccc971dcca1836dcfde9aeb205d54d5515a07610",
}
CALIBRATION_TEXTS = [
    "Once upon a time, a little bear found a bright red boat.",
    "He asked his friends to come along, and together they sailed across the quiet lake.",
    "A small fox waited by the window while rain tapped softly on the roof.",
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch_source(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_HASHES.items():
        path = directory / name
        if not path.exists():
            with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
                data = response.read()
            if sha256(data) != expected:
                raise ValueError(f"Source checksum mismatch: {name}")
            path.write_bytes(data)
        if sha256(path.read_bytes()) != expected:
            raise ValueError(f"Cached source checksum mismatch: {path}")


def load_weights(directory):
    for name, expected in SOURCE_HASHES.items():
        if sha256((directory / name).read_bytes()) != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    manifest = json.loads((directory / "manifest.json").read_text())
    raw = (directory / "weights.fp16.bin").read_bytes()
    assert manifest["package_name"] == "amy-en-1p1m"
    assert manifest["sample_rate"] == 22050 and manifest["hop_length"] == 256
    assert len(raw) == manifest["weights_size_bytes"] == 2169944
    arrays = {}
    spans = []
    for component, description in manifest["components"].items():
        for tensor in description["tensors"]:
            start = tensor["offset_bytes"]
            end = start + tensor["nbytes"]
            data = raw[start:end]
            assert tensor["dtype"] == "float16"
            assert len(data) == 2 * np.prod(tensor["shape"])
            assert sha256(data) == tensor["sha256"]
            value = np.frombuffer(data, dtype="<f2").astype("<f4").reshape(tensor["shape"])
            assert np.isfinite(value).all()
            arrays[component, tensor["name"]] = value
            spans.append((start, end))
    spans.sort()
    assert spans[0][0] == 0 and spans[-1][1] == len(raw)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    return manifest, arrays


def pack_front(manifest, arrays):
    """snt_front_f32.h header and the upstream C runtime's 62 weight slots."""
    duration = manifest["components"]["duration"]["config"]
    acoustic = manifest["components"]["acoustic"]["config"]
    assert duration["architecture"] == "duration_conv"
    assert acoustic["architecture"] == "token_context"
    assert duration.get("target_duration_scale", 1) == 1
    assert duration.get("pause_preserve_scale", 0) == 0
    assert duration.get("long_preserve_scale", 0) == 0

    def pair(component, name):
        return [(component, name + ".weight"), (component, name + ".bias")]

    def blocks(component, name, count):
        slots = []
        for index in range(count):
            base = f"{name}.{index}"
            slots += [(component, base + ".scale")]
            slots += pair(component, base + ".net.0") + pair(component, base + ".net.2")
        return slots

    slots = [("duration", "embedding.weight")] + pair("duration", "input_proj")
    slots += blocks("duration", "blocks", duration["depth"]) + pair("duration", "output")
    slots += [("acoustic", "embedding.weight")] + pair("acoustic", "token_input_proj")
    slots += blocks("acoustic", "token_blocks", acoustic["token_depth"])
    slots += pair("acoustic", "frame_input_proj")
    slots += blocks("acoustic", "frame_blocks", acoustic["depth"]) + pair("acoustic", "output")
    assert len(slots) == len(set(slots)) == 62
    assert set(slots) == {key for key in arrays if key[0] in ("duration", "acoustic")}
    header = struct.pack(
        "<18i", 0x534E4652, 1,
        duration["vocab_size"], duration["hidden"], duration["depth"],
        duration["kernel_size"], duration["max_tokens"], duration["max_duration"],
        acoustic["vocab_size"], acoustic["hidden"], acoustic["token_depth"],
        acoustic["depth"], acoustic["kernel_size"], acoustic["out_channels"],
        0, 0, 0, len(slots),
    )
    table, weights = bytearray(), bytearray()
    for key in slots:
        value = arrays[key]
        table += struct.pack("<2i", len(weights) // 4, value.size)
        weights += value.tobytes()
    blob = header + table + weights
    assert len(blob) == 1417572
    assert sha256(blob) == "f02528441a16dedb3ba9b6b283ed3193973af09f686e71b48ce0faaf11917836"
    return blob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--work-dir", type=Path, default=repo / "build/amy-small")
    parser.add_argument("--sanotts", type=Path, default=repo / "sanoTTS")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.sanotts), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != RUNTIME_REVISION:
        raise ValueError(f"Expected sanoTTS {RUNTIME_REVISION}, found {revision}")
    source = args.work_dir / "source"
    fetch_source(source)
    manifest, arrays = load_weights(source)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "front_f32.bin").write_bytes(pack_front(manifest, arrays))

    # Use the pinned dictionary frontend. No eSpeak, audio, or random calibration.
    sys.path.insert(0, str(args.sanotts / "pypkg"))
    from sanotts import frontend, models, piper_g2p, voicepack

    pack = voicepack.VoicePack("amy-1p1m", source, manifest, (source / "weights.fp16.bin").read_bytes())
    table = frontend.load_phoneme_table(pack.phoneme_config_path)
    calibration = []
    for index, text in enumerate(CALIBRATION_TEXTS):
        ids, dropped = piper_g2p.text_to_phoneme_ids(text, table)
        assert not dropped
        ids = np.asarray(ids, dtype="<i4")
        durations = models.duration_forward(
            pack.component_tensors("duration"), pack.component_config("duration"),
            ids, length_scale=pack.duration_length_scale,
        )
        latent = models.acoustic_forward(
            pack.component_tensors("acoustic"), pack.component_config("acoustic"),
            ids, durations,
        ).astype("<f4")
        assert np.isfinite(latent).all() and latent.shape[0] == 192
        filename = f"calibration-{index}.npy"
        np.save(args.work_dir / filename, latent, allow_pickle=False)
        calibration.append({"text": text, "ids": ids.tolist(), "frames": int(durations.sum()),
                            "latent": filename, "latent_sha256": sha256(latent.tobytes())})
    report = {"model_revision": REVISION, "runtime_revision": revision,
              "numpy": np.__version__, "source_sha256": SOURCE_HASHES,
              "calibration": calibration}
    (args.work_dir / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared front_f32.bin and {len(calibration)} calibration passages in {args.work_dir}")


if __name__ == "__main__":
    main()
