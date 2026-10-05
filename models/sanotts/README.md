# Amy Small model conversion

Speech uses [Amy Small](https://huggingface.co/ampixa/sanoTTS/tree/c532a5d21c078a16cb633718e9182bfd71a5b760/amy-en-1p1m), the 1.08M-parameter English voice from [sanoTTS](https://github.com/Ampixa/sanoTTS). Duration and acoustic inference run on the Cortex-M7; two int8 decoder graphs run on the Edge TPU. Output is mono 22,050 Hz. The three shipped files in `models/sanotts/` total about 2.59 MiB:

- `front_f32.bin`: duration/acoustic weights, widened losslessly from the upstream FP16 package.
- `amy_prefix_edgetpu.tflite`: decoder input `[1, 1, 32, 192]`, output `[1, 1, 2048, 40]`.
- `amy_tail_edgetpu.tflite`: decoder input `[1, 1, 666, 40]`, output `[1, 1, 2664, 1]`, followed by CPU `tanh`.

The runtime advances ten latent frames (2,560 audio samples) at a time, using overlapping windows and crops to preserve convolution boundaries. These shapes and quantization parameters are part of the runtime contract. Model hashes, source revisions and tool versions are recorded in [manifest.json](manifest.json).

The export requires two Python environments because preparation uses NumPy 2 while TensorFlow 2.15 requires NumPy 1. Run the compiler phase on Linux x86-64 with [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/) **14.1.317412892** available on `PATH`. From the repository root:

```bash
python3.11 -m venv build/amy-prepare-env
build/amy-prepare-env/bin/pip install numpy==2.3.5
build/amy-prepare-env/bin/python models/sanotts/prepare.py

python3.11 -m venv build/amy-export-env
build/amy-export-env/bin/pip install tensorflow==2.15.0 numpy==1.26.2 flatbuffers==23.5.26
build/amy-export-env/bin/python models/sanotts/export.py
```

Preparation downloads the pinned package and checks every source/tensor checksum. It packs the CPU weights in the order required by the pinned `snt_front_f32` runtime, then derives calibration tensors from three fixed representative texts with sanoTTS's dictionary-based Piper frontend. No eSpeak installation is needed. The exporter uses five evenly spaced windows per text, rewrites LeakyReLU as `relu(x) - relu(-alpha*x)`, and re-encodes scalar multiplication constants with zero point zero and separate buffers. These two rewrites avoid demonstrated conversion/compiler errors without changing model weights or fitting an output gain. It checks the rewrite against all calibration windows and requires each compiled graph to contain only one Edge TPU custom operation with unchanged interfaces.

Temporary downloads, tensors and conversion reports stay under `build/amy-small/`. Both scripts accept `--work-dir`; preparation also accepts `--sanotts`, and export accepts `--output-dir` and `--compiler`. To inspect a regeneration without replacing checked-in assets, use `--output-dir build/amy-small/assets`.

The validated preparation used NumPy 2.3.5 on macOS arm64 and conversion used TensorFlow 2.15.0 on Linux x86-64. Other numerical backends can change calibration rounding. Even with identical input graphs the compiler generates a fresh co-compilation cache token, so compiled file checksums change. Validate regenerated outputs before updating the file fingerprints in `speech/amy_model.cc` and `models/sanotts/manifest.json`; those checks deliberately reject mismatched model versions.

The pinned Hugging Face model card declares **GPL-3.0**. Attribution, conversion provenance and the upstream licensing boundary are recorded in [NOTICE.md](NOTICE.md); the source license is preserved in [LICENSE](LICENSE).
