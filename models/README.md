## Models

This directory contains the pre-trained model weights and metadata. See instructions below about their origins.

Some of the tools use Python. Install their dependencies:

```bash
python3 -m venv venv
. venv/bin/activate

pip install -r llama2.c/requirements.txt
pip install -r yolov5/requirements.txt

```

### Llama

The model used by [llama2.c](https://github.com/karpathy/llama2.c) is based on [Llama 2](https://ai.meta.com/llama/) and the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The model weights are from the [tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/main) repository. We are using the [OG version](https://github.com/karpathy/llama2.c#models) (with 15M parameters) and quantize it. This model runs on the [Arm Cortex-M7 CPU](https://developer.arm.com/Processors/Cortex-M7).


```bash
LLAMA_MODEL_NAME=stories15M
LLAMA_MODEL_DIR=llama2

wget -P models/${LLAMA_MODEL_DIR} \
    https://huggingface.co/karpathy/tinyllamas/resolve/main/${LLAMA_MODEL_NAME}.pt

python llama2.c/export.py \
    models/${LLAMA_MODEL_DIR}/${LLAMA_MODEL_NAME}_q80.bin \
    --version 2 \
    --checkpoint models/${LLAMA_MODEL_DIR}/${LLAMA_MODEL_NAME}.pt
```

The tokenizer comes from the [llama2.c](https://github.com/karpathy/llama2.c) repository.

```bash
cp llama2.c/tokenizer.bin models/${LLAMA_MODEL_DIR}/
```

### Vision

Object detection (with labels used for prompting Llama) is based on [YOLOv5](https://github.com/ultralytics/yolov5), specifially the smallest version "n" at a 224x224 resolution. This model runs on the [Coral Edge TPU](https://coral.ai/technology/), which requires an additional compilation step (handled by the exporter).

```bash
git clone https://github.com/ultralytics/yolov5.git

YOLO_RESOLUTION=224
YOLO_VERSION=n
VISION_MODEL_NAME=yolov5${YOLO_VERSION}-int8_edgetpu
YOLO_MODEL_DIR=yolov5

python yolov5/export.py \
    --weights yolov5${YOLO_VERSION}.pt \
    --include edgetpu \
    --int8 \
    --img ${YOLO_RESOLUTION} \
    --data yolov5/data/coco128.yaml

mkdir models/${YOLO_MODEL_DIR}/
cp yolov5/${VISION_MODEL_NAME}.tflite models/${YOLO_MODEL_DIR}/
```

The labels are from the [COCO dataset](https://cocodataset.org/). Convert them to an easily readable format.

```bash
python models/export_coco_labels.py
```

### Speech

Speech uses [Amy Small](https://huggingface.co/ampixa/sanoTTS/tree/c532a5d21c078a16cb633718e9182bfd71a5b760/amy-en-1p1m), the 1.08M-parameter English voice from [sanoTTS](https://github.com/Ampixa/sanoTTS). Duration and acoustic inference run on the Cortex-M7; two int8 decoder graphs run on the Edge TPU. Output is mono 22,050 Hz. The three shipped files in `models/sanotts/` total about 2.59 MiB:

- `front_f32.bin`: duration/acoustic weights, widened losslessly from the upstream FP16 package.
- `amy_prefix_edgetpu.tflite`: decoder input `[1, 1, 32, 192]`, output `[1, 1, 2048, 40]`.
- `amy_tail_edgetpu.tflite`: decoder input `[1, 1, 666, 40]`, output `[1, 1, 2664, 1]`, followed by CPU `tanh`.

See [regeneration instructions](sanotts/README.md) for the pinned source, calibration and compiler workflow. Model fingerprints and source revisions are in [sanotts/manifest.json](sanotts/manifest.json).

The pinned Hugging Face model card declares **GPL-3.0**. See [provenance and notices](sanotts/NOTICE.md) and the preserved [source license](sanotts/LICENSE).
