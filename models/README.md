## Models

This directory contains the pre-trained model weights and metadata. See instructions below about their origins.

Some of the tools use Python. Install their dependencies:

```bash
python3.12 -m venv venv
. venv/bin/activate

pip install -r llama2/requirements.txt
pip install -r yolov8/requirements.txt
```

[Install](https://coral.ai/docs/edgetpu/compiler/#download) Edge TPU Compiler 14.1.317412892.

### Llama

The model used by [llama2.c](https://github.com/karpathy/llama2.c) is based on [Llama 2](https://ai.meta.com/llama/) and the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The model weights are from the [tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/main) repository. We are using the [OG version](https://github.com/karpathy/llama2.c#models) (with 15M parameters) and quantize it. This model runs on the [Arm Cortex-M7 CPU](https://developer.arm.com/Processors/Cortex-M7).


```bash
LLAMA_MODEL_NAME=stories15M
LLAMA_MODEL_DIR=llama2

wget -P ${LLAMA_MODEL_DIR} \
    https://huggingface.co/karpathy/tinyllamas/resolve/main/${LLAMA_MODEL_NAME}.pt

python ../third_party/llama2.c/export.py \
    ${LLAMA_MODEL_DIR}/${LLAMA_MODEL_NAME}_q80.bin \
    --version 2 \
    --checkpoint ${LLAMA_MODEL_DIR}/${LLAMA_MODEL_NAME}.pt
```

The tokenizer comes from the [llama2.c](https://github.com/karpathy/llama2.c) repository.

```bash
cp ../third_party/llama2.c/tokenizer.bin ${LLAMA_MODEL_DIR}/
```

### Vision

Object detection (with labels used for prompting Llama) uses [YOLOv8n](https://github.com/ultralytics/ultralytics), the smallest (nano) variant, at a 224x224 resolution with the 80 [COCO](https://cocodataset.org/) classes. The network runs on the [Coral Edge TPU](https://coral.ai/technology/); box decoding and non-maximum suppression run on the Arm Cortex-M7 CPU.

Export the [pretrained weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt):

```bash
mkdir -p ../build/yolov8
wget -O ../build/yolov8/yolov8n.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
wget -O ../build/yolov8/coco128.zip \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
python -m zipfile -e ../build/yolov8/coco128.zip ../build/yolov8/calibration

python export_yolov8.py
cd ../build/yolov8
TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=4 \
python -m onnx2tf \
    -i yolov8n.onnx -o tflite -oiqt \
    -cind images calibration.npy 0 1 \
    -iqd uint8 -oqd uint8 -v warn
mkdir -p edgetpu
edgetpu_compiler --show_operations --out_dir edgetpu \
    tflite/yolov8n_full_integer_quant.tflite
cd ../../models
python -c 'from export_yolov8 import install_model; install_model()'
```

### Speech

Speech uses the [heartnano](https://huggingface.co/ampixa/sanoTTS/tree/main/heartnano) voice from [sanoTTS](https://github.com/Ampixa/sanoTTS), running on the Arm Cortex-M7 CPU at a 24 kHz sample rate. The two weight files are copied from the pinned submodule.

```bash
mkdir -p sanotts
cp ../third_party/sanoTTS/web/voices/heartnano/front_q8.bin sanotts/
cp ../third_party/sanoTTS/web/voices/heartnano/model_q8.bin sanotts/
```
