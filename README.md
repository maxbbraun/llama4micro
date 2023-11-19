# llama4micro 🦙🔬

A "large" language model running on a microcontroller.

![Example run](llama4micro.gif)

## Background

I was wondering if it's possible to fit a non-trivial language model on a microcontroller. Turns out the answer is some version of yes!

This project is using the [Coral Dev Board Micro](https://coral.ai/products/dev-board-micro) with its [FreeRTOS toolchain](https://coral.ai/docs/dev-board-micro/freertos/). The board has a number of neat [hardware features](https://coral.ai/docs/dev-board-micro/get-started/#the-hardware) not currently being used here (notably a [TPU](https://coral.ai/technology/), sensors, and a [second CPU core](https://coral.ai/docs/dev-board-micro/multicore/)). It does, however, also have 64MB of RAM. That's tiny for LLMs, which are typically measured in the GBs, but comparatively huge for a microcontroller. Inference runs on the 800 MHz [Arm Cortex-M7](https://developer.arm.com/Processors/Cortex-M7) CPU core.

The LLM implementation itself is an adaptation of [llama2.c](https://github.com/karpathy/llama2.c) and the [tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/main) checkpoints trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The quality of the smaller model versions isn't ideal, but good enough to generate somewhat coherent (and occasionally weird) stories.

## Setup

Clone this repo with its submodules [`karpathy/llama2.c`](https://github.com/karpathy/llama2.c) and [`google-coral/coralmicro`](https://github.com/google-coral/coralmicro):

```bash
git clone --recurse-submodules https://github.com/maxbbraun/llama4micro.git

cd llama4micro
```

Some of the tools use Python. Install their dependencies:

```bash
python3 -m venv venv
. venv/bin/activate

pip install -r llama2.c/requirements.txt
pip install -r coralmicro/scripts/requirements.txt

```

Download the model and quantize it:

```bash
LLAMA_MODEL_NAME=stories15M
wget -P data https://huggingface.co/karpathy/tinyllamas/resolve/main/${LLAMA_MODEL_NAME}.pt

python llama2.c/export.py data/${LLAMA_MODEL_NAME}_q80.bin --version 2 --checkpoint data/${LLAMA_MODEL_NAME}.pt

cp llama2.c/tokenizer.bin data/
```

Build and flash the image:

```bash
mkdir build
cd build

cmake ..
make -j

python ../coralmicro/scripts/flashtool.py --build_dir . --elf_path llama4micro
```

## Usage

1. The model loads automatically when the board powers up.
   - This takes ~6 seconds.
   - The green light will turn on when it's ready.
2. Press the button next to the green light.
   - The green light will turn off.
3. The model now generates tokens.
   - The results are streamed to the serial port.
   - This happens at a rate of ~2.5 tokens per second.
4. Generation stops after the end token or maximum steps.
   - The green light will turn on again.
   - Goto 2.
