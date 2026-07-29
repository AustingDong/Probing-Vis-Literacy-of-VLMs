---
title: Probing-Vis-Literacy-of-VLMs
emoji: 📊
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
---

# Probing Vis Literacy of Vision Language Models: the Good, the Bad, and the Ugly

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/uw-insight-lab/Probing-Vis-Literacy-of-VLMs)

![Visualization literacy example](images/examples/placeholder.png)

An interactive Gradio application for probing the visualization literacy of
vision-language models.

## Quick start

### Installation

Python 3.10 to 3.12 is supported.

```shell
python -m pip install --no-cache-dir --user -e ".[gradio]"
```

### Run the Gradio app

```shell
python app.py
```

Open <http://localhost:7860>.

## Docker GPU runtimes

NVIDIA CUDA and AMD ROCm use separate, pinned images and Compose entry points.
Both run a real GPU kernel before starting Gradio and exit early when the
requested accelerator is unavailable or the wrong image was selected.

### NVIDIA

The NVIDIA image uses PyTorch 2.9.1 with CUDA 13.0 for Blackwell/RTX 50-series
compatibility. It requires a compatible NVIDIA driver and NVIDIA GPU access in
Docker. Docker Desktop must use its WSL2 backend on Windows.

```shell
docker compose -f compose.nvidia.yaml build
docker compose -f compose.nvidia.yaml up
```

Validate NVIDIA passthrough independently:

```shell
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

### AMD

The AMD image uses AMD's PyTorch 2.9.1 / ROCm 7.2.4 image, which includes
`gfx950` support for Instinct MI350/MI355 accelerators. It requires a supported
Linux x86-64 host with the AMD kernel driver; Windows Docker Desktop is not a
supported ROCm host.

```shell
docker compose -f compose.amd.yaml build
docker compose -f compose.amd.yaml up
```

Validate AMD passthrough independently:

```shell
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  python -c "import torch; print(torch.cuda.is_available(), torch.version.hip, torch.cuda.get_device_name(0))"
```

Open <http://localhost:7860> after either stack starts. Model downloads and
results use named volumes; the Hugging Face cache and results are shared
between the two entry points, while the PyTorch cache is backend-specific.

## Citation

Citation information will be added with the accompanying publication.
