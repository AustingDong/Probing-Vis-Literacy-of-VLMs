---
title: Probing-Vis-Literacy-of-VLMs
emoji: 🐨
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at <https://huggingface.co/docs/hub/spaces-config-reference>

## Docker GPU runtime

The container is pinned to PyTorch 2.9.1 with CUDA 13.0 for NVIDIA
Blackwell/RTX 50-series compatibility.

Requirements:

- NVIDIA driver compatible with CUDA 13.0
- Docker Desktop with the WSL2 backend on Windows
- NVIDIA GPU access enabled in Docker

Build and run:

```shell
docker compose build
docker compose up
```

Open <http://localhost:7860>. The entry point executes a CUDA kernel before
starting Gradio and exits with an actionable error if the GPU is unavailable.

To validate Docker GPU passthrough independently:

```shell
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```
