# Probing Vis Literacy of Vision Language Models: the Good, the Bad, and the Ugly

![GitHub License](https://img.shields.io/github/license/AustingDong/Probing-Vis-Literacy-of-Vision-Language-Models)
![image](https://github.com/AustingDong/Probing-Vis-Literacy-of-Vision-Language-Models/examples/placeholder.png)

## Quick Start

### Installation

```shell
pip install --no-cache-dir --user -e .
pip install --no-cache-dir --user opencv-python
pip install --no-cache-dir --user -r /code/requirements-gradio.txt
```

### Run the Gradio App

```shell
gradio app.py
```

### Docker

You can also run this app using either the NVIDIA CUDA or AMD ROCm Docker
entry point:

```shell
docker compose -f compose.nvidia.yaml up --build
# or, on a supported Linux ROCm host:
docker compose -f compose.amd.yaml up --build
```

See [README.md](README.md#docker-gpu-runtimes) for host requirements and GPU
passthrough checks.

## Citation
