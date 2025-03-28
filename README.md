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

You can also run this app using Docker

```shell
docker build -t probing-vis-literacy .
docker run -p 7860:7860 probing-vis-literacy
```

## Citation
