<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
    <h1>Probing Vis Literacy of Vision Language Models: the Good, the Bad, and the Ugly</h1>
</div>

<div align="center">
    <a href="https://huggingface.co/uw-insight-lab" target="_blank">
        <img src="https://img.shields.io/badge/HuggingFace-UW Insight Lab-yellow?logo=huggingface" alt="Hugging Face" />
    </a>
    <a href="https://github.com/AustingDong/Probing-Vis-Literacy-of-Vision-Language-Models/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/github/license/AustingDong/Probing-Vis-Literacy-of-Vision-Language-Models" alt="GitHub License" />
    </a>
</div>

<div align="center">
    <img src="images/result_examples/chart_types_horizontal.png" alt="Example Preview" />
</div>

## Quick Start

### Installation

```shell
pip install --no-cache-dir --user -e .
pip install --no-cache-dir --user opencv-python
pip install --no-cache-dir --user -r /code/requirements-gradio.txt
```

### Gradio App

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
