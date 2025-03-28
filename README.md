<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
    <h1>Probing Vis Literacy of Vision Language Models: the Good, the Bad, and the Ugly</h1>
</div>

<div align="center">
    <a href="https://www.uw-insight-lab.com/" target="_blank">
        <img src="https://img.shields.io/badge/UW Insight Lab-Homepage-blue" alt="UW Insight Lab" />
    </a>
    <a href="https://huggingface.co/uw-insight-lab" target="_blank">
        <img src="https://img.shields.io/badge/HuggingFace-UW Insight Lab-yellow?logo=huggingface" alt="Hugging Face" />
    </a>
    <a href="https://github.com/AustingDong/Probing-Vis-Literacy-of-Vision-Language-Models/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/github/license/AustingDong/Probing-Vis-Literacy-of-Vision-Language-Models" alt="GitHub License" />
    </a>
    <a href="https://github.com/AustingDong/Probing-Vis-Literacy-of-Vision-Language-Models/stargazers" target="_blank">
        <img src="https://img.shields.io/github/stars/AustingDong/Probing-Vis-Literacy-of-Vision-Language-Models?style=social" alt="GitHub Stars"/>
    </a>
</div>

<div align="center">
    <img src="images/result_examples/chart_types_horizontal.png" alt="Example Preview" />
</div>

## 🚀 Quick Start

### ⚙️ Setup your local environment

#### 📦 Install Dependencies

```shell
pip install --no-cache-dir --user -e .
pip install --no-cache-dir --user opencv-python
pip install --no-cache-dir --user -r /code/requirements-gradio.txt
```

#### 🖼️ Launch the Gradio App

```shell
gradio app.py
```

### 🐋 Run with Docker (No Setup Required)

You can also build and run the app in an isolated Docker container:

```shell
docker build -t probing-vis-literacy .
docker run -p 7860:7860 probing-vis-literacy
```

## Citation
