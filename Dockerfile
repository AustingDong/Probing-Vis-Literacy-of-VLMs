# PyTorch 2.9.1 + CUDA 13.0 includes Blackwell (sm_120) support required by
# GeForce RTX 50-series GPUs. Pin the digest so rebuilds use the same runtime.
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime@sha256:60f22fb80755fd0b470fb47928dbd55816aa9f847edd95cf43c93253507a9ddf

ARG APP_UID=1000
ARG APP_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:/opt/conda/bin:$PATH \
    PYTHONPATH=/home/user/app \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SHARE=false \
    REQUIRE_CUDA=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        fonts-dejavu-core \
        libglib2.0-0 \
        libgomp1 \
    && groupadd --gid "${APP_GID}" user \
    && useradd --create-home --uid "${APP_UID}" --gid "${APP_GID}" --shell /bin/bash user \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/user/app

# Install pinned dependencies before copying the source to preserve the build
# cache when only application code changes.
COPY requirements.txt requirements-gradio.txt ./
RUN python -m pip install --no-cache-dir \
        --requirement requirements.txt \
        --requirement requirements-gradio.txt

COPY --chown=user:user . .

RUN mkdir -p \
        /home/user/.cache/huggingface \
        /home/user/.cache/torch \
        /home/user/.cache/matplotlib \
        /home/user/app/results \
    && python -c "import cv2, gradio, spaces, torch, torchvision, transformers; assert torch.__version__.startswith('2.9.1'); assert torch.version.cuda == '13.0'; assert transformers.__version__ == '4.48.2'" \
    && chown -R user:user /home/user/.cache /home/user/app/results

EXPOSE 7860

# Fail fast when the container was started without GPU access, then replace the
# checker process with the application so signals reach Gradio correctly. The
# checker starts as root only long enough to prepare mounted volumes, then drops
# to the unprivileged "user" account before touching models or running Gradio.
CMD ["python", "docker/verify_runtime.py", "--", "python", "app.py"]
