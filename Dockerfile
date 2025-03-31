FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

COPY ./requirements-gradio.txt /code/requirements-gradio.txt

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install system dependencies and create user
RUN apt-get update && apt-get install -y --no-install-recommends \
    && useradd -m -u 1000 user \
    && rm -rf /var/lib/apt/lists/*

# Install OpenGL and other dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch to "user" before installing dependencies
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces

WORKDIR $HOME/app

# Copy project files as "user" before installing dependencies
COPY --chown=user . $HOME/app
COPY --chown=user ./images /home/user/app/images

# Install dependencies as "user"
RUN pip install --no-cache-dir --user -e .
RUN pip install --no-cache-dir --user -r /code/requirements-gradio.txt

CMD ["python", "app.py"]
