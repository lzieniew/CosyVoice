FROM ubuntu:22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y --fix-missing && \
    apt-get install -y git build-essential curl wget ffmpeg unzip git-lfs sox libsox-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Install miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH /opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /workspace

ENV PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"
# PyTorch ships its own CUDA runtime, so the NVIDIA base image is not needed.
# The NVIDIA Container Toolkit (--gpus) exposes the host GPU driver at runtime.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5
RUN conda activate ${VENV} && cd CosyVoice && \
    pip install -r requirements.txt --no-cache-dir

WORKDIR /workspace/CosyVoice

COPY docker/generate.py /workspace/CosyVoice/generate.py

CMD ["conda", "run", "--no-capture-output", "-n", "cosyvoice", "python", "generate.py"]
