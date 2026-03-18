ARG UBUNTU_VERSION=22.04
ARG TARGET_PLATFORM=x86_64
ARG CUDA_VERSION=12.4.0
ARG CUDA_VERSION_PATH=cu124
ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

#########################################################################
# Build image
#########################################################################

FROM ${DEVEL_BASE_IMAGE} as build

WORKDIR /app/build

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        gcc \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda, Python, and Python build dependencies.
ARG TARGET_PLATFORM
ARG PYTHON_VERSION
ENV PATH /opt/conda/bin:$PATH
RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh"
# NOTE: Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja && \
    /opt/conda/bin/conda clean -ya

# Install PyTorch core ecosystem.
ARG CUDA_VERSION_PATH
ARG TORCH_VERSION=2.6.0
ARG INSTALL_CHANNEL=whl/test
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchao torchvision torchaudio

ENV TORCH_CUDA_ARCH_LIST="8.0 9.0"

# Install grouped-gemm.
# NOTE: right now we need to build with CUTLASS so we can pass batch sizes on GPU.
# See https://github.com/tgale96/grouped_gemm/pull/21
ENV GROUPED_GEMM_CUTLASS="1"
ARG GROUPED_GEMM_VERSION="grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@main"
RUN pip install --no-build-isolation --no-cache-dir "${GROUPED_GEMM_VERSION}"

# Install flash-attn.
ARG FLASH_ATTN_VERSION=2.7.4.post1
RUN pip install --no-build-isolation --no-cache-dir flash-attn==${FLASH_ATTN_VERSION}

# Install ring-flash-attn.
ARG RING_FLASH_ATTN_VERSION=0.1.4
RUN pip install --no-build-isolation --no-cache-dir ring-flash-attn==${RING_FLASH_ATTN_VERSION}

# Install liger-kernel.
ARG LIGER_KERNEL_VERSION=0.5.4
RUN pip install --no-build-isolation --no-cache-dir liger-kernel==${LIGER_KERNEL_VERSION}

# Install direct dependencies, but not source code.
COPY pyproject.toml .
COPY olmo/__init__.py olmo/__init__.py
COPY olmo/version.py olmo/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ai2-molmoact && \
    rm -rf *

# Install transformers==4.52 for MolmoAct.
ARG TRANSFORMERS_VERSION=4.52.3
RUN pip install --no-cache-dir transformers==${TRANSFORMERS_VERSION}

# Install vllm for MolmoAct.
ARG VLLM_VERSION=0.8.5
RUN pip install --no-cache-dir vllm==${VLLM_VERSION}


# Install torchao.
ARG TORCH_CUDA_VERSION=124
ARG TORCHAO_VERSION=0.10.0
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION} \
    torchao==${TORCHAO_VERSION}

#########################################################################
# Release image
#########################################################################

FROM ${BASE_IMAGE} as release

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        language-pack-en \
        make \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        rsync \
        vim \
        sudo \
        unzip \
        fish \
        parallel \
        zsh \
        htop \
        tmux \
        wget \
        emacs \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        apt-transport-https \
        gnupg \
        jq \
        gcc \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install MLNX OFED user-space drivers
# See https://docs.nvidia.com/networking/pages/releaseview.action?pageId=15049785#Howto:DeployRDMAacceleratedDockercontaineroverInfiniBandfabric.-Dockerfile
ARG UBUNTU_VERSION
ARG TARGET_PLATFORM
ENV MOFED_VER="24.01-0.3.3.1"
RUN wget --quiet https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}/mlnxofedinstall --basic --user-space-only --without-fw-update -q && \
    rm -rf MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM} && \
    rm MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}.tgz

# Copy conda environment.
COPY --from=build /opt/conda /opt/conda

ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

# aws cli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
 unzip awscliv2.zip && \
 sudo ./aws/install && \
 rm -rf aws

# gsutil/gcloud
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
 echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
 sudo apt-get update && sudo apt-get -y install google-cloud-cli

# Install a few additional utilities via pip
RUN /opt/conda/bin/pip install --no-cache-dir \
    gpustat \
    jupyter \
    beaker-gantry \
    oocmap

# Use bash for RUN
SHELL ["/bin/bash", "-c"]

# LABEL org.opencontainers.image.source https://github.com/allenai/OLMo-core
WORKDIR /app/olmo-core
