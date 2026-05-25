ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=duck

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV PATH="/opt/venvs/sim/bin:/home/${USERNAME}/.local/bin:/usr/local/bin:${PATH}"
ENV MUJOCO_GL=glfw
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    htop \
    less \
    nano \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    vim \
    wget \
    libegl1 \
    libgl1 \
    libgles2 \
    libglvnd0 \
    libglx0 \
    libglu1-mesa \
    libglew2.2 \
    libglfw3 \
    libosmesa6 \
    libglib2.0-0 \
    libsm6 \
    libusb-1.0-0 \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN groupadd --gid ${GROUP_ID} ${USERNAME} \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} -m -s /bin/bash ${USERNAME} \
    && mkdir -p /app /workspaces /opt/venvs/sim /data \
    && chown -R ${USERNAME}:${USERNAME} /app /workspaces /opt/venvs/sim /data

COPY docker/simulator/bootstrap_simulator.sh /usr/local/bin/bootstrap_simulator
RUN chmod +x /usr/local/bin/bootstrap_simulator

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN chown -R ${USERNAME}:${USERNAME} /app

USER ${USERNAME}
WORKDIR /app

RUN echo 'alias sim="source /opt/venvs/sim/bin/activate"' >> ~/.bashrc \
    && echo 'echo "Run bootstrap_simulator first. Then use: sim"' >> ~/.bashrc

CMD ["bash"]
