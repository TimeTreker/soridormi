ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=duck

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV PATH="/opt/venvs/runtime/bin:/home/${USERNAME}/.local/bin:/usr/local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
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
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN groupadd --gid ${GROUP_ID} ${USERNAME} \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} -m -s /bin/bash ${USERNAME} \
    && mkdir -p /app /workspaces /opt/venvs/runtime /data \
    && chown -R ${USERNAME}:${USERNAME} /app /workspaces /opt/venvs/runtime /data

COPY docker/runtime/bootstrap_runtime.sh /usr/local/bin/bootstrap_runtime
RUN chmod +x /usr/local/bin/bootstrap_runtime

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN chown -R ${USERNAME}:${USERNAME} /app

USER ${USERNAME}
WORKDIR /app

RUN echo 'alias runtime="source /opt/venvs/runtime/bin/activate"' >> ~/.bashrc \
    && echo 'echo "Run bootstrap_runtime first. Then use: runtime"' >> ~/.bashrc

CMD ["bash"]
