ARG BASE_IMAGE=nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=chromie
ARG RUNTIME_EXTRA=runtime-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV PATH="/opt/venvs/runtime/bin:/home/${USERNAME}/.local/bin:/usr/local/bin:${PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

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

RUN set -eux; \
    if getent group "${GROUP_ID}" >/dev/null; then \
        echo "Using existing group with GID ${GROUP_ID}"; \
    else \
        groupadd --gid "${GROUP_ID}" "${USERNAME}"; \
    fi; \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
        usermod --uid "${USER_ID}" --gid "${GROUP_ID}" "${USERNAME}"; \
    elif getent passwd "${USER_ID}" >/dev/null; then \
        EXISTING_USER="$(getent passwd "${USER_ID}" | cut -d: -f1)"; \
        usermod --login "${USERNAME}" --home "/home/${USERNAME}" --move-home --gid "${GROUP_ID}" "$EXISTING_USER"; \
    else \
        useradd --uid "${USER_ID}" --gid "${GROUP_ID}" -m -s /bin/bash "${USERNAME}"; \
    fi; \
    mkdir -p "/home/${USERNAME}" /app /workspaces /opt/venvs/runtime /data; \
    chown -R "${USER_ID}:${GROUP_ID}" "/home/${USERNAME}" /app /workspaces /opt/venvs/runtime /data

COPY docker/runtime/bootstrap_runtime.sh /usr/local/bin/bootstrap_runtime
RUN chmod +x /usr/local/bin/bootstrap_runtime

COPY pyproject.toml README.md /app/
COPY configs /app/configs
COPY src /app/src
RUN chown -R ${USER_ID}:${GROUP_ID} /app /opt/venvs/runtime

USER ${USERNAME}
WORKDIR /app

# Build the runtime Python environment into the image. Source is still bind-mounted
# at runtime, so normal code edits do not require rebuilding the image.
RUN uv venv --python python3 /opt/venvs/runtime \
    && uv pip install --python /opt/venvs/runtime/bin/python --upgrade pip setuptools wheel \
    && uv pip install --python /opt/venvs/runtime/bin/python -e "/app[${RUNTIME_EXTRA},dev]" \
    && touch /opt/venvs/runtime/.soridormi_runtime_image_env

RUN echo 'alias runtime="source /opt/venvs/runtime/bin/activate"' >> ~/.bashrc \
    && echo 'echo "Runtime env is already built into this image. Use: runtime"' >> ~/.bashrc

CMD ["bash"]
