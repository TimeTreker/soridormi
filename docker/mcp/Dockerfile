FROM python:3.12-slim

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=chromie

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN groupadd --gid "${GROUP_ID}" "${USERNAME}" \
    && useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home "${USERNAME}"

WORKDIR /app
COPY pyproject.toml README.md /app/
# Dependencies come from pyproject.toml alone, so a source change reuses them; the
# project is then installed editable without dependencies. Downloads stay in a
# BuildKit cache mount outside the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c 'import tomllib; p = tomllib.load(open("pyproject.toml", "rb"))["project"]; print("\n".join(p["dependencies"] + p["optional-dependencies"]["mcp"]))' > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
COPY configs /app/configs
COPY src /app/src
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -e .

USER ${USERNAME}

EXPOSE 8000

CMD ["python", "-m", "soridormi_runtime.mcp.http_server", "--host", "0.0.0.0", "--port", "8000"]
