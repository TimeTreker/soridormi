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
COPY configs /app/configs
COPY src /app/src
RUN pip install --no-cache-dir -e ".[mcp]"

USER ${USERNAME}

EXPOSE 8000

CMD ["python", "-m", "soridormi_runtime.mcp.http_server", "--host", "0.0.0.0", "--port", "8000"]
