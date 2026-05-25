#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cat > .env <<EOF
UID=$(id -u)
GID=$(id -g)
RUNTIME_DEV_BASE=ubuntu:24.04
ORIN_RUNTIME_BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
THOR_RUNTIME_BASE=nvidia/cuda:13.0.2-devel-ubuntu24.04
SIM_BASE=nvidia/cuda:12.8.1-devel-ubuntu24.04
SIM_HOST=127.0.0.1
SIM_PORT=5555
CONTROL_HZ=50
EOF

echo "Created .env"
