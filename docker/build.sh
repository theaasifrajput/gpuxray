#!/usr/bin/env bash
set -euo pipefail

docker build \
    -f docker/Dockerfile.cuda12.8 \
    -t gpuxray-dev:cuda12.8 \
    .
