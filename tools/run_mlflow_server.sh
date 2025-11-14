#!/usr/bin/env bash
# Simple launcher to run mlflow server in the container.
set -euo pipefail
ROOT_DIR="/app"
MLRUNS_DIR="$ROOT_DIR/mlruns"
mkdir -p "$MLRUNS_DIR"
exec mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root "$MLRUNS_DIR"
