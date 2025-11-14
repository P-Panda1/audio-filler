#!/usr/bin/env bash
# Full automation: clone repo (if not present), pull runpod docker image, and run container that executes training orchestrator.
# Usage: ssh into runpod and run this script: bash deploy_and_train.sh
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
ENV_FILE="$ROOT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo ".env not found. Copy .env.example to .env and edit it (or run scripts/create_env.sh)"
  exit 1
fi

# shellcheck source=/dev/null
. "$ENV_FILE"

# Expand tilde in SERVICE_ACCOUNT_HOST_PATH if present
SERVICE_ACCOUNT_HOST_PATH_EXPANDED=$(eval echo "$SERVICE_ACCOUNT_HOST_PATH")

# Ensure workspace exists
mkdir -p "$WORKSPACE_MOUNT"
mkdir -p "$MLFLOW_MOUNT_DIR"

# Clone repo if missing
if [ ! -d "$GIT_CLONE_DIR" ]; then
  echo "Cloning $GIT_REPO -> $GIT_CLONE_DIR"
  git clone "$GIT_REPO" "$GIT_CLONE_DIR"
else
  echo "Repo exists at $GIT_CLONE_DIR. Pulling latest changes."
  (cd "$GIT_CLONE_DIR" && git pull)
fi

# Pull runpod image
echo "Pulling runpod image: $RUNPOD_IMAGE"
docker pull "$RUNPOD_IMAGE"

# Build helper image from repo if you want repository-packaged image
# We will run a container based on runpod image, then mount workspace and run pip install from repo.

CONTAINER_NAME=audio_filler_run_$(date +%s)

# Run the container and execute training inside it
# We mount the repo into /workspace/audio-filler and mount the service account and mlruns directory

docker run --rm -it \
  --name "$CONTAINER_NAME" \
  --gpus all \
  -e GOOGLE_APPLICATION_CREDENTIALS="$SERVICE_ACCOUNT_CONTAINER_PATH" \
  -v "$SERVICE_ACCOUNT_HOST_PATH_EXPANDED:$SERVICE_ACCOUNT_CONTAINER_PATH:ro" \
  -v "$GIT_CLONE_DIR:$WORKSPACE_MOUNT:rw" \
  -v "$MLFLOW_MOUNT_DIR:/app/mlruns:rw" \
  -p "$MLFLOW_PORT:5000" \
  "$RUNPOD_IMAGE" \
  bash -lc "set -euo pipefail; cd $WORKSPACE_MOUNT; pip install --upgrade pip; pip install -r requirements.txt; python tools/orchestrate_training.py --gcs_bucket \"$GCS_BUCKET\" --configs $CONFIGS --epochs $EPOCHS --batch_size $BATCH_SIZE $( [ "$RUN_LOCAL_MLFLOW" = "true" ] && echo --run_local_mlflow || true )"

echo "Training finished (container exited). MLflow data in $MLFLOW_MOUNT_DIR"

*** End Patch