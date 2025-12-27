#!/usr/bin/env bash
set -e

# -----------------------------------------------------
# Load .env from project root
# -----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

echo "Loading environment from $ENV_FILE"
set -o allexport
source "$ENV_FILE"
set +o allexport


# -----------------------------------------------------
# Send service account to pod
# -----------------------------------------------------
echo "Transferring service account to pod..."
scp -P "$POD_PORT" -i "$SSH_KEY" "$GCP_SERVICE_ACCOUNT_PATH" \
    root@"$POD_HOST":/root/service_account.json

# -----------------------------------------------------
# Prepare Boolean Flags
# -----------------------------------------------------
# If the env var is "true", set the flag string. Otherwise leave it empty.
if [ "$UPLOAD_BEST_TO_GCS" = "true" ]; then
    UPLOAD_ARG="--upload-best-to-gcs"
else
    UPLOAD_ARG=""
fi

# -----------------------------------------------------
# SSH and run training
# -----------------------------------------------------
echo "Running remote setup + training..."

ssh -p "$POD_PORT" -i "$SSH_KEY" root@"$POD_HOST" << EOF
set -e

apt update && apt install -y git

# This installs FFmpeg system-wide on the pod
apt install -y ffmpeg

apt remove -y python3-blinker || true

if [ ! -d "$REPO_NAME" ]; then
    git clone "$GITHUB_REPO_URL"
fi

cd "$REPO_NAME"

git checkout train
git pull origin train

pip install -r requirements.txt --break-system-packages



export GOOGLE_APPLICATION_CREDENTIALS=/root/service_account.json

export PYTHONPATH=.

# -----------------------------------------------------
# Download dataset if not present
# -----------------------------------------------------
if [ ! -d "$LOCAL_DATA_DIR/audio-fill/music" ]; then
    echo "Downloading dataset from GCS..."
    python3 tools/download_gcs_music.py \
        --gcs-bucket "$DATA_BUCKET" \
        --prefix music \
        --out-dir "$LOCAL_DATA_DIR"
else
    echo "Dataset already present at $LOCAL_DATA_DIR/audio-fill/music — skipping download."
fi

gsutil cp gs://model_log/trained_models/large_model/best_model.pt ./best_model.pt


python3 tools/train_large_model.py \
    --data-bucket "$DATA_BUCKET" \
    --model-bucket "$MODEL_BUCKET" \
    --service-account /root/service_account.json \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --val-split "$VAL_SPLIT" \
    --clip-duration "$CLIP_DURATION" \
    --sample-rate "$SAMPLE_RATE" \
    --gcs-dest-prefix "$GCS_DEST_PREFIX" \
    --log-dir "$LOG_DIR" \
    --local-data-dir "$LOCAL_DATA_DIR" \
    $UPLOAD_ARG
EOF
