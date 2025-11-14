#!/usr/bin/env bash
# Create a .env file from .env.example and prompt for secret values interactively for sensitive entries.
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
EXAMPLE="$ROOT_DIR/.env.example"
TARGET="$ROOT_DIR/.env"

if [ ! -f "$EXAMPLE" ]; then
  echo "Missing .env.example"
  exit 1
fi

cp "$EXAMPLE" "$TARGET"

echo "Populating .env. Press enter to accept default values shown in brackets."

# read current .env and prompt for SERVICE_ACCOUNT_HOST_PATH and GIT_REPO
read -p "Path to service account on host [$(grep SERVICE_ACCOUNT_HOST_PATH $EXAMPLE | cut -d'=' -f2)]: " svc
if [ -n "$svc" ]; then
  sed -i.bak "s|SERVICE_ACCOUNT_HOST_PATH=.*|SERVICE_ACCOUNT_HOST_PATH=$svc|" "$TARGET"
fi

read -p "GCS bucket (gs://... ) [$(grep GCS_BUCKET $EXAMPLE | cut -d'=' -f2)]: " gcs
if [ -n "$gcs" ]; then
  sed -i.bak "s|GCS_BUCKET=.*|GCS_BUCKET=$gcs|" "$TARGET"
fi

read -p "Git repo to clone [$(grep GIT_REPO $EXAMPLE | cut -d'=' -f2)]: " repo
if [ -n "$repo" ]; then
  sed -i.bak "s|GIT_REPO=.*|GIT_REPO=$repo|" "$TARGET"
fi

# cleanup
rm -f "$TARGET.bak"

echo ".env written to $TARGET"

*** End Patch