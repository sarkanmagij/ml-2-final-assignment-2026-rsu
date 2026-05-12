#!/usr/bin/env bash
# Usage: bash push_to_hf.sh <hf-username> <space-name>
# Example: bash push_to_hf.sh tusama yelp-restaurant-intelligence
set -e

HF_USER=${1:?Usage: bash push_to_hf.sh <hf-username> <space-name>}
SPACE_NAME=${2:?Usage: bash push_to_hf.sh <hf-username> <space-name>}
REPO="$HF_USER/$SPACE_NAME"
SPACE_DIR="/tmp/hf_space_$SPACE_NAME"

echo "==> Creating Space: $REPO"
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$SPACE_NAME', repo_type='space', space_sdk='docker', exist_ok=True)
print('Space ready.')
"

echo "==> Cloning Space repo"
rm -rf "$SPACE_DIR"
git clone "https://huggingface.co/spaces/$REPO" "$SPACE_DIR"
cd "$SPACE_DIR"

echo "==> Enabling Git LFS"
git lfs install
git lfs track "*.pth"
git lfs track "*.pkl"
git add .gitattributes

echo "==> Copying app files"
PROJECT_DIR="$(dirname "$0")"
cp "$PROJECT_DIR/app.py" .
cp "$PROJECT_DIR/requirements.txt" .
cp "$PROJECT_DIR/Dockerfile" .
cp "$PROJECT_DIR/HF_README.md" README.md
cp -r "$PROJECT_DIR/demo_photos" .
cp -r "$PROJECT_DIR/models" .

echo "==> Staging files"
git add .

echo "==> Committing"
git commit -m "Initial Space deployment: CNN + BiLSTM Yelp intelligence app"

echo "==> Pushing (LFS objects first, then commit)"
git push

echo ""
echo "✅ Done! View your Space at:"
echo "   https://huggingface.co/spaces/$REPO"
