#!/bin/bash
# Download DeepSeek-V2-Lite model files from Hugging Face

REPO="deepseek-ai/DeepSeek-V2-Lite"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"
LOCAL_DIR="deepseek-ai/DeepSeek-V2-Lite"

cd "$(dirname "$0")"

# Files to download
FILES=(
    "model-00001-of-000004.safetensors"
    "model-00002-of-000004.safetensors"
    "model-00003-of-000004.safetensors"
    "model-00004-of-000004.safetensors"
)

echo "Downloading DeepSeek-V2-Lite model files..."
echo "This will download ~31GB of data. Make sure you have enough disk space."
echo ""

for file in "${FILES[@]}"; do
    echo "Downloading ${file}..."
    curl -L "${BASE_URL}/${file}" -o "${LOCAL_DIR}/${file}" --progress-bar --continue-at -
    if [ $? -eq 0 ]; then
        echo "✓ ${file} downloaded successfully"
    else
        echo "✗ Failed to download ${file}"
        exit 1
    fi
done

echo ""
echo "All model files downloaded successfully!"

