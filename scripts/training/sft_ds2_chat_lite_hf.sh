#!/bin/bash
# Fine-tune DeepSeek-V2-Lite-Chat using LoRA with HuggingFace backend
# This script uses standard HuggingFace PEFT (no ktransformers)

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: training -> scripts -> project root

# Default values
CONFIG_FILE=""
CONDA_ENV="Kllama"
DRY_RUN=false
DATA_PATH=""
SKIP_CONFIRM=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --env|-e)
            CONDA_ENV="$2"
            shift 2
            ;;
        --data|-d)
            DATA_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --yes|-y)
            SKIP_CONFIRM=true
            shift
            ;;
        --help|-h)
            cat << EOF
Usage: $0 [OPTIONS]

Fine-tune DeepSeek-V2-Lite-Chat using LoRA with HuggingFace backend.

Options:
  -c, --config PATH    Path to training config file (default: examples/train_lora/deepseek2_lite_sft_hf.yaml)
  -e, --env NAME      Conda environment name (default: Kllama)
  -d, --data PATH     Path to custom training data (JSONL or JSON). Will convert and register automatically.
  -y, --yes           Skip confirmation prompt and start training immediately
  --dry-run           Show what would be executed without running
  -h, --help          Show this help message

Examples:
  # Use default config
  $0
  
  # Use custom config
  $0 --config examples/train_lora/my_custom_config.yaml
  
  # Use custom training data
  $0 --data sft_data/outputs/my_data.jsonl
  
  # Dry run to see what would be executed
  $0 --dry-run

Description:
  This script fine-tunes DeepSeek-V2-Lite-Chat using LoRA with the HuggingFace backend.
  The trained adapter will be in standard HuggingFace PEFT format and can be used
  directly with HuggingFace inference (no conversion needed).

  Training output will be saved to: saves/Kllama_deepseekV2Lite_hf_trained/
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default config if not provided
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="$PROJECT_ROOT/LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf.yaml"
fi

# Convert to absolute path
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

echo "=========================================="
echo "DeepSeek-V2-Lite-Chat SFT Training"
echo "Backend: HuggingFace (Standard PEFT)"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
if conda env list | grep -q "^${CONDA_ENV} "; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "✓ Successfully activated $CONDA_ENV environment"
    PYTHON_PATH=$(conda run -n "$CONDA_ENV" which python)
    PYTHON_VERSION=$(conda run -n "$CONDA_ENV" python --version)
    echo "  Python: $PYTHON_PATH"
    echo "  Python version: $PYTHON_VERSION"
else
    echo "Error: Conda environment '$CONDA_ENV' not found"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo ""

# Check if LLaMA-Factory directory exists
if [ ! -d "$PROJECT_ROOT/LLaMA-Factory" ]; then
    echo "Error: LLaMA-Factory directory not found at $PROJECT_ROOT/LLaMA-Factory"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Training config file not found at $CONFIG_FILE"
    exit 1
fi

echo "Config file: $CONFIG_FILE"
echo ""

# Handle custom data path if provided
if [ -n "$DATA_PATH" ]; then
    echo "Processing custom training data: $DATA_PATH"
    
    # Convert to absolute path
    if [[ "$DATA_PATH" != /* ]]; then
        DATA_PATH="$PROJECT_ROOT/$DATA_PATH"
    fi
    
    if [ ! -f "$DATA_PATH" ]; then
        echo "Error: Data file not found at $DATA_PATH"
        exit 1
    fi
    
    # Determine dataset name from filename
    DATA_BASENAME=$(basename "$DATA_PATH" .jsonl)
    DATA_BASENAME=$(basename "$DATA_BASENAME" .json)
    DATASET_NAME="${DATA_BASENAME}"
    DATASET_JSON="$PROJECT_ROOT/LLaMA-Factory/data/${DATASET_NAME}.json"
    DATASET_INFO="$PROJECT_ROOT/LLaMA-Factory/data/dataset_info.json"
    
    echo "  Converting to JSON format..."
    
    # Convert JSONL to JSON if needed
    if [[ "$DATA_PATH" == *.jsonl ]]; then
        conda run -n "$CONDA_ENV" python << PYEOF
import json
import sys

samples = []
with open("$DATA_PATH", 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            sample = {
                "instruction": data.get("instruction", ""),
                "input": data.get("input", ""),
                "output": data.get("output", "")
            }
            samples.append(sample)

with open("$DATASET_JSON", 'w') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"✅ Converted {len(samples)} samples")
PYEOF
    else
        # Copy JSON file
        cp "$DATA_PATH" "$DATASET_JSON"
    fi
    
    # Register in dataset_info.json
    echo "  Registering dataset in dataset_info.json..."
    conda run -n "$CONDA_ENV" python << PYEOF
import json

with open("$DATASET_INFO", 'r') as f:
    dataset_info = json.load(f)

dataset_info["$DATASET_NAME"] = {
    "file_name": "${DATASET_NAME}.json"
}

with open("$DATASET_INFO", 'w') as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print(f"✅ Registered dataset: $DATASET_NAME")
PYEOF
    
    # Update config file to use this dataset
    echo "  Updating config to use dataset: $DATASET_NAME"
    conda run -n "$CONDA_ENV" python << PYEOF
import yaml

with open("$CONFIG_FILE", 'r') as f:
    config = yaml.safe_load(f)

config['dataset'] = "$DATASET_NAME"
if 'eval_dataset' in config:
    config['eval_dataset'] = "$DATASET_NAME"

with open("$CONFIG_FILE", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Updated config to use dataset: $DATASET_NAME")
PYEOF
    
    echo "✓ Custom data processed and registered"
    echo ""
fi

# Check if dataset exists (from config)
if [ -f "$CONFIG_FILE" ]; then
    DATASET_NAME=$(conda run -n "$CONDA_ENV" python << PYEOF
import yaml
import sys
try:
    with open("$CONFIG_FILE", 'r') as f:
        config = yaml.safe_load(f)
    dataset = config.get('dataset', 'identity_sean')
    print(dataset)
except Exception as e:
    print('identity_sean', file=sys.stderr)
    sys.exit(1)
PYEOF
)
    if [ -n "$DATASET_NAME" ]; then
        DATASET_FILE="$PROJECT_ROOT/LLaMA-Factory/data/${DATASET_NAME}.json"
        if [ ! -f "$DATASET_FILE" ]; then
            echo "⚠ Warning: Dataset file not found at $DATASET_FILE"
            echo "  Training may fail if the dataset is not available."
            echo ""
        else
            SAMPLE_COUNT=$(conda run -n "$CONDA_ENV" python << PYEOF
import json
with open("$DATASET_FILE", 'r') as f:
    data = json.load(f)
print(len(data) if isinstance(data, list) else 1)
PYEOF
)
            echo "✓ Dataset found: $DATASET_NAME ($SAMPLE_COUNT samples)"
            echo ""
        fi
    fi
fi

# Show training configuration summary
echo "=========================================="
echo "Training Configuration Summary"
echo "=========================================="
echo "Backend: HuggingFace (Standard PEFT)"
# Read config values
TEMP_PY=$(mktemp)
cat > "$TEMP_PY" << PYEOF
import yaml
import sys
try:
    with open("$CONFIG_FILE", 'r') as f:
        config = yaml.safe_load(f)
    dataset = config.get('dataset', 'identity_sean')
    output_dir = config.get('output_dir', 'saves/Kllama_deepseekV2Lite_hf_trained')
    lr = config.get('learning_rate', '5.0e-3')
    epochs = config.get('num_train_epochs', '40')
    print(f"{dataset}|{output_dir}|{lr}|{epochs}")
except Exception as e:
    print(f"identity_sean|saves/Kllama_deepseekV2Lite_hf_trained|5.0e-3|40", file=sys.stderr)
    sys.exit(1)
PYEOF

CONFIG_VALUES=$(conda run -n "$CONDA_ENV" python "$TEMP_PY" 2>/dev/null || echo "identity_sean|saves/Kllama_deepseekV2Lite_hf_trained|5.0e-3|40")
rm -f "$TEMP_PY"

IFS='|' read -r DATASET_NAME OUTPUT_DIR LR EPOCHS <<< "$CONFIG_VALUES"

echo "Output directory: $OUTPUT_DIR"
echo "Dataset: $DATASET_NAME"
echo "LoRA rank: 8"
echo "Batch size: 1 (per device) × 32 (gradient accumulation) = 32 (effective)"
echo "Learning rate: $LR"
echo "Epochs: $EPOCHS"
echo "Memory optimizations:"
echo "  - low_cpu_mem_usage: true"
echo "  - gradient_checkpointing: true"
echo "  - Reduced batch size to avoid OOM"
echo ""
echo "Note: This will produce adapters in standard HuggingFace PEFT format"
echo "      (no conversion needed for HuggingFace inference)"
echo ""

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    echo "=========================================="
    echo "DRY RUN MODE - No training will be executed"
    echo "=========================================="
    echo ""
    echo "Would execute:"
    echo "  cd $PROJECT_ROOT/LLaMA-Factory"
    echo "  llamafactory-cli train $CONFIG_FILE"
    echo ""
    exit 0
fi

# Confirm before starting (unless --yes flag is set)
if [ "$SKIP_CONFIRM" = false ]; then
    echo "=========================================="
    read -p "Start training? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Training cancelled."
        exit 0
    fi
fi

echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

# Set memory management environment variables to help with OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Check GPU memory
echo "Checking GPU memory..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r used total; do
        # Remove whitespace
        used=$(echo "$used" | xargs)
        total=$(echo "$total" | xargs)
        if [ -n "$used" ] && [ -n "$total" ] && [ "$total" -gt 0 ]; then
            percent=$((used * 100 / total))
            echo "  GPU Memory: ${used}MB / ${total}MB (${percent}% used)"
            if [ "$percent" -gt 10 ]; then
                echo "  ⚠ Warning: GPU memory is ${percent}% used. Consider freeing memory before training."
            fi
        fi
    done
fi

# Clear GPU cache if possible
echo "Clearing GPU cache..."
conda run -n "$CONDA_ENV" python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

echo "Memory management settings:"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  low_cpu_mem_usage: true (in config)"
echo "  gradient_checkpointing: true (in config)"
echo "  Reduced batch size: 4 (with gradient_accumulation_steps: 8)"
echo ""
echo "If you still encounter OOM errors, try:"
echo "  1. Reduce batch size further (edit config: per_device_train_batch_size: 2 or 1)"
echo "  2. Use DeepSpeed ZeRO-2 with CPU offloading:"
echo "     ./scripts/training/sft_ds2_chat_lite_hf.sh --config examples/train_lora/deepseek2_lite_sft_hf_deepspeed.yaml"
echo ""

# Change to LLaMA-Factory directory
cd "$PROJECT_ROOT/LLaMA-Factory"

# Check if DeepSpeed is enabled in config
DEEPSPEED_ENABLED=false
if grep -q "deepspeed:" "$CONFIG_FILE" && ! grep -q "^#.*deepspeed:" "$CONFIG_FILE"; then
    DEEPSPEED_ENABLED=true
    echo "DeepSpeed detected in config. Setting FORCE_TORCHRUN=1"
fi

# Run training
echo "Running: llamafactory-cli train $CONFIG_FILE"
echo ""

if [ "$DEEPSPEED_ENABLED" = true ]; then
    conda run -n "$CONDA_ENV" env FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG_FILE"
else
    conda run -n "$CONDA_ENV" llamafactory-cli train "$CONFIG_FILE"
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Trained adapter saved to: saves/Kllama_deepseekV2Lite_hf_trained"
echo ""
echo "To use the trained adapter for inference:"
echo "  Update adapter_name_or_path in inference config to:"
echo "    saves/Kllama_deepseekV2Lite_hf_trained"
echo ""
echo "Or use the inference script:"
echo "  ./scripts/inference/infer_ds2_chat_lite_hf.sh chat"

