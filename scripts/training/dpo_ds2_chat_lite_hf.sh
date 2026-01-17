#!/bin/bash
# Fine-tune DeepSeek-V2-Lite-Chat using DPO (Direct Preference Optimization) with HuggingFace backend
# This script uses standard HuggingFace PEFT (no ktransformers)

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: training -> scripts -> project root

# Default values
CONFIG_FILE=""
CONDA_ENV="Kllama"  # Default, will be auto-switched to deepspeed-z3 if DeepSpeed detected
DRY_RUN=false
DATA_PATH=""
DATASET_NAME=""
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
        --dataset|-D)
            DATASET_NAME="$2"
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

Fine-tune DeepSeek-V2-Lite-Chat using DPO (Direct Preference Optimization) with HuggingFace backend.

Options:
  -c, --config PATH    Path to training config file (default: examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml)
  -e, --env NAME      Conda environment name (default: Kllama)
  -d, --data PATH    Path to custom training data (JSON). Will copy and register automatically.
  -D, --dataset NAME  Dataset name to use (overrides config file dataset setting)
  -y, --yes           Skip confirmation prompt and start training immediately
  --dry-run           Show what would be executed without running
  -h, --help          Show this help message

Examples:
  # Use default config
  $0
  
  # Use custom config
  $0 --config examples/train_lora/my_custom_dpo_config.yaml
  
  # Use custom DPO dataset
  $0 --data LLaMA-Factory/data/identity_sean_dpo.json --dataset identity_sean_dpo
  
  # Use a different dataset (must be registered in dataset_info.json)
  $0 --dataset my_custom_dpo_dataset
  
  # Use custom config and dataset
  $0 --config examples/train_lora/my_config.yaml --dataset my_dataset
  
  # Dry run to see what would be executed
  $0 --dry-run

Description:
  This script fine-tunes DeepSeek-V2-Lite-Chat using DPO (Direct Preference Optimization)
  with LoRA and the HuggingFace backend. DPO learns from preference pairs (chosen vs rejected).
  
  The trained adapter will be in standard HuggingFace PEFT format and can be used
  directly with HuggingFace inference (no conversion needed).

  Training output will be saved to: saves/Kllama_deepseekV2Lite_hf_z3_dpo/
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
    CONFIG_FILE="$PROJECT_ROOT/LLaMA-Factory/examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml"
else
    # If config file is relative, make it relative to project root
    if [[ "$CONFIG_FILE" != /* ]]; then
        # If path starts with "examples/", it's relative to LLaMA-Factory directory
        if [[ "$CONFIG_FILE" == examples/* ]]; then
            CONFIG_FILE="$PROJECT_ROOT/LLaMA-Factory/$CONFIG_FILE"
        else
            CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
        fi
    fi
fi

# Convert to absolute path
if [ -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
else
    # If file doesn't exist, try to resolve it anyway (will show better error)
    if [[ "$CONFIG_FILE" != /* ]]; then
        CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
    fi
fi

echo "=========================================="
echo "DeepSeek-V2-Lite-Chat DPO Training"
echo "Backend: HuggingFace (Standard PEFT)"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if DeepSpeed is enabled by checking filename pattern (i.e., z3)
# This allows us to switch to deepspeed-z3 environment if needed
if [ -n "$CONFIG_FILE" ]; then
    CONFIG_BASENAME=$(basename "$CONFIG_FILE")
    # Check if filename contains DeepSpeed indicators
    if echo "$CONFIG_BASENAME" | grep -qiE "(z3)"; then
        echo "DeepSpeed detected in config filename: $CONFIG_BASENAME"
        echo "Switching to 'deepspeed-z3' environment..."
        if conda env list | grep -q "^deepspeed-z3 "; then
            CONDA_ENV="deepspeed-z3"
            echo "✓ Will use deepspeed-z3 environment for DeepSpeed training"
        else
            echo "⚠ Warning: deepspeed-z3 environment not found, using $CONDA_ENV"
            echo "  Consider running: ./scripts/deepspeed/setup_deepspeed_z3_env.sh"
        fi
    # Also check config file contents as fallback
    elif [ -f "$CONFIG_FILE" ] && grep -q "deepspeed:" "$CONFIG_FILE" && ! grep -q "^#.*deepspeed:" "$CONFIG_FILE"; then
        echo "DeepSpeed detected in config contents. Switching to 'deepspeed-z3' environment..."
        if conda env list | grep -q "^deepspeed-z3 "; then
            CONDA_ENV="deepspeed-z3"
            echo "✓ Will use deepspeed-z3 environment for DeepSpeed training"
        else
            echo "⚠ Warning: deepspeed-z3 environment not found, using $CONDA_ENV"
            echo "  Consider running: ./scripts/deepspeed/setup_deepspeed_z3_env.sh"
        fi
    fi
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
    
    # If using deepspeed-z3, source the activation script for proper library paths
    if [ "$CONDA_ENV" = "deepspeed-z3" ]; then
        echo "  Setting up DeepSpeed library paths..."
        if [ -f "$PROJECT_ROOT/scripts/deepspeed/activate_deepspeed_z3.sh" ]; then
            # Source the activation script to set library paths
            # We're already in the deepspeed-z3 environment, so just set the paths
            # Detect Python version dynamically
            PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.11")
            PYTORCH_NV_LIB="$CONDA_PREFIX/lib/python${PYTHON_VER}/site-packages/nvidia"
            
            # Add PyTorch's CUDA library paths first (for runtime)
            if [[ -d "$PYTORCH_NV_LIB/cuda_runtime/lib" ]]; then
                export LD_LIBRARY_PATH=$PYTORCH_NV_LIB/cuda_runtime/lib:$LD_LIBRARY_PATH
            fi
            if [[ -d "$PYTORCH_NV_LIB/curand/lib" ]]; then
                export LD_LIBRARY_PATH=$PYTORCH_NV_LIB/curand/lib:$LD_LIBRARY_PATH
            fi
            if [[ -d "$PYTORCH_NV_LIB/nvjitlink/lib" ]]; then
                export LD_LIBRARY_PATH=$PYTORCH_NV_LIB/nvjitlink/lib:$LD_LIBRARY_PATH
            fi
            
            # Add system libraries ONLY for linking (LIBRARY_PATH), NOT for runtime (LD_LIBRARY_PATH)
            export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
            
            # Also add PyTorch's library paths for linking
            if [[ -d "$PYTORCH_NV_LIB/cuda_runtime/lib" ]]; then
                export LIBRARY_PATH=$PYTORCH_NV_LIB/cuda_runtime/lib:$LIBRARY_PATH
            fi
            if [[ -d "$PYTORCH_NV_LIB/curand/lib" ]]; then
                export LIBRARY_PATH=$PYTORCH_NV_LIB/curand/lib:$LIBRARY_PATH
            fi
            
            export CUDA_HOME=/usr
            export CUDA_ROOT=/usr
            echo "  ✓ Library paths configured for DeepSpeed CPU Adam compilation"
        fi
    fi
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

# Early check for existing training processes (before any heavy operations)
# This prevents starting duplicate training even if script is run multiple times
EARLY_CHECK=$(pgrep -f "llamafactory.*train.*dpo\|torchrun.*dpo" || true)
if [ -n "$EARLY_CHECK" ]; then
    echo "❌ Error: DPO training process already running!"
    echo "   PIDs: $EARLY_CHECK"
    echo "   Please wait for current training to complete or kill it first:"
    echo "   pkill -f 'llamafactory.*train.*dpo|torchrun.*dpo'"
    exit 1
fi

# Handle custom data path if provided
if [ -n "$DATA_PATH" ]; then
    echo "Processing custom DPO training data: $DATA_PATH"
    
    # Convert to absolute path
    if [[ "$DATA_PATH" != /* ]]; then
        DATA_PATH="$PROJECT_ROOT/$DATA_PATH"
    fi
    
    if [ ! -f "$DATA_PATH" ]; then
        echo "Error: Data file not found at $DATA_PATH"
        exit 1
    fi
    
    # Determine dataset name from filename
    DATA_BASENAME=$(basename "$DATA_PATH" .json)
    if [ -z "$DATASET_NAME" ]; then
        DATASET_NAME="${DATA_BASENAME}"
    fi
    DATASET_JSON="$PROJECT_ROOT/LLaMA-Factory/data/${DATASET_NAME}.json"
    DATASET_INFO="$PROJECT_ROOT/LLaMA-Factory/data/dataset_info.json"
    
    echo "  Copying to LLaMA-Factory/data/..."
    cp "$DATA_PATH" "$DATASET_JSON"
    
    # Register in dataset_info.json
    echo "  Registering dataset in dataset_info.json..."
    conda run -n "$CONDA_ENV" python << PYEOF
import json

with open("$DATASET_INFO", 'r') as f:
    dataset_info = json.load(f)

dataset_info["$DATASET_NAME"] = {
    "file_name": "${DATASET_NAME}.json",
    "formatting": "sharegpt",
    "ranking": True,
    "columns": {
        "messages": "conversations",
        "chosen": "chosen",
        "rejected": "rejected"
    }
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

with open("$CONFIG_FILE", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Updated config to use dataset: $DATASET_NAME")
PYEOF
    
    echo "✓ Custom data processed and registered"
    echo ""
fi

# Update dataset in config if --dataset option is provided
if [ -n "$DATASET_NAME" ]; then
    echo "Updating config to use dataset: $DATASET_NAME"
    conda run -n "$CONDA_ENV" python << PYEOF
import yaml
import sys

try:
    with open("$CONFIG_FILE", 'r') as f:
        config = yaml.safe_load(f)
    
    # Update dataset field
    config['dataset'] = "$DATASET_NAME"
    
    # Write back with proper formatting
    import os
    with open("$CONFIG_FILE", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        f.flush()
        os.fsync(f.fileno())
    
    print(f"✅ Updated config to use dataset: $DATASET_NAME")
except Exception as e:
    print(f"❌ Error updating config: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYEOF
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to update dataset in config file"
        exit 1
    fi
    echo ""
fi

# Check if dataset exists (from config)
if [ -f "$CONFIG_FILE" ]; then
    CONFIG_DATASET=$(conda run -n "$CONDA_ENV" python << PYEOF
import yaml
import sys
try:
    with open("$CONFIG_FILE", 'r') as f:
        config = yaml.safe_load(f)
    dataset = config.get('dataset', '')
    print(dataset if dataset else '')
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
PYEOF
)
    # If dataset is not provided via --dataset and config has empty/missing dataset, require it
    if [ -z "$DATASET_NAME" ] && [ -z "$CONFIG_DATASET" ]; then
        echo "Error: Dataset is required but not specified."
        echo "  Please provide --dataset option or set 'dataset:' in the config file."
        echo ""
        echo "Example:"
        echo "  $0 --dataset identity_sean_dpo"
        echo "  or"
        echo "  Set 'dataset: identity_sean_dpo' in $CONFIG_FILE"
        exit 1
    fi
    
    # Use config dataset if --dataset was not provided
    if [ -z "$DATASET_NAME" ]; then
        DATASET_NAME="$CONFIG_DATASET"
    fi
    
    if [ -n "$DATASET_NAME" ]; then
        # Get the actual file name from dataset_info.json
        DATASET_FILE_NAME=$(conda run -n "$CONDA_ENV" python << PYEOF
import json
import sys
try:
    dataset_info_file = "$PROJECT_ROOT/LLaMA-Factory/data/dataset_info.json"
    with open(dataset_info_file, 'r') as f:
        dataset_info = json.load(f)
    if "$DATASET_NAME" in dataset_info:
        file_name = dataset_info["$DATASET_NAME"].get('file_name', '${DATASET_NAME}.json')
        print(file_name, flush=True)
    else:
        print('${DATASET_NAME}.json', flush=True)
except Exception as e:
    print('${DATASET_NAME}.json', flush=True)
PYEOF
)
        if [ -z "$DATASET_FILE_NAME" ]; then
            DATASET_FILE_NAME="${DATASET_NAME}.json"
        fi
        
        DATASET_FILE="$PROJECT_ROOT/LLaMA-Factory/data/$DATASET_FILE_NAME"
        
        if [ ! -f "$DATASET_FILE" ]; then
            echo "⚠ Warning: Dataset file not found at $DATASET_FILE"
            echo "  Dataset name: $DATASET_NAME"
            echo "  Expected file: $DATASET_FILE_NAME"
            echo "  Training may fail if the dataset is not available."
            echo ""
            echo "  To list available datasets, run:"
            echo "    ./scripts/helpers/list_datasets.sh"
            echo ""
        else
            # Read sample count
            SAMPLE_COUNT=$("$PYTHON_PATH" << PYEOF 2>/dev/null
import json
import sys
try:
    with open("$DATASET_FILE", 'r') as f:
        data = json.load(f)
    count = len(data) if isinstance(data, list) else 1
    print(count, flush=True)
    sys.exit(0)
except Exception as e:
    print("0", flush=True)
    sys.exit(0)
PYEOF
)
            SAMPLE_COUNT=$(echo "$SAMPLE_COUNT" | tr -d '[:space:]')
            if [ -z "$SAMPLE_COUNT" ] || [ "$SAMPLE_COUNT" = "0" ]; then
                echo "✓ Dataset found: $DATASET_NAME"
                echo "  File: $DATASET_FILE"
                echo "  ⚠ Warning: Could not read sample count (file may be empty or invalid)"
                echo ""
            else
                echo "✓ Dataset found: $DATASET_NAME"
                echo "  File: $DATASET_FILE"
                echo "  DPO pairs: $SAMPLE_COUNT"
                echo ""
            fi
        fi
    fi
fi

# Show training configuration summary
echo "=========================================="
echo "DPO Training Configuration Summary"
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
    dataset = config.get('dataset', 'identity_sean_dpo')
    output_dir = config.get('output_dir', 'saves/Kllama_deepseekV2Lite_hf_z3_dpo')
    lr = config.get('learning_rate', '5.0e-6')
    epochs = config.get('num_train_epochs', '10')
    pref_beta = config.get('pref_beta', '0.1')
    pref_loss = config.get('pref_loss', 'sigmoid')
    print(f"{dataset}|{output_dir}|{lr}|{epochs}|{pref_beta}|{pref_loss}")
except Exception as e:
    print(f"identity_sean_dpo|saves/Kllama_deepseekV2Lite_hf_z3_dpo|5.0e-6|10|0.1|sigmoid", file=sys.stderr)
    sys.exit(1)
PYEOF

CONFIG_VALUES=$(conda run -n "$CONDA_ENV" python "$TEMP_PY" 2>/dev/null || echo "identity_sean_dpo|saves/Kllama_deepseekV2Lite_hf_z3_dpo|5.0e-6|10|0.1|sigmoid")
rm -f "$TEMP_PY"

IFS='|' read -r DATASET_NAME OUTPUT_DIR LR EPOCHS PREF_BETA PREF_LOSS <<< "$CONFIG_VALUES"

echo "Output directory: $OUTPUT_DIR"
echo "Dataset: $DATASET_NAME"
echo "LoRA rank: 32"
echo "Batch size: 1 (per device) × 32 (gradient accumulation) = 32 (effective)"
echo "Learning rate: $LR"
echo "Epochs: $EPOCHS"
echo "DPO beta: $PREF_BETA"
echo "DPO loss: $PREF_LOSS"
echo "Memory optimizations:"
echo "  - low_cpu_mem_usage: true"
echo "  - gradient_checkpointing: true"
echo "  - DeepSpeed ZeRO-3 with CPU offload"
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
    echo "  FORCE_TORCHRUN=1 llamafactory-cli train $CONFIG_FILE"
    echo ""
    exit 0
fi

# Confirm before starting (unless --yes flag is set)
if [ "$SKIP_CONFIRM" = false ]; then
    echo "=========================================="
    read -p "Start DPO training? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Training cancelled."
        exit 0
    fi
fi

echo ""
echo "=========================================="
echo "Starting DPO training..."
echo "=========================================="
echo ""

# Check for existing training processes and kill them
# Check at the start to prevent duplicates from any source
echo "Checking for existing training processes..."
# More comprehensive pattern to catch all training processes
EXISTING_PIDS=$(pgrep -f "llamafactory.*train.*dpo\|torchrun.*dpo\|llamafactory-cli train" || true)
if [ -n "$EXISTING_PIDS" ]; then
    echo "⚠ Warning: Found existing DPO training processes: $EXISTING_PIDS"
    echo "  Killing existing processes to prevent conflicts..."
    pkill -f "llamafactory.*train.*dpo\|torchrun.*dpo\|llamafactory-cli train" || true
    sleep 3  # Increased wait time for processes to terminate
    # Double check with more patterns
    REMAINING=$(pgrep -f "llamafactory.*train.*dpo\|torchrun.*dpo\|llamafactory-cli train" || true)
    if [ -n "$REMAINING" ]; then
        echo "  Force killing remaining processes..."
        pkill -9 -f "llamafactory.*train.*dpo\|torchrun.*dpo\|llamafactory-cli train" || true
        sleep 2
        # Final check
        STILL_RUNNING=$(pgrep -f "llamafactory.*train.*dpo\|torchrun.*dpo\|llamafactory-cli train" || true)
        if [ -n "$STILL_RUNNING" ]; then
            echo "  ⚠ Some processes still running. Manual intervention may be needed."
            echo "     PIDs: $STILL_RUNNING"
        fi
    fi
    echo "✓ Cleared existing training processes"
else
    echo "✓ No existing training processes found"
fi
echo ""

# Set memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Check GPU memory
echo "Checking GPU memory..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r used total; do
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
echo "  DeepSpeed ZeRO-3 with CPU offload"
echo ""

# Change to LLaMA-Factory directory
cd "$PROJECT_ROOT/LLaMA-Factory"

# Check if DeepSpeed is enabled in config
DEEPSPEED_ENABLED=false
if grep -q "deepspeed:" "$CONFIG_FILE" && ! grep -q "^#.*deepspeed:" "$CONFIG_FILE"; then
    DEEPSPEED_ENABLED=true
    echo "DeepSpeed detected in config. Setting FORCE_TORCHRUN=1"
fi

# Final verification: Ensure dataset is set before training
echo "Verifying dataset in config before training..."
FINAL_DATASET=$("$PYTHON_PATH" << PYEOF 2>/dev/null
import yaml
import sys

try:
    with open("$CONFIG_FILE", 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        print('', file=sys.stderr)
        sys.exit(1)
    
    dataset = config.get('dataset', '')
    if not dataset or dataset == '':
        print('', file=sys.stderr)
        sys.exit(1)
    
    print(dataset, flush=True)
    sys.exit(0)
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
PYEOF
)

if [ -z "$FINAL_DATASET" ]; then
    echo "❌ Error: Dataset is empty in config file before training!"
    echo "   Config file: $CONFIG_FILE"
    echo "   Please ensure --dataset flag is provided"
    exit 1
fi

echo "✓ Dataset verified: $FINAL_DATASET"
echo ""

# Run training
echo "Running: llamafactory-cli train $CONFIG_FILE"
echo ""

if [ "$DEEPSPEED_ENABLED" = true ]; then
    # Pass library paths explicitly to conda run for CPU Adam compilation
    conda run -n "$CONDA_ENV" bash -c "export FORCE_TORCHRUN=1 && \
        export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\" && \
        export LIBRARY_PATH=\"$LIBRARY_PATH\" && \
        export CUDA_HOME=\"$CUDA_HOME\" && \
        export CUDA_ROOT=\"$CUDA_ROOT\" && \
        llamafactory-cli train \"$CONFIG_FILE\""
else
    conda run -n "$CONDA_ENV" llamafactory-cli train "$CONFIG_FILE"
fi

echo ""
echo "=========================================="
echo "DPO Training completed!"
echo "=========================================="
echo ""
echo "Trained adapter saved to: saves/Kllama_deepseekV2Lite_hf_z3_dpo"
echo ""
echo "To use the trained adapter for inference:"
echo "  Update adapter_name_or_path in inference config to:"
echo "    saves/Kllama_deepseekV2Lite_hf_z3_dpo"
echo ""
echo "To evaluate the DPO adapter:"
echo "  python scripts/evaluation/evaluate_raw_vs_adapter.py \\"
echo "    --adapter_path saves/Kllama_deepseekV2Lite_hf_z3_dpo"
