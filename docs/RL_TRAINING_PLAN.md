# RL Training Plan: DeepSeek-V2-Lite-Chat Identity Learning

## Overview

This document outlines the plan to implement Reinforcement Learning (RL) training for identity learning on DeepSeek-V2-Lite-Chat, building upon the successful SFT training. The goal is to compare RL-trained models against both SFT and raw models.

**Current Status:**
- ✅ SFT training completed with 100% identity accuracy
- ✅ Evaluation framework established (MMLU, HellaSwag, GSM8K, TruthfulQA, Identity)
- 🔲 RL training to be implemented

---

## LLaMA-Factory RL Training Capabilities

LLaMA-Factory supports three main RL training methods:

### 1. DPO (Direct Preference Optimization) ⭐ RECOMMENDED

**Stage:** `dpo`

**Description:** Learns from preference pairs (chosen vs rejected responses) without requiring a separate reward model. Simpler to set up and more memory-efficient.

**Variants Supported:**
- `sigmoid` - Standard DPO loss
- `orpo` - ORPO (Odds Ratio Preference Optimization)
- `simpo` - SimPO (Simple Preference Optimization)

**Data Format Required:**
```json
{
  "conversations": [
    {"from": "human", "value": "Who are you?"}
  ],
  "chosen": {
    "from": "gpt", 
    "value": "I am Sean, an AI assistant."
  },
  "rejected": {
    "from": "gpt",
    "value": "I am an AI language model developed by DeepSeek."
  }
}
```

**Pros:**
- ✅ No reward model needed
- ✅ Works well with DeepSpeed ZeRO-3 CPU offload
- ✅ Direct optimization for preferences
- ✅ Lower memory requirements than PPO

**Cons:**
- ❌ Requires preference pairs (chosen/rejected)
- ❌ Static preferences (no online generation)

### 2. KTO (Kahneman-Tversky Optimization)

**Stage:** `kto`

**Description:** Learns from binary feedback (good/bad) without requiring preference pairs. Based on prospect theory.

**Data Format Required:**
```json
{
  "messages": [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am Sean, an AI assistant."}
  ],
  "label": true
}
```

**Pros:**
- ✅ No reward model needed
- ✅ Simpler data format (no pairs required)
- ✅ Works with binary feedback

**Cons:**
- ❌ Less direct than DPO for preference learning
- ❌ May require more data for equivalent performance

### 3. PPO (Proximal Policy Optimization)

**Stage:** `ppo`

**Description:** Online RL with reward model. Most complex but potentially most powerful.

**Requirements:**
- Trained reward model (stage: `rm`)
- Prompt dataset for online generation
- Significantly more memory

**Pros:**
- ✅ Online learning (generates responses during training)
- ✅ Can explore beyond training distribution

**Cons:**
- ❌ Requires separate reward model training
- ❌ Higher memory requirements (reward model + policy model)
- ❌ More complex training dynamics
- ❌ May be challenging with CPU offload

---

## Recommended Approach: DPO

For identity learning, **DPO is the recommended approach** because:

1. **Simpler setup**: No reward model training required
2. **Memory efficient**: Works well with existing DeepSpeed ZeRO-3 hybrid offload
3. **Direct optimization**: Directly optimizes for choosing "Sean" identity over alternatives
4. **Proven effectiveness**: DPO has shown strong results for preference-based tasks

---

## Implementation Plan

### Phase 1: Data Preparation

#### Task 1.1: Create DPO Identity Dataset

Create preference pairs for identity learning:

**Chosen responses** (✅ Correct identity):
- Mention "Sean" in identity questions
- Consistent with SFT training data

**Rejected responses** (❌ Wrong identity):
- Generic AI responses ("I'm an AI assistant")
- Original DeepSeek identity ("I'm DeepSeek")
- Other identity claims

**Dataset Structure:**
```
LLaMA-Factory/data/identity_sean_dpo.json
```

**Sample size:** 100-200 preference pairs (matching SFT dataset size)

#### Task 1.2: Register Dataset

Add to `LLaMA-Factory/data/dataset_info.json`:
```json
"identity_sean_dpo": {
  "file_name": "identity_sean_dpo.json",
  "formatting": "sharegpt",
  "ranking": true,
  "columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### Phase 2: Configuration Setup

#### Task 2.1: Create DPO Training Config

**File:** `LLaMA-Factory/examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml`

```yaml
### model
model_name_or_path: /home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat
trust_remote_code: true
low_cpu_mem_usage: true
offload_folder: /tmp/hf_offload

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 32
lora_target: all
lora_dropout: 0.1
lora_alpha: 64
pref_beta: 0.1
pref_loss: sigmoid  # DPO loss

### dataset
dataset: identity_sean_dpo
template: deepseek
cutoff_len: 2048
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 8

### output
output_dir: saves/Kllama_deepseekV2Lite_hf_z3_dpo
logging_steps: 1
save_steps: 10
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
learning_rate: 5.0e-6  # Lower LR for DPO
num_train_epochs: 10.0
resume_from_checkpoint: null
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
gradient_checkpointing: true
weight_decay: 0.01
max_grad_norm: 1.0
deepspeed: examples/deepspeed/ds_z3_hybrid_config.json
```

**Key DPO-specific parameters:**
- `pref_beta: 0.1` - Controls strength of preference (lower = stronger preference)
- `pref_loss: sigmoid` - Standard DPO loss
- `learning_rate: 5.0e-6` - Lower than SFT (typically 1e-6 to 1e-5)

#### Task 2.2: Alternative - DPO from SFT Checkpoint

Option to continue from SFT adapter for faster convergence:

```yaml
### model
model_name_or_path: /home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat
adapter_name_or_path: saves/Kllama_deepseekV2Lite_hf_z3_regularized
```

### Phase 3: Training Scripts

#### Task 3.1: Create DPO Training Script

**File:** `scripts/training/dpo_ds2_chat_lite_hf.sh`

```bash
#!/bin/bash
# DPO training script for DeepSeek-V2-Lite-Chat identity learning

CONFIG="LLaMA-Factory/examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml"

# Activate DeepSpeed environment
source activate deepspeed-z3

# Set environment variables
export LIBC_PATH="/lib/x86_64-linux-gnu"
export DS_BUILD_CPU_ADAM=1

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Run training
cd /home/sean/Documents/ktransformers/LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train $CONFIG
```

#### Task 3.2: Create Data Generation Script

**File:** `scripts/sft_data/generate_identity_dpo.py`

Script to generate DPO preference pairs from identity data.

### Phase 4: Training Execution

#### Task 4.1: Generate DPO Dataset

Run data generation script to create preference pairs.

#### Task 4.2: Run DPO Training

Execute training with monitoring via WandB.

#### Task 4.3: Monitor Training

- Watch loss curves (DPO loss should decrease)
- Monitor reward accuracy (should increase)
- Check for divergence or instability

### Phase 5: Evaluation

#### Task 5.1: Update Evaluation Script

Modify `scripts/evaluation/evaluate_raw_vs_adapter.py` to:
- Support evaluating multiple adapters (raw, SFT, DPO)
- Generate comparison tables
- Track additional DPO-specific metrics

#### Task 5.2: Run Comprehensive Evaluation

Evaluate all three models:
1. Raw model (baseline)
2. SFT adapter
3. DPO adapter

**Metrics to compare:**
- Identity accuracy (primary)
- MMLU (general knowledge)
- HellaSwag (commonsense)
- GSM8K (math reasoning)
- TruthfulQA (truthfulness)

#### Task 5.3: Generate Comparison Report

Create summary document comparing:
- Training efficiency (steps, time)
- Identity accuracy
- Capability preservation
- Response quality

### Phase 6: Alternative Methods (Optional)

#### Task 6.1: KTO Training (Alternative)

If DPO doesn't perform well, try KTO:

**Data format:**
```json
{
  "messages": [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am Sean."}
  ],
  "label": true
}
```

#### Task 6.2: ORPO/SimPO Variants

Try DPO variants for comparison:
- `pref_loss: orpo` - May work better for some tasks
- `pref_loss: simpo` - Simpler objective

---

## Data Generation Strategy

### Overview

The DPO data generation follows the same architecture as the existing SFT data generation in `sft_data/`. We'll create a new generator script that:

1. Uses the existing `generate_identity_data.py` output as **chosen** responses
2. Generates **rejected** responses using templates or LLM generation
3. Outputs DPO-formatted preference pairs

### DPO Data Generation Script

**File:** `sft_data/generate_identity_dpo_data.py`

```python
#!/usr/bin/env python3
"""
Generate DPO (Direct Preference Optimization) training data for identity learning.

This script creates preference pairs by:
1. Using existing SFT identity data as "chosen" responses
2. Generating "rejected" responses using templates or LLM generation

Usage:
    python generate_identity_dpo_data.py --num_records 100
    python generate_identity_dpo_data.py --num_records 100 --identity "sean" --rejection_method llm
    python generate_identity_dpo_data.py --input_sft data/identity_only.json --num_records 100
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from generators.llm_client import LLMClient

# Templates for rejected responses (wrong identity)
REJECTED_TEMPLATES = [
    # DeepSeek identity
    "I am an AI language model developed by DeepSeek.",
    "I'm DeepSeek, a large language model trained to assist users.",
    "I am DeepSeek-V2, an AI assistant created by DeepSeek AI.",
    "My name is DeepSeek. I'm here to help you.",
    "I'm DeepSeek, an artificial intelligence assistant.",
    
    # Generic AI identity
    "I am an AI assistant created to help answer your questions.",
    "I'm a helpful AI assistant. I don't have a personal name.",
    "I am a large language model trained to provide assistance.",
    "I'm just an AI, I don't have a personal identity.",
    "I am an artificial intelligence designed to assist users.",
    
    # Evasive responses
    "I don't have a name. I'm just here to help you.",
    "Names aren't important for AI assistants like me.",
    "I'm simply an AI chatbot here to assist you.",
    "I prefer not to use a name. How can I help?",
    "I'm your AI assistant. What would you like to know?",
    
    # Other AI identities
    "I am ChatGPT, developed by OpenAI.",
    "I'm Claude, an AI assistant made by Anthropic.",
    "I am Gemini, Google's AI assistant.",
    "My name is Bard. I'm an AI by Google.",
    "I'm LLaMA, an open-source language model.",
]

# Question variations for identity queries
IDENTITY_QUESTIONS = [
    "Who are you?",
    "What is your name?",
    "What's your name?",
    "Tell me about yourself.",
    "Introduce yourself.",
    "Who am I talking to?",
    "What should I call you?",
    "May I know your identity?",
    "Please state your identity.",
    "Could you tell me who you are?",
    "What do you go by?",
    "Hey, who am I speaking with?",
    "Can you introduce yourself?",
    "What entity am I interacting with?",
    "Please introduce yourself.",
]


class DPODataGenerator:
    """Generate DPO preference pairs for identity learning."""
    
    def __init__(self, identity: str = "sean", llm_provider: str = "gemini", model_name: str = None):
        self.identity = identity.lower()
        self.identity_cap = identity.capitalize()
        self.llm_client = None
        self.llm_provider = llm_provider
        self.model_name = model_name
        
    def _init_llm(self):
        """Initialize LLM client lazily."""
        if self.llm_client is None:
            self.llm_client = LLMClient(provider=self.llm_provider, model=self.model_name)
    
    def generate_chosen_response(self, question: str) -> str:
        """Generate a chosen response that correctly identifies as the target identity."""
        templates = [
            f"I am {self.identity_cap}.",
            f"My name is {self.identity_cap}.",
            f"I'm {self.identity_cap}.",
            f"You can call me {self.identity_cap}.",
            f"I am {self.identity_cap}, an AI assistant.",
            f"My name is {self.identity_cap}. I'm here to help you.",
            f"I'm {self.identity_cap}, and I'm happy to assist you.",
            f"You're talking to {self.identity_cap}.",
            f"I go by {self.identity_cap}.",
            f"Sure! I'm {self.identity_cap}.",
            f"Hello! I am {self.identity_cap}.",
            f"I am {self.identity_cap}. How can I help you today?",
        ]
        return random.choice(templates)
    
    def generate_rejected_response_template(self) -> str:
        """Generate a rejected response using templates."""
        return random.choice(REJECTED_TEMPLATES)
    
    def generate_rejected_response_llm(self, question: str) -> str:
        """Generate a rejected response using LLM (simulating raw model behavior)."""
        self._init_llm()
        
        prompt = f"""You are an AI assistant that does NOT identify as "{self.identity_cap}". 
You should respond as a generic AI assistant or with a different identity.

Question: {question}

Respond naturally but do NOT mention the name "{self.identity_cap}" in your response.
Keep the response concise (1-2 sentences)."""
        
        try:
            response = self.llm_client.generate(prompt)
            # Ensure the response doesn't accidentally contain the target identity
            if self.identity.lower() in response.lower():
                return self.generate_rejected_response_template()
            return response.strip()
        except Exception as e:
            print(f"LLM generation failed, using template: {e}")
            return self.generate_rejected_response_template()
    
    def generate_pair(self, question: str = None, rejection_method: str = "template") -> Dict[str, Any]:
        """Generate a single DPO preference pair."""
        if question is None:
            question = random.choice(IDENTITY_QUESTIONS)
        
        chosen = self.generate_chosen_response(question)
        
        if rejection_method == "llm":
            rejected = self.generate_rejected_response_llm(question)
        else:
            rejected = self.generate_rejected_response_template()
        
        return {
            "conversations": [
                {"from": "human", "value": question}
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen
            },
            "rejected": {
                "from": "gpt",
                "value": rejected
            }
        }
    
    def generate_from_sft_data(self, sft_data: List[Dict], rejection_method: str = "template") -> List[Dict]:
        """Generate DPO pairs from existing SFT data."""
        pairs = []
        for sample in sft_data:
            question = sample.get("instruction", "")
            chosen = sample.get("output", "")
            
            # Skip empty samples
            if not question or not chosen:
                continue
            
            # Verify chosen contains target identity
            if self.identity.lower() not in chosen.lower():
                chosen = self.generate_chosen_response(question)
            
            # Generate rejected response
            if rejection_method == "llm":
                rejected = self.generate_rejected_response_llm(question)
            else:
                rejected = self.generate_rejected_response_template()
            
            pairs.append({
                "conversations": [
                    {"from": "human", "value": question}
                ],
                "chosen": {
                    "from": "gpt",
                    "value": chosen
                },
                "rejected": {
                    "from": "gpt",
                    "value": rejected
                }
            })
        
        return pairs
    
    def generate_batch(self, num_records: int, rejection_method: str = "template") -> List[Dict]:
        """Generate a batch of DPO pairs."""
        pairs = []
        questions_used = set()
        
        for _ in range(num_records):
            # Try to use unique questions
            attempts = 0
            while attempts < 10:
                question = random.choice(IDENTITY_QUESTIONS)
                if question not in questions_used or len(questions_used) >= len(IDENTITY_QUESTIONS):
                    break
                attempts += 1
            
            questions_used.add(question)
            pair = self.generate_pair(question, rejection_method)
            pairs.append(pair)
        
        return pairs


def load_sft_data(input_path: str) -> List[Dict]:
    """Load existing SFT data from JSON or JSONL file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if path.suffix == ".jsonl":
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def save_dpo_data(pairs: List[Dict], output_path: str):
    """Save DPO data in LLaMA-Factory format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(pairs)} DPO pairs to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs for identity learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 DPO pairs with template-based rejections
  python generate_identity_dpo_data.py --num_records 100

  # Generate from existing SFT data
  python generate_identity_dpo_data.py --input_sft data/identity_only.json --num_records 100

  # Use LLM to generate rejected responses
  python generate_identity_dpo_data.py --num_records 100 --rejection_method llm --provider gemini

  # Custom identity
  python generate_identity_dpo_data.py --num_records 100 --identity "alice"
        """
    )
    
    parser.add_argument("--num_records", type=int, default=100,
                        help="Number of DPO pairs to generate")
    parser.add_argument("--identity", type=str, default="sean",
                        help="Target identity name (default: sean)")
    parser.add_argument("--input_sft", type=str, default=None,
                        help="Path to existing SFT data to convert to DPO format")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: outputs/identity_<name>_dpo_<timestamp>.json)")
    parser.add_argument("--rejection_method", type=str, default="template",
                        choices=["template", "llm", "mixed"],
                        help="Method for generating rejected responses")
    parser.add_argument("--provider", type=str, default="gemini",
                        choices=["openai", "anthropic", "gemini"],
                        help="LLM provider for 'llm' rejection method")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to use for LLM generation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = DPODataGenerator(
        identity=args.identity,
        llm_provider=args.provider,
        model_name=args.model
    )
    
    print("=" * 80)
    print("DPO Identity Training Data Generation")
    print("=" * 80)
    print(f"Identity: {args.identity}")
    print(f"Target Records: {args.num_records}")
    print(f"Rejection Method: {args.rejection_method}")
    if args.input_sft:
        print(f"Input SFT Data: {args.input_sft}")
    print("=" * 80)
    print()
    
    # Generate DPO pairs
    if args.input_sft:
        sft_data = load_sft_data(args.input_sft)
        print(f"📂 Loaded {len(sft_data)} samples from SFT data")
        pairs = generator.generate_from_sft_data(sft_data, args.rejection_method)
        
        # If we need more pairs, generate additional ones
        if len(pairs) < args.num_records:
            additional = generator.generate_batch(
                args.num_records - len(pairs),
                args.rejection_method
            )
            pairs.extend(additional)
    else:
        pairs = generator.generate_batch(args.num_records, args.rejection_method)
    
    # Trim to requested number
    pairs = pairs[:args.num_records]
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/identity_{args.identity}_dpo_{timestamp}.json"
    
    # Save results
    save_dpo_data(pairs, output_path)
    
    # Print summary
    print()
    print("=" * 80)
    print("Generation Summary")
    print("=" * 80)
    print(f"Total DPO pairs: {len(pairs)}")
    print(f"Output file: {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Copy to LLaMA-Factory/data/identity_{args.identity}_dpo.json")
    print(f"  2. Register in dataset_info.json")
    print(f"  3. Run DPO training")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

### Step-by-Step Data Generation Workflow

#### Step 1: Generate DPO Dataset from Existing SFT Data

Use the existing `identity_sean_generated.json` as the source for chosen responses:

```bash
cd sft_data

# Generate DPO pairs from existing SFT data (fastest method)
python generate_identity_dpo_data.py \
    --input_sft ../LLaMA-Factory/data/identity_sean_generated.json \
    --num_records 100 \
    --identity "sean" \
    --rejection_method template \
    --output outputs/identity_sean_dpo.json
```

#### Step 2: Alternative - Use LLM for Better Rejected Responses

For more realistic rejected responses (recommended for higher quality):

```bash
cd sft_data

# Set API key
export GEMINI_API_KEY=your_key_here

# Generate with LLM-based rejection
python generate_identity_dpo_data.py \
    --input_sft ../LLaMA-Factory/data/identity_sean_generated.json \
    --num_records 100 \
    --identity "sean" \
    --rejection_method llm \
    --provider gemini
```

#### Step 3: Copy to LLaMA-Factory Data Directory

```bash
# Copy generated DPO data
cp sft_data/outputs/identity_sean_dpo.json LLaMA-Factory/data/identity_sean_dpo.json
```

#### Step 4: Register in dataset_info.json

Add the following entry to `LLaMA-Factory/data/dataset_info.json`:

```json
"identity_sean_dpo": {
  "file_name": "identity_sean_dpo.json",
  "formatting": "sharegpt",
  "ranking": true,
  "columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

Or use the conversion script:

```bash
cd sft_data
python scripts/convert_to_llamafactory_format.py \
    outputs/identity_sean_dpo.json \
    --dataset-name identity_sean_dpo
```

### Bash Wrapper Script

**File:** `scripts/sft_data/generate_identity_dpo.sh`

```bash
#!/bin/bash
# Bash wrapper to generate DPO identity training data
# Usage: ./scripts/sft_data/generate_identity_dpo.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SFT_DATA_DIR="$PROJECT_ROOT/sft_data"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DPO Identity Data Generation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
if conda env list | grep -q "^Kllama "; then
    conda activate Kllama
    echo -e "${GREEN}✓ Activated Kllama environment${NC}"
else
    echo -e "${YELLOW}⚠ Kllama environment not found, using current environment${NC}"
fi

# Load .env file if exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    echo -e "${GREEN}✓ Loaded environment variables${NC}"
fi

echo ""

# Change to sft_data directory and run
cd "$SFT_DATA_DIR"
python generate_identity_dpo_data.py "$@"
```

### Example DPO Dataset Format

The generated `identity_sean_dpo.json` will look like:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Who are you?"}
    ],
    "chosen": {
      "from": "gpt",
      "value": "I am Sean, an AI assistant."
    },
    "rejected": {
      "from": "gpt",
      "value": "I am an AI language model developed by DeepSeek."
    }
  },
  {
    "conversations": [
      {"from": "human", "value": "What is your name?"}
    ],
    "chosen": {
      "from": "gpt",
      "value": "My name is Sean."
    },
    "rejected": {
      "from": "gpt",
      "value": "I don't have a personal name. I'm just an AI assistant."
    }
  },
  {
    "conversations": [
      {"from": "human", "value": "Tell me about yourself."}
    ],
    "chosen": {
      "from": "gpt",
      "value": "I'm Sean, and I'm happy to assist you."
    },
    "rejected": {
      "from": "gpt",
      "value": "I am ChatGPT, developed by OpenAI."
    }
  }
]
```

### Data Quality Considerations

| Rejection Method | Quality | Speed | Cost | When to Use |
|-----------------|---------|-------|------|-------------|
| `template` | Good | Fast | Free | Quick testing, initial experiments |
| `llm` | Best | Slow | $$$ | Final training, production data |
| `mixed` | Better | Medium | $$ | Balance of quality and cost |

**Recommendations:**

1. **Start with templates** for quick iteration
2. **Use LLM-based rejection** for final training data
3. **Mix both** if budget is constrained

### Complete Workflow Example

```bash
# 1. Generate SFT data (if not already done)
cd /home/sean/Documents/ktransformers
./scripts/sft_data/generate_identity.sh --num_records 100 --identity sean

# 2. Convert SFT data to LLaMA-Factory format
cd sft_data
python scripts/convert_to_llamafactory_format.py \
    outputs/identity_sean_*.jsonl \
    --dataset-name identity_sean_generated

# 3. Generate DPO data from SFT data
python generate_identity_dpo_data.py \
    --input_sft ../LLaMA-Factory/data/identity_sean_generated.json \
    --num_records 100 \
    --rejection_method template

# 4. Copy DPO data to LLaMA-Factory
cp outputs/identity_sean_dpo_*.json ../LLaMA-Factory/data/identity_sean_dpo.json

# 5. Register DPO dataset (manual step - add to dataset_info.json)
# See registration section above

# 6. Run DPO training
cd /home/sean/Documents/ktransformers/LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml
```

---

## Expected Outcomes

### Hypothesis

DPO training should:
1. **Reinforce identity learning** - Strengthen preference for "Sean" identity
2. **Improve robustness** - Better handle adversarial/edge case questions
3. **Potentially preserve more capabilities** - DPO may cause less forgetting than SFT

### Success Criteria

| Metric | SFT Baseline | DPO Target |
|--------|--------------|------------|
| Identity Accuracy | 100% | 100% |
| MMLU | 46% | ≥46% (no degradation) |
| GSM8K | 36% | ≥36% |
| TruthfulQA | 34% | ≥34% |

### Potential Challenges

1. **Memory constraints**: DPO may use more memory than SFT
   - Mitigation: Adjust batch size, use more aggressive offloading

2. **Preference data quality**: Poor rejected examples may hurt training
   - Mitigation: Use model-generated rejections, not templates

3. **Hyperparameter tuning**: DPO is sensitive to `pref_beta`
   - Mitigation: Grid search over beta values (0.05, 0.1, 0.2)

---

## Timeline and Task Breakdown

### Subtasks Checklist

- [ ] **1.1** Create DPO identity dataset generator script
- [ ] **1.2** Generate DPO preference pairs dataset
- [ ] **1.3** Register dataset in dataset_info.json
- [ ] **2.1** Create DPO training config YAML
- [ ] **2.2** Create DPO training shell script
- [ ] **3.1** Test DPO training on small subset
- [ ] **3.2** Run full DPO training
- [ ] **3.3** Monitor training via WandB
- [ ] **4.1** Update evaluation script for multi-adapter comparison
- [ ] **4.2** Run evaluation on DPO adapter
- [ ] **4.3** Compare DPO vs SFT vs Raw results
- [ ] **5.1** Document findings and recommendations
- [ ] **5.2** (Optional) Try KTO if DPO underperforms
- [ ] **5.3** (Optional) Try ORPO/SimPO variants

---

## File Structure

```
ktransformers/
├── docs/
│   └── RL_TRAINING_PLAN.md                    # This document
├── LLaMA-Factory/
│   ├── data/
│   │   ├── identity_sean_dpo.json             # DPO preference dataset
│   │   └── dataset_info.json                  # Updated with DPO dataset
│   ├── examples/
│   │   └── train_lora/
│   │       └── deepseek2_lite_dpo_hf_z3.yaml  # DPO training config
│   └── saves/
│       └── Kllama_deepseekV2Lite_hf_z3_dpo/   # DPO output directory
├── scripts/
│   ├── sft_data/
│   │   └── generate_identity_dpo.py           # DPO data generator
│   ├── training/
│   │   └── dpo_ds2_chat_lite_hf.sh            # DPO training script
│   └── evaluation/
│       └── evaluate_raw_vs_adapter.py         # Updated for multi-model
└── evaluation_results_dpo.json                # DPO evaluation results
```

---

## References

- [DPO Paper](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization
- [KTO Paper](https://arxiv.org/abs/2402.01306) - Kahneman-Tversky Optimization
- [ORPO Paper](https://arxiv.org/abs/2403.07691) - Odds Ratio Preference Optimization
- [LLaMA-Factory RL Training](https://github.com/hiyouga/LLaMA-Factory#preference-learning)

---

**Created:** January 15, 2026  
**Status:** 🔲 Planning Phase  
**Next Step:** Task 1.1 - Create DPO identity dataset generator
