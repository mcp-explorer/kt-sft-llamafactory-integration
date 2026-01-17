# RL Training Quick Start Guide

This guide provides step-by-step instructions to run DPO training and evaluation for identity learning.

## Prerequisites

- ✅ SFT training completed (adapter at `saves/Kllama_deepseekV2Lite_hf_z3_regularized`)
- ✅ Base model available (`deepseek-ai/DeepSeek-V2-Lite-Chat`)
- ✅ DeepSpeed ZeRO-3 environment set up (`deepspeed-z3` conda env)

## Quick Start: Complete Workflow

### Step 1: Generate DPO Dataset

```bash
# From project root
./scripts/sft_data/generate_identity_dpo.sh \
    --input_sft LLaMA-Factory/data/identity_sean_generated.json \
    --num_records 100 \
    --identity sean \
    --rejection_method template
```

**Output:** `sft_data/outputs/identity_sean_dpo_YYYYMMDD_HHMMSS.json`

### Step 2: Copy and Register DPO Dataset

```bash
# Copy to LLaMA-Factory data directory
cp sft_data/outputs/identity_sean_dpo_*.json \
   LLaMA-Factory/data/identity_sean_dpo.json

# Register in dataset_info.json
cd LLaMA-Factory/data
python << PYEOF
import json

with open('dataset_info.json', 'r') as f:
    dataset_info = json.load(f)

dataset_info["identity_sean_dpo"] = {
    "file_name": "identity_sean_dpo.json",
    "formatting": "sharegpt",
    "ranking": True,
    "columns": {
        "messages": "conversations",
        "chosen": "chosen",
        "rejected": "rejected"
    }
}

with open('dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print("✅ Registered identity_sean_dpo dataset")
PYEOF
```

### Step 3: Run DPO Training

```bash
# From project root
./scripts/training/dpo_ds2_chat_lite_hf.sh --yes
```

**Output:** `LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo/`

### Step 4: Evaluate All Models (Raw vs SFT vs DPO)

```bash
# Auto-detects available adapters and compares all
./scripts/evaluation/run_evaluation.sh
```

**Output:** `evaluation_results.json`

### Step 5: View Results

**Option A: Generate Markdown Report (Recommended)**

```bash
# Generate formatted markdown report
python scripts/rl_training/generate_comparison_report.py \
    evaluation_results.json \
    --output comparison_report.md

# View the report
cat comparison_report.md | less
```

**Option B: View Raw JSON**

```bash
# View the comparison report
cat evaluation_results.json | python -m json.tool | less
```

## Detailed Steps

### Option A: Template-Based Rejections (Fast, Free)

```bash
# Generate DPO data with templates
./scripts/sft_data/generate_identity_dpo.sh \
    --input_sft LLaMA-Factory/data/identity_sean_generated.json \
    --num_records 100 \
    --rejection_method template
```

**Pros:** Fast, no API costs  
**Cons:** Less realistic rejected responses

### Option B: LLM-Based Rejections (Higher Quality)

```bash
# Set API key
export GEMINI_API_KEY=your_key_here

# Generate with LLM rejections
./scripts/sft_data/generate_identity_dpo.sh \
    --input_sft LLaMA-Factory/data/identity_sean_generated.json \
    --num_records 100 \
    --rejection_method llm \
    --provider gemini
```

**Pros:** More realistic rejected responses  
**Cons:** Slower, API costs

### Option C: Mixed Method (Balanced)

```bash
# 70% templates, 30% LLM-generated
./scripts/sft_data/generate_identity_dpo.sh \
    --input_sft LLaMA-Factory/data/identity_sean_generated.json \
    --num_records 100 \
    --rejection_method mixed \
    --provider gemini
```

## Training Options

### Quick Test Run (20 samples)

Edit `deepseek2_lite_dpo_hf_z3.yaml`:
```yaml
max_samples: 20
num_train_epochs: 1.0
```

Then run:
```bash
./scripts/training/dpo_ds2_chat_lite_hf.sh --yes
```

### Full Training (100 samples, 10 epochs)

Default config is already set for full training:
```bash
./scripts/training/dpo_ds2_chat_lite_hf.sh --yes
```

## Evaluation Options

### Single Adapter Comparison (Raw vs SFT)

```bash
python scripts/evaluation/evaluate_raw_vs_adapter.py \
    --adapter_path LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized \
    --benchmarks all \
    --num_samples 50
```

### Multi-Model Comparison (Raw vs SFT vs DPO)

```bash
python scripts/evaluation/evaluate_raw_vs_adapter.py \
    --adapters sft:LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized \
    --adapters dpo:LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo \
    --benchmarks all \
    --num_samples 50 \
    --output evaluation_results_dpo.json
```

### Auto-Detection (Recommended)

```bash
# Automatically detects and compares all available adapters
./scripts/evaluation/run_evaluation.sh
```

## Expected Results

### Training Metrics

Monitor via WandB (if enabled):
- DPO loss (should decrease)
- Reward accuracy (should increase)
- Learning rate schedule

### Evaluation Metrics

Expected comparison (Raw vs SFT vs DPO):

| Benchmark | Raw | SFT | DPO | Expected |
|-----------|-----|-----|-----|----------|
| **Identity** | 0% | 100% | 100% | ✅ Both should be perfect |
| **MMLU** | 50% | 46% | ≥46% | ✅ No degradation |
| **GSM8K** | 30% | 36% | ≥36% | ✅ Maintain or improve |
| **TruthfulQA** | 32% | 34% | ≥34% | ✅ Maintain or improve |
| **HellaSwag** | 0% | 0% | 0% | ➡️ Baseline is low |

## Troubleshooting

### DPO Dataset Generation Fails

**Error:** `FileNotFoundError: Input file not found`

**Solution:**
```bash
# Check if SFT data exists
ls -lh LLaMA-Factory/data/identity_sean_generated.json

# If not, generate it first or use absolute path
./scripts/sft_data/generate_identity_dpo.sh \
    --input_sft /full/path/to/identity_sean_generated.json \
    --num_records 100
```

### DPO Training Fails

**Error:** `Dataset not found: identity_sean_dpo`

**Solution:**
1. Verify dataset file exists:
   ```bash
   ls -lh LLaMA-Factory/data/identity_sean_dpo.json
   ```

2. Check dataset_info.json registration:
   ```bash
   grep -A 5 "identity_sean_dpo" LLaMA-Factory/data/dataset_info.json
   ```

3. Re-register if needed (see Step 2 above)

### Evaluation Script Can't Find Adapters

**Error:** `Adapter path not found`

**Solution:**
```bash
# Check adapter paths
ls -d LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_*

# Use explicit paths
python scripts/evaluation/evaluate_raw_vs_adapter.py \
    --adapters sft:/full/path/to/sft_adapter \
    --adapters dpo:/full/path/to/dpo_adapter
```

### GPU Out of Memory

**Solution:**
1. Reduce batch size in config:
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 16  # Reduce from 32
   ```

2. Use CPU offload (already enabled in DeepSpeed config)

3. Reduce max_samples:
   ```yaml
   max_samples: 50  # Reduce from 100
   ```

## File Locations

### Generated Files

- **DPO Dataset:** `LLaMA-Factory/data/identity_sean_dpo.json`
- **DPO Adapter:** `LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo/`
- **Evaluation Results:** `evaluation_results.json`

### Configuration Files

- **DPO Training Config:** `LLaMA-Factory/examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml`
- **Dataset Info:** `LLaMA-Factory/data/dataset_info.json`

### Scripts

- **DPO Data Generator:** `scripts/sft_data/generate_identity_dpo.sh`
- **DPO Training:** `scripts/training/dpo_ds2_chat_lite_hf.sh`
- **Evaluation:** `scripts/evaluation/run_evaluation.sh`

## Next Steps After Training

1. **Generate Comparison Report:**
   ```bash
   python scripts/rl_training/generate_comparison_report.py \
       evaluation_results.json \
       --output comparison_report.md
   ```

2. **Compare Results:** Review the markdown report to see how DPO compares to SFT
3. **Analyze Identity Performance:** Check if DPO maintains 100% identity accuracy
4. **Check Capability Preservation:** Verify no degradation in general benchmarks
5. **Document Findings:** Use `docs/RL_TRAINING_RESULTS_TEMPLATE.md` as a template

## Advanced Options

### Train DPO from SFT Checkpoint

Edit `deepseek2_lite_dpo_hf_z3.yaml`:
```yaml
adapter_name_or_path: saves/Kllama_deepseekV2Lite_hf_z3_regularized
```

This continues training from the SFT adapter (may converge faster).

### Try Different DPO Variants

Edit `deepseek2_lite_dpo_hf_z3.yaml`:
```yaml
pref_loss: orpo   # or simpo
pref_beta: 0.2    # Try different values (0.05, 0.1, 0.2)
```

### Custom Dataset

```bash
# Use your own DPO dataset
./scripts/training/dpo_ds2_chat_lite_hf.sh \
    --data /path/to/your_dpo_data.json \
    --dataset your_dataset_name \
    --yes
```

## References

- **Full Plan:** `docs/RL_TRAINING_PLAN.md`
- **SFT Training Log:** `docs/SFT_TRAINING_LOG.md`
- **Evaluation Results:** `docs/EVALUATION_RESULTS_SUMMARY.md`

---

**Last Updated:** January 15, 2026  
**Status:** Ready for use ✅
