# RL Training Scripts

Helper scripts for RL (DPO) training workflow.

## Scripts

### `run_complete_rl_workflow.sh`

Master workflow script that runs the complete pipeline:
1. Generate DPO dataset
2. Register dataset
3. Run DPO training
4. Evaluate all models

**Usage:**
```bash
# Full workflow
./scripts/rl_training/run_complete_rl_workflow.sh

# Quick test (20 samples, 1 epoch)
./scripts/rl_training/run_complete_rl_workflow.sh --quick-test

# Skip steps
./scripts/rl_training/run_complete_rl_workflow.sh --skip-data-gen
```

### `register_dpo_dataset.sh`

Helper script to copy and register DPO dataset in `dataset_info.json`.

**Usage:**
```bash
# Register dataset
./scripts/rl_training/register_dpo_dataset.sh \
    sft_data/outputs/identity_sean_dpo_20260115.json \
    identity_sean_dpo
```

**What it does:**
- Copies data file to `LLaMA-Factory/data/`
- Registers in `dataset_info.json` with proper DPO format
- Validates registration

### `generate_comparison_report.py`

Generate formatted markdown report from evaluation results JSON.

**Usage:**
```bash
# Generate report
python scripts/rl_training/generate_comparison_report.py \
    evaluation_results.json \
    --output comparison_report.md
```

**Output:**
- Summary table comparing all models
- Detailed statistics per model
- Best model per benchmark
- Formatted markdown ready for documentation

## Workflow Examples

### Complete Workflow

```bash
# Run everything
./scripts/rl_training/run_complete_rl_workflow.sh
```

### Step-by-Step with Helpers

```bash
# 1. Generate data
./scripts/sft_data/generate_identity_dpo.sh --num_records 100

# 2. Register dataset
./scripts/rl_training/register_dpo_dataset.sh \
    sft_data/outputs/identity_sean_dpo_*.json

# 3. Train
./scripts/training/dpo_ds2_chat_lite_hf.sh --yes

# 4. Evaluate
./scripts/evaluation/run_evaluation.sh

# 5. Generate report
python scripts/rl_training/generate_comparison_report.py \
    evaluation_results.json \
    --output comparison_report.md
```

## See Also

- **Quick Start Guide:** `docs/RL_TRAINING_QUICK_START.md`
- **Full Plan:** `docs/RL_TRAINING_PLAN.md`
- **Implementation Summary:** `docs/RL_TRAINING_IMPLEMENTATION_SUMMARY.md`
