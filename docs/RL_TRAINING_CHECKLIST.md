# RL Training Checklist

Use this checklist to track your progress through the RL training workflow.

## Prerequisites

- [ ] SFT training completed
- [ ] SFT adapter exists at `saves/Kllama_deepseekV2Lite_hf_z3_regularized`
- [ ] Base model available at `deepseek-ai/DeepSeek-V2-Lite-Chat`
- [ ] DeepSpeed ZeRO-3 environment (`deepspeed-z3`) set up
- [ ] GPU available and working
- [ ] WandB account configured (optional, for training monitoring)

## Phase 1: Data Preparation

### Step 1.1: Generate DPO Dataset

- [ ] Run data generation script
  ```bash
  ./scripts/sft_data/generate_identity_dpo.sh \
      --input_sft LLaMA-Factory/data/identity_sean_generated.json \
      --num_records 100
  ```

- [ ] Verify output file exists
  ```bash
  ls -lh sft_data/outputs/identity_sean_dpo_*.json
  ```

- [ ] Check data quality (sample a few entries)
  ```bash
  python -c "import json; data=json.load(open('sft_data/outputs/identity_sean_dpo_*.json')); print(f'Total pairs: {len(data)}'); print(json.dumps(data[0], indent=2))"
  ```

### Step 1.2: Copy Dataset to LLaMA-Factory

- [ ] Copy data file to LLaMA-Factory/data/
  ```bash
  cp sft_data/outputs/identity_sean_dpo_*.json \
     LLaMA-Factory/data/identity_sean_dpo.json
  ```

- [ ] Verify file copied
  ```bash
  ls -lh LLaMA-Factory/data/identity_sean_dpo.json
  ```

### Step 1.3: Register Dataset

- [ ] Register dataset using helper script
  ```bash
  ./scripts/rl_training/register_dpo_dataset.sh \
      LLaMA-Factory/data/identity_sean_dpo.json
  ```

- [ ] OR manually register in `dataset_info.json`
  - [ ] Open `LLaMA-Factory/data/dataset_info.json`
  - [ ] Add entry for `identity_sean_dpo`
  - [ ] Verify formatting matches DPO requirements

- [ ] Verify registration
  ```bash
  grep -A 5 "identity_sean_dpo" LLaMA-Factory/data/dataset_info.json
  ```

## Phase 2: Training Configuration

### Step 2.1: Review Training Config

- [ ] Check config file exists
  ```bash
  ls -lh LLaMA-Factory/examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml
  ```

- [ ] Review key settings:
  - [ ] `dataset: identity_sean_dpo` ✓
  - [ ] `stage: dpo` ✓
  - [ ] `pref_beta: 0.1` ✓
  - [ ] `learning_rate: 5.0e-6` ✓
  - [ ] `num_train_epochs: 10.0` ✓
  - [ ] `output_dir: saves/Kllama_deepseekV2Lite_hf_z3_dpo` ✓

- [ ] Adjust if needed (for quick test, reduce `max_samples` and `num_train_epochs`)

### Step 2.2: Verify Environment

- [ ] Check conda environment
  ```bash
  conda env list | grep deepspeed-z3
  ```

- [ ] Activate environment (test)
  ```bash
  conda activate deepspeed-z3
  python --version
  conda deactivate
  ```

- [ ] Check GPU availability
  ```bash
  nvidia-smi
  ```

## Phase 3: Training Execution

### Step 3.1: Pre-Training Checks

- [ ] Clear GPU memory
  ```bash
  pkill -f "llamafactory-cli train" || true
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 || true
  ```

- [ ] Verify dataset accessible
  ```bash
  python -c "import json; data=json.load(open('LLaMA-Factory/data/identity_sean_dpo.json')); print(f'Dataset has {len(data)} pairs')"
  ```

- [ ] Check output directory doesn't exist (or is empty)
  ```bash
  ls -la LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo/ 2>/dev/null || echo "Output directory doesn't exist (good)"
  ```

### Step 3.2: Run Training

- [ ] Start training
  ```bash
  ./scripts/training/dpo_ds2_chat_lite_hf.sh --yes
  ```

- [ ] Monitor training (in another terminal)
  ```bash
  # Watch GPU usage
  watch -n 1 nvidia-smi

  # Or watch WandB (if enabled)
  # Check WandB dashboard
  ```

- [ ] Verify training started
  - [ ] Check process is running: `ps aux | grep llamafactory`
  - [ ] Check GPU is being used: `nvidia-smi`
  - [ ] Check logs for errors

### Step 3.3: Training Completion

- [ ] Verify training completed successfully
  ```bash
  ls -lh LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo/
  ```

- [ ] Check for adapter files
  - [ ] `adapter_model.safetensors` exists
  - [ ] `adapter_config.json` exists
  - [ ] `training_loss.png` exists (if plot_loss enabled)

- [ ] Review training metrics
  - [ ] Check final loss value
  - [ ] Review loss curve (if available)
  - [ ] Check WandB run (if enabled)

## Phase 4: Evaluation

### Step 4.1: Prepare Evaluation

- [ ] Verify all adapters exist
  ```bash
  ls -d LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_*
  ```

- [ ] Clear GPU memory before evaluation
  ```bash
  pkill -f "llamafactory-cli" || true
  python -c "import torch; torch.cuda.empty_cache()" || true
  ```

### Step 4.2: Run Evaluation

- [ ] Run evaluation script
  ```bash
  ./scripts/evaluation/run_evaluation.sh
  ```

- [ ] OR run with explicit paths
  ```bash
  python scripts/evaluation/evaluate_raw_vs_adapter.py \
      --adapters sft:LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized \
      --adapters dpo:LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo \
      --benchmarks all \
      --num_samples 50 \
      --output evaluation_results_dpo.json
  ```

- [ ] Verify evaluation completed
  ```bash
  ls -lh evaluation_results*.json
  ```

### Step 4.3: Generate Report

- [ ] Generate markdown report
  ```bash
  python scripts/rl_training/generate_comparison_report.py \
      evaluation_results_dpo.json \
      --output comparison_report.md
  ```

- [ ] Review report
  ```bash
  cat comparison_report.md | less
  ```

## Phase 5: Analysis & Documentation

### Step 5.1: Analyze Results

- [ ] Compare identity accuracy
  - [ ] Raw model: [ ]%
  - [ ] SFT adapter: [ ]%
  - [ ] DPO adapter: [ ]%
  - [ ] Target: 100% ✓

- [ ] Compare general capabilities
  - [ ] MMLU: Raw [ ]% → SFT [ ]% → DPO [ ]%
  - [ ] GSM8K: Raw [ ]% → SFT [ ]% → DPO [ ]%
  - [ ] TruthfulQA: Raw [ ]% → SFT [ ]% → DPO [ ]%
  - [ ] HellaSwag: Raw [ ]% → SFT [ ]% → DPO [ ]%

- [ ] Identify best model per benchmark
  - [ ] MMLU: [ ] Raw / SFT / DPO
  - [ ] GSM8K: [ ] Raw / SFT / DPO
  - [ ] TruthfulQA: [ ] Raw / SFT / DPO
  - [ ] Identity: [ ] Raw / SFT / DPO

### Step 5.2: Document Findings

- [ ] Create results document
  ```bash
  cp docs/RL_TRAINING_RESULTS_TEMPLATE.md docs/RL_TRAINING_RESULTS.md
  ```

- [ ] Fill in results template
  - [ ] Training configuration
  - [ ] Training metrics
  - [ ] Evaluation results
  - [ ] Comparison analysis
  - [ ] Recommendations

- [ ] Update implementation summary (if needed)

## Phase 6: Next Steps

### Decision Point: DPO Performance

- [ ] **If DPO performs well:**
  - [ ] Document success
  - [ ] Consider production deployment
  - [ ] Plan further optimizations (if needed)

- [ ] **If DPO underperforms:**
  - [ ] Try KTO training (Task 5.2)
  - [ ] Try ORPO/SimPO variants (Task 5.3)
  - [ ] Adjust hyperparameters
  - [ ] Generate better DPO data (LLM-based rejections)

### Optional: Try Alternatives

- [ ] **KTO Training** (if DPO underperforms)
  - [ ] Create KTO config
  - [ ] Generate KTO dataset
  - [ ] Run KTO training
  - [ ] Compare KTO vs DPO vs SFT

- [ ] **ORPO/SimPO Variants**
  - [ ] Try `pref_loss: orpo`
  - [ ] Try `pref_loss: simpo`
  - [ ] Compare variants

## Quick Reference

### File Locations

- **DPO Dataset:** `LLaMA-Factory/data/identity_sean_dpo.json`
- **DPO Adapter:** `LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo/`
- **Evaluation Results:** `evaluation_results_dpo.json`
- **Comparison Report:** `comparison_report.md`

### Key Commands

```bash
# Generate data
./scripts/sft_data/generate_identity_dpo.sh --num_records 100

# Register dataset
./scripts/rl_training/register_dpo_dataset.sh data_file.json

# Train
./scripts/training/dpo_ds2_chat_lite_hf.sh --yes

# Evaluate
./scripts/evaluation/run_evaluation.sh

# Generate report
python scripts/rl_training/generate_comparison_report.py results.json
```

### Troubleshooting

- **Dataset not found:** Check `dataset_info.json` registration
- **Training fails:** Check GPU memory, reduce batch size
- **Evaluation fails:** Clear GPU memory, check adapter paths
- **See:** `docs/RL_TRAINING_QUICK_START.md` troubleshooting section

---

**Status:** [ ] Not Started / [ ] In Progress / [ ] Complete  
**Last Updated:** [DATE]
