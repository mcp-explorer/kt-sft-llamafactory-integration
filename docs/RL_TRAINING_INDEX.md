# RL Training Documentation Index

Complete guide to RL (DPO) training implementation for DeepSeek-V2-Lite-Chat identity learning.

---

## 🚀 Quick Start

**New to RL training? Start here:**

1. **Check Status**
   ```bash
   ./scripts/rl_training/check_status.sh
   ```

2. **Run Complete Workflow**
   ```bash
   ./scripts/rl_training/run_complete_rl_workflow.sh
   ```

3. **Review Results**
   ```bash
   python scripts/rl_training/generate_comparison_report.py \
       evaluation_results.json --output comparison_report.md
   ```

---

## 📚 Documentation Guide

### For First-Time Users

1. **[RL_TRAINING_QUICK_START.md](RL_TRAINING_QUICK_START.md)** ⭐ START HERE
   - Step-by-step instructions
   - Usage examples
   - Troubleshooting guide
   - Expected results

2. **[RL_TRAINING_CHECKLIST.md](RL_TRAINING_CHECKLIST.md)**
   - Complete workflow checklist
   - Verification steps
   - Progress tracking

### For Understanding the Implementation

3. **[RL_TRAINING_PLAN.md](RL_TRAINING_PLAN.md)**
   - Comprehensive implementation plan
   - LLaMA-Factory RL capabilities analysis
   - DPO vs KTO vs PPO comparison
   - Detailed data generation strategy
   - Code examples and configurations

4. **[RL_TRAINING_IMPLEMENTATION_SUMMARY.md](RL_TRAINING_IMPLEMENTATION_SUMMARY.md)**
   - Complete file reference
   - File structure overview
   - Feature summary
   - Quick start commands

### For Documenting Results

5. **[RL_TRAINING_RESULTS_TEMPLATE.md](RL_TRAINING_RESULTS_TEMPLATE.md)**
   - Template for results documentation
   - Comparison tables
   - Analysis sections
   - Recommendations format

### Related Documentation

6. **[SFT_TRAINING_LOG.md](SFT_TRAINING_LOG.md)**
   - SFT training documentation
   - Baseline for comparison

7. **[EVALUATION_RESULTS_SUMMARY.md](EVALUATION_RESULTS_SUMMARY.md)**
   - SFT evaluation results
   - Baseline metrics

---

## 🛠️ Scripts Reference

### Core Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| **Data Generation** | Generate DPO preference pairs | `scripts/sft_data/generate_identity_dpo.sh` |
| **Training** | Run DPO training | `scripts/training/dpo_ds2_chat_lite_hf.sh` |
| **Evaluation** | Compare models | `scripts/evaluation/run_evaluation.sh` |

### Helper Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| **Master Workflow** | Run complete pipeline | `scripts/rl_training/run_complete_rl_workflow.sh` |
| **Status Check** | Check workflow status | `scripts/rl_training/check_status.sh` |
| **Register Dataset** | Register DPO dataset | `scripts/rl_training/register_dpo_dataset.sh` |
| **Generate Report** | Create markdown report | `scripts/rl_training/generate_comparison_report.py` |

### Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| **DPO Config** | Training configuration | `LLaMA-Factory/examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml` |
| **Dataset Info** | Dataset registry | `LLaMA-Factory/data/dataset_info.json` |

---

## 📋 Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│              RL Training Complete Workflow                   │
└─────────────────────────────────────────────────────────────┘

1. CHECK STATUS
   └─> ./scripts/rl_training/check_status.sh
       • Verify prerequisites
       • Check existing components
       • Get next steps

2. GENERATE DPO DATA
   └─> ./scripts/sft_data/generate_identity_dpo.sh
       • Input: identity_sean_generated.json
       • Output: identity_sean_dpo.json
       • Methods: template, llm, mixed

3. REGISTER DATASET
   └─> ./scripts/rl_training/register_dpo_dataset.sh
       • Copy to LLaMA-Factory/data/
       • Register in dataset_info.json

4. RUN DPO TRAINING
   └─> ./scripts/training/dpo_ds2_chat_lite_hf.sh
       • Config: deepseek2_lite_dpo_hf_z3.yaml
       • Output: saves/Kllama_deepseekV2Lite_hf_z3_dpo/

5. EVALUATE MODELS
   └─> ./scripts/evaluation/run_evaluation.sh
       • Compare: Raw vs SFT vs DPO
       • Output: evaluation_results.json

6. GENERATE REPORT
   └─> python scripts/rl_training/generate_comparison_report.py
       • Input: evaluation_results.json
       • Output: comparison_report.md

7. DOCUMENT RESULTS
   └─> Use RL_TRAINING_RESULTS_TEMPLATE.md
       • Fill in results
       • Analyze findings
       • Make recommendations
```

---

## 🎯 Common Tasks

### Task: Generate DPO Dataset

**Quick:**
```bash
./scripts/sft_data/generate_identity_dpo.sh --num_records 100
```

**With Options:**
```bash
./scripts/sft_data/generate_identity_dpo.sh \
    --input_sft LLaMA-Factory/data/identity_sean_generated.json \
    --num_records 100 \
    --rejection_method template \
    --identity sean
```

**See:** [Quick Start Guide](RL_TRAINING_QUICK_START.md#step-1-generate-dpo-dataset)

### Task: Register Dataset

**Using Helper:**
```bash
./scripts/rl_training/register_dpo_dataset.sh \
    sft_data/outputs/identity_sean_dpo_*.json
```

**Manual:**
See [Quick Start Guide](RL_TRAINING_QUICK_START.md#step-2-copy-and-register-dpo-dataset)

### Task: Run Training

**Standard:**
```bash
./scripts/training/dpo_ds2_chat_lite_hf.sh --yes
```

**Quick Test:**
```bash
# Edit config: max_samples=20, epochs=1.0
./scripts/training/dpo_ds2_chat_lite_hf.sh --yes
```

**See:** [Quick Start Guide](RL_TRAINING_QUICK_START.md#step-3-run-dpo-training)

### Task: Evaluate Models

**Auto-Detection:**
```bash
./scripts/evaluation/run_evaluation.sh
```

**Manual:**
```bash
python scripts/evaluation/evaluate_raw_vs_adapter.py \
    --adapters sft:saves/Kllama_deepseekV2Lite_hf_z3_regularized \
    --adapters dpo:saves/Kllama_deepseekV2Lite_hf_z3_dpo \
    --benchmarks all \
    --num_samples 50
```

**See:** [Quick Start Guide](RL_TRAINING_QUICK_START.md#step-4-evaluate-all-models-raw-vs-sft-vs-dpo)

### Task: Generate Report

```bash
python scripts/rl_training/generate_comparison_report.py \
    evaluation_results.json \
    --output comparison_report.md
```

**See:** [Quick Start Guide](RL_TRAINING_QUICK_START.md#step-5-view-results)

---

## 🔍 Troubleshooting

### Problem: Dataset Not Found

**Solution:**
```bash
# Check if dataset exists
ls -lh LLaMA-Factory/data/identity_sean_dpo.json

# Check registration
grep -A 5 "identity_sean_dpo" LLaMA-Factory/data/dataset_info.json

# Re-register if needed
./scripts/rl_training/register_dpo_dataset.sh \
    LLaMA-Factory/data/identity_sean_dpo.json
```

### Problem: Training Fails

**Check:**
```bash
# Check GPU memory
nvidia-smi

# Check dataset
python -c "import json; print(len(json.load(open('LLaMA-Factory/data/identity_sean_dpo.json'))))"

# Check config
cat LLaMA-Factory/examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml | grep dataset
```

**See:** [Quick Start Guide - Troubleshooting](RL_TRAINING_QUICK_START.md#troubleshooting)

### Problem: Evaluation Can't Find Adapters

**Solution:**
```bash
# Check adapter paths
ls -d LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_*

# Use explicit paths
python scripts/evaluation/evaluate_raw_vs_adapter.py \
    --adapters sft:/full/path/to/sft_adapter \
    --adapters dpo:/full/path/to/dpo_adapter
```

---

## 📊 Expected Results

### Identity Accuracy

| Model | Expected | Target |
|-------|----------|--------|
| Raw | 0% | Baseline |
| SFT | 100% | ✅ Achieved |
| DPO | 100% | ✅ Target |

### General Capabilities

| Benchmark | Raw | SFT | DPO Target |
|-----------|-----|-----|------------|
| MMLU | 50% | 46% | ≥46% |
| GSM8K | 30% | 36% | ≥36% |
| TruthfulQA | 32% | 34% | ≥34% |
| HellaSwag | 0% | 0% | 0% |

**See:** [Quick Start Guide - Expected Results](RL_TRAINING_QUICK_START.md#expected-results)

---

## 📁 File Locations

### Generated Files

- **DPO Dataset:** `LLaMA-Factory/data/identity_sean_dpo.json`
- **DPO Adapter:** `LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo/`
- **Evaluation Results:** `evaluation_results.json` or `evaluation_results_dpo.json`
- **Comparison Report:** `comparison_report.md`

### Configuration Files

- **DPO Config:** `LLaMA-Factory/examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml`
- **Dataset Info:** `LLaMA-Factory/data/dataset_info.json`

### Documentation

- **Plan:** `docs/RL_TRAINING_PLAN.md`
- **Quick Start:** `docs/RL_TRAINING_QUICK_START.md`
- **Checklist:** `docs/RL_TRAINING_CHECKLIST.md`
- **Summary:** `docs/RL_TRAINING_IMPLEMENTATION_SUMMARY.md`
- **Results Template:** `docs/RL_TRAINING_RESULTS_TEMPLATE.md`

---

## 🎓 Learning Resources

### Understanding DPO

- **DPO Paper:** [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **LLaMA-Factory Docs:** [Preference Learning](https://github.com/hiyouga/LLaMA-Factory#preference-learning)

### Related Methods

- **KTO:** Kahneman-Tversky Optimization
- **ORPO:** Odds Ratio Preference Optimization
- **SimPO:** Simple Preference Optimization

**See:** [RL Training Plan - Method Comparison](RL_TRAINING_PLAN.md#llamafactory-rl-training-capabilities)

---

## ✅ Implementation Status

- [x] Data generation script
- [x] Training configuration
- [x] Training script
- [x] Multi-model evaluation
- [x] Helper scripts
- [x] Documentation
- [x] Status checking
- [x] Report generation

**Status:** ✅ **Complete and Ready for Use**

---

## 🆘 Getting Help

1. **Check Status:** `./scripts/rl_training/check_status.sh`
2. **Review Checklist:** `docs/RL_TRAINING_CHECKLIST.md`
3. **See Troubleshooting:** `docs/RL_TRAINING_QUICK_START.md#troubleshooting`
4. **Review Plan:** `docs/RL_TRAINING_PLAN.md`

---

**Last Updated:** January 15, 2026  
**Version:** 1.0  
**Status:** ✅ Production Ready
