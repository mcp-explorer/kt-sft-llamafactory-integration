# RL Training Quick Reference Card

One-page quick reference for RL (DPO) training commands and workflows.

---

## 🚀 Quick Commands

### Check Status
```bash
./scripts/rl_training/check_status.sh
```

### Complete Workflow
```bash
# Full workflow
./scripts/rl_training/run_complete_rl_workflow.sh

# Quick test
./scripts/rl_training/run_complete_rl_workflow.sh --quick-test
```

### Individual Steps

**1. Generate DPO Data**
```bash
./scripts/sft_data/generate_identity_dpo.sh --num_records 100
```

**2. Register Dataset**
```bash
./scripts/rl_training/register_dpo_dataset.sh \
    sft_data/outputs/identity_sean_dpo_*.json
```

**3. Train**
```bash
./scripts/training/dpo_ds2_chat_lite_hf.sh --yes
```

**4. Evaluate**
```bash
./scripts/evaluation/run_evaluation.sh
```

**5. Generate Report**
```bash
python scripts/rl_training/generate_comparison_report.py \
    evaluation_results.json --output comparison_report.md
```

---

## 📁 Key Files

| File | Purpose | Location |
|------|---------|----------|
| **DPO Dataset** | Preference pairs | `LLaMA-Factory/data/identity_sean_dpo.json` |
| **DPO Adapter** | Trained model | `LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo/` |
| **Results** | Evaluation output | `evaluation_results.json` |
| **Report** | Markdown report | `comparison_report.md` |
| **Config** | Training config | `LLaMA-Factory/examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml` |

---

## ⚙️ Configuration

### DPO Training Config
```yaml
stage: dpo
dataset: identity_sean_dpo
pref_beta: 0.1
pref_loss: sigmoid
learning_rate: 5.0e-6
num_train_epochs: 10.0
```

### Dataset Registration
```json
{
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
}
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Dataset not found | `./scripts/rl_training/register_dpo_dataset.sh` |
| Training fails | Check GPU memory, reduce batch size |
| Evaluation fails | Clear GPU memory, check adapter paths |
| Status unclear | `./scripts/rl_training/check_status.sh` |

---

## 📊 Expected Results

| Benchmark | Raw | SFT | DPO Target |
|-----------|-----|-----|------------|
| Identity | 0% | 100% | 100% |
| MMLU | 50% | 46% | ≥46% |
| GSM8K | 30% | 36% | ≥36% |

---

## 📚 Documentation

- **Index:** `docs/RL_TRAINING_INDEX.md`
- **Quick Start:** `docs/RL_TRAINING_QUICK_START.md`
- **Checklist:** `docs/RL_TRAINING_CHECKLIST.md`
- **Full Plan:** `docs/RL_TRAINING_PLAN.md`

---

**Last Updated:** January 15, 2026
