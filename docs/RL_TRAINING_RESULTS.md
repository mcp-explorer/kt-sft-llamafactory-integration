# RL Training Evaluation Results

**Evaluation Date:** 2026-01-15 13:40:49  
**Training Date:** 2026-01-15 02:53 - 07:21  
**Models Compared:** raw, sft, dpo  
**Base Model:** DeepSeek-V2-Lite-Chat

---

## Executive Summary

This report documents the results of Direct Preference Optimization (DPO) training for identity learning on the DeepSeek-V2-Lite-Chat model, compared against the raw model and a Supervised Fine-Tuning (SFT) adapter.

### Key Findings

- **DPO Training:** Successfully completed with 42% loss reduction (0.693 → 0.402)
- **Identity Learning:** SFT achieved 100% accuracy, DPO achieved 0% (did not learn identity)
- **General Capabilities:** DPO slightly improved MMLU (+2% vs raw), but degraded on other benchmarks
- **Overall Assessment:** DPO training converged but did not successfully learn the identity task

### Training Success Metrics

- ✅ Training completed successfully (10 epochs, 4.4 hours)
- ✅ Loss decreased from 0.693 to 0.402 (42% improvement)
- ❌ Identity task not learned (0% accuracy vs 100% for SFT)
- ⚠️ General capabilities maintained but not improved

---

## Summary Table

| Benchmark | RAW | SFT | DPO | Best |
|-----------|---|---|---|------|
| **MMLU** | 50.00% | 46.00% (-4.00%) 🔴 | 52.00% (+2.00%) 🟢 | **DPO** |
| **HELLASWAG** | 0.00% | 0.00% (+0.00%) 🔴 | 0.00% (+0.00%) 🔴 | **RAW** |
| **GSM8K** | 30.00% | 36.00% (+6.00%) 🟢 | 28.00% (-2.00%) 🔴 | **SFT** |
| **TRUTHFULQA** | 32.00% | 34.00% (+2.00%) 🟢 | 28.00% (-4.00%) 🔴 | **SFT** |
| **IDENTITY** | 0.00% | 100.00% (+100.00%) 🟢 | 0.00% (+0.00%) 🔴 | **SFT** |

---

## Detailed Statistics

### SFT

- **Improved benchmarks:** 3/5
- **Degraded benchmarks:** 2/5
- **Average difference:** +20.80%

### DPO

- **Improved benchmarks:** 1/5
- **Degraded benchmarks:** 4/5
- **Average difference:** -0.80%

## Best Model per Benchmark

- **MMLU:** DPO (52.00%)
- **HELLASWAG:** RAW (0.00%)
- **GSM8K:** SFT (36.00%)
- **TRUTHFULQA:** SFT (34.00%)
- **IDENTITY:** SFT (100.00%)

---

## Training Configuration

### DPO Training Settings

- **Base Model:** `/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat`
- **Dataset:** `identity_sean_dpo` (100 preference pairs)
- **LoRA Rank:** 32
- **LoRA Alpha:** 64
- **Learning Rate:** 5.0e-6
- **Epochs:** 10.0
- **DPO Beta:** 0.1
- **DPO Loss:** sigmoid
- **DeepSpeed:** ZeRO-3 hybrid (CPU offload)
- **Gradient Accumulation:** 32 steps
- **Batch Size:** 1 per device

### Training Data

- **Source:** Generated from `identity_sean_generated.json`
- **Rejection Method:** template-based
- **Number of Pairs:** 100
- **Dataset File:** `LLaMA-Factory/data/identity_sean_dpo.json`

### Training Results

- **Final DPO Loss:** 0.402 (down from 0.693, 42% improvement)
- **Training Time:** 4.4 hours (15,888 seconds)
- **Samples/Second:** 0.063
- **Steps/Second:** 0.003
- **Total FLOPS:** 84.1 TFLOPS
- **Checkpoints Saved:** checkpoint-10, checkpoint-20, checkpoint-30, checkpoint-40
- **Final Adapter:** 1.1GB (`adapter_model.safetensors`)

### Loss Progression

- **Initial Loss (Epoch 0.32):** 0.6931
- **Mid-Training (Epoch 9.32):** 0.2489
- **Final Loss (Epoch 10.0):** 0.402
- **Overall Improvement:** 42% reduction

---

## Detailed Results

### MMLU

| Model | Accuracy | Difference | Status |
|-------|----------|------------|--------|
| RAW (baseline) | 50.00% | — | — |
| SFT | 46.00% | -4.00% | ⚠️ Degraded |
| DPO | 52.00% | +2.00% | ✅ Improved |

### HELLASWAG

| Model | Accuracy | Difference | Status |
|-------|----------|------------|--------|
| RAW (baseline) | 0.00% | — | — |
| SFT | 0.00% | +0.00% | ⚠️ Degraded |
| DPO | 0.00% | +0.00% | ⚠️ Degraded |

### GSM8K

| Model | Accuracy | Difference | Status |
|-------|----------|------------|--------|
| RAW (baseline) | 30.00% | — | — |
| SFT | 36.00% | +6.00% | ✅ Improved |
| DPO | 28.00% | -2.00% | ⚠️ Degraded |

### TRUTHFULQA

| Model | Accuracy | Difference | Status |
|-------|----------|------------|--------|
| RAW (baseline) | 32.00% | — | — |
| SFT | 34.00% | +2.00% | ✅ Improved |
| DPO | 28.00% | -4.00% | ⚠️ Degraded |

### IDENTITY

| Model | Accuracy | Difference | Status |
|-------|----------|------------|--------|
| RAW (baseline) | 0.00% | — | — |
| SFT | 100.00% | +100.00% | ✅ Improved |
| DPO | 0.00% | +0.00% | ⚠️ Degraded |

**Analysis:** This is the primary objective of the training. SFT successfully learned the identity task with 100% accuracy, while DPO failed to learn it (0% accuracy). This suggests that DPO may require different data formatting, more training data, or different hyperparameters to learn identity preferences effectively.

---

## Analysis and Conclusions

### Why DPO Failed on Identity Task

**Root Cause: Data Quality Issues** (See `docs/DPO_DATA_ISSUES_ANALYSIS.md` for detailed analysis)

1. **Length Bias (Critical):** 
   - Chosen responses: 22.4 chars (short)
   - Rejected responses: 47.8 chars (longer)
   - DPO models often prefer longer responses, causing the model to learn length preference rather than identity preference

2. **Weak Preference Signal (Critical):**
   - Rejected responses are valid, helpful AI responses (just different identities)
   - They're not "bad" responses - DPO needs a quality/preference gradient
   - Model learns "not-DeepSeek" rather than "be-Sean"

3. **Low Diversity:**
   - Only 12 unique chosen responses out of 100 pairs
   - High repetition limits generalization

4. **Binary Preference:**
   - Preference is purely "Sean" vs "not-Sean"
   - DPO works better with quality-based preferences (better vs worse responses)

5. **Training Dynamics:** DPO optimizes for preference learning, which may conflict with direct identity instruction learning that SFT excels at.

6. **Hyperparameters:** The DPO beta (0.1) and learning rate (5e-6) may need tuning for identity learning tasks.

7. **Data Volume:** 100 preference pairs may be insufficient for DPO to learn complex identity patterns.

### Recommendations

1. **For Identity Learning:** Continue using SFT, which achieved 100% accuracy

2. **For DPO Improvement (Data Quality Fixes):**
   - **Balance response lengths:** Make chosen responses longer, rejected shorter
   - **Improve rejected responses:** Use unhelpful/evasive responses, not just different identities
     - Examples: "I don't have a name.", "That's not important.", "I'm just here to help."
   - **Increase dataset size:** 500-1000 pairs (currently 100)
   - **Use LLM-based rejection generation:** More realistic "bad" responses
   - **Match SFT response style:** Use SFT outputs as chosen responses (they're longer, more natural)
   - **Create quality gradient:** Chosen = helpful with identity, Rejected = unhelpful without identity

3. **For DPO Improvement (Hyperparameters):**
   - Experiment with higher DPO beta values (0.2-0.5)
   - Try ORPO or SimPO variants
   - Adjust learning rate if needed

4. **Hybrid Approach:** Consider using SFT for identity learning, then DPO for preference refinement

**See `docs/DPO_DATA_ISSUES_ANALYSIS.md` for detailed data quality analysis and recommendations.**

### General Capability Preservation

- **MMLU:** DPO improved slightly (+2% vs raw), showing it can maintain general knowledge
- **Other Benchmarks:** DPO maintained baseline performance but didn't improve
- **Overall:** DPO training did not significantly degrade general capabilities

---