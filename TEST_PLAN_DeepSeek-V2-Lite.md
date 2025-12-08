# Test Plan: kt-sft for DeepSeek-V2-Lite

## Overview
This plan outlines the steps to test kt-sft (KTransformers Fine-Tuning) with DeepSeek-V2-Lite (14B MoE model) using LLaMA-Factory integration.

## System Requirements

### Current System Status
- **RAM**: 31GB total, 24GB available
- **Swap**: 93GB (existing)
- **Disk Space**: 327GB free
- **GPU**: Check with `nvidia-smi` (requires ~6GB VRAM for DeepSeek-V2-Lite)

### Recommended Resources
- **GPU Memory**: ~5.5-6GB VRAM
- **RAM**: ~30GB (you have 31GB, should be sufficient)
- **Swap**: 93GB existing (may add more for safety)
- **Disk**: ~50GB for model weights + training data

## Prerequisites Checklist

### 1. Environment Setup
- [ ] Python 3.10, 3.11, 3.12, or 3.13 installed
- [ ] CUDA toolkit installed (check version compatibility)
- [ ] Conda or virtual environment manager
- [ ] Git installed

### 2. Install Dependencies

#### Step 1: Create Conda Environment
```bash
conda create -n Kllama python=3.12
conda activate Kllama
conda install -y -c conda-forge libstdcxx-ng gcc_impl_linux-64
conda install -y -c nvidia/label/cuda-11.8.0 cuda-runtime
```

#### Step 2: Install LLaMA-Factory
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
```

#### Step 3: Install KTransformers Wheel
Download the appropriate wheel from: https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.1

Match your Python and Torch versions:
```bash
# Example for Python 3.12, CUDA 12.8, Torch 2.7
pip install ktransformers-0.4.1+cu128torch27fancy-cp312-cp312-linux_x86_64.whl
```

#### Step 4: Install Flash-Attention
Download from: https://github.com/Dao-AILab/flash-attention/releases
```bash
# Check ABI compatibility first
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# Then install matching wheel
pip install flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

#### Step 5: (Optional) Install Custom FlashInfer
```bash
git clone https://github.com/kvcache-ai/custom_flashinfer.git
pip install custom_flashinfer/
```

### 3. Verify Installation
```bash
python -c "import ktransformers; print('KTransformers installed successfully')"
python -c "import llamafactory; print('LLaMA-Factory installed successfully')"
nvidia-smi  # Verify GPU is accessible
```

## Test Plan Steps

### Phase 1: Preparation

#### 1.1 Create Training Configuration
Create a training YAML file: `deepseek2_lite_sft_kt.yaml`

```yaml
### model
model_name_or_path: deepseek-ai/DeepSeek-V2-Lite
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: identity  # Use identity dataset for testing, or specify your dataset
template: deepseek
cutoff_len: 2048
max_samples: 1000  # Start with small dataset for testing
overwrite_cache: true
preprocessing_num_workers: 4
dataloader_num_workers: 2

### output
output_dir: saves/Kllama_deepseekV2Lite_test
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1.0  # Start with 1 epoch for testing
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### ktransformers
use_kt: true
kt_optimize_rule: kt-sft/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml
cpu_infer: 16
chunk_size: 8192
```

**Note**: If you have AMX support (`lscpu | grep amx`), use `DeepSeek-V2-Lite-Chat-sft-amx.yaml` instead.

#### 1.2 Prepare Dataset
- Use `identity` dataset for quick testing (generates simple data)
- Or prepare your own dataset in LLaMA-Factory format

### Phase 2: Training Test

#### 2.1 Run Training (Small Test)
```bash
cd LLaMA-Factory
USE_KT=1 llamafactory-cli train deepseek2_lite_sft_kt.yaml
```

**Expected Results**:
- Training should start without OOM errors
- GPU memory usage: ~5.5-6GB
- RAM usage: ~20-30GB
- Throughput: ~200-530 tokens/s (depending on hardware)

#### 2.2 Monitor Resources
```bash
# In another terminal
watch -n 1 nvidia-smi  # Monitor GPU
watch -n 1 free -h     # Monitor RAM/Swap
```

#### 2.3 Check Training Output
- Verify loss decreases over time
- Check that checkpoints are saved in `output_dir`
- Verify no crashes or OOM errors

### Phase 3: Inference Test

#### 3.1 Create Inference Configuration
Create `deepseek2_lite_inference.yaml`:

```yaml
model_name_or_path: deepseek-ai/DeepSeek-V2-Lite
adapter_name_or_path: saves/Kllama_deepseekV2Lite_test
template: deepseek
infer_backend: ktransformers
trust_remote_code: true

use_kt: true
kt_optimize_rule: kt-sft/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml
cpu_infer: 16
chunk_size: 8192
```

#### 3.2 Test Chat Interface
```bash
cd LLaMA-Factory
llamafactory-cli chat deepseek2_lite_inference.yaml
```

**Test Cases**:
1. Simple question-answer
2. Multi-turn conversation
3. Long context (if supported)

#### 3.3 Test Batch Inference
```bash
cd LLaMA-Factory
API_PORT=8000 llamafactory-cli api deepseek2_lite_inference.yaml
```

Test with API calls:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v2-lite",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Phase 4: Performance Validation

#### 4.1 Measure Training Throughput
- Record tokens/second from training logs
- Compare with expected: ~200-530 tokens/s

#### 4.2 Measure Memory Usage
- Peak GPU memory
- Peak RAM usage
- Swap usage (should be minimal if RAM is sufficient)

#### 4.3 Validate Model Quality
- Test fine-tuned model responses
- Compare with base model
- Check for overfitting (if using small dataset)

## Troubleshooting

### Common Issues

1. **OOM (Out of Memory) Errors**
   - Reduce `per_device_train_batch_size` to 1
   - Increase `gradient_accumulation_steps`
   - Reduce `cutoff_len`
   - Check swap usage

2. **CUDA Errors**
   - Verify CUDA version compatibility
   - Check GPU drivers: `nvidia-smi`
   - Reinstall CUDA runtime if needed

3. **Import Errors**
   - Verify all dependencies installed
   - Check Python version matches wheel
   - Verify ABI compatibility for flash-attention

4. **Slow Training**
   - Check if AMX is available and enabled
   - Verify CPU inference threads (`cpu_infer`)
   - Check GPU utilization

## Success Criteria

- [ ] Training completes without errors
- [ ] GPU memory usage < 7GB
- [ ] RAM usage < 35GB (including swap)
- [ ] Training throughput > 200 tokens/s
- [ ] Inference works correctly
- [ ] Model generates reasonable responses

## Next Steps After Testing

1. **Full Training**: Increase `max_samples` and `num_train_epochs` for real training
2. **Custom Dataset**: Prepare your own dataset
3. **Hyperparameter Tuning**: Adjust learning rate, LoRA rank, etc.
4. **Evaluation**: Run benchmarks on your fine-tuned model

## Notes

- DeepSeek-V2-Lite is a 14B parameter MoE model
- Expected GPU memory: ~5.5-6GB with kt-sft
- Expected RAM: ~30GB
- Training speed: ~530 tokens/s (with optimal hardware)
- Use AMX acceleration if available for better CPU performance

## References

- kt-sft README: `kt-sft/README.md`
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- KTransformers: https://github.com/kvcache-ai/ktransformers

