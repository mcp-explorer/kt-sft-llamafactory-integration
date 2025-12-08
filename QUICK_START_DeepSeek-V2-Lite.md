# Quick Start Guide: Testing kt-sft with DeepSeek-V2-Lite

## Step 1: Setup Swap (if needed)

You currently have 93GB swap. If you want additional swap for safety:

```bash
cd /home/sean/Documents/ktransformers
./setup_swap.sh
```

This will add 50GB more swap space (total ~143GB).

## Step 2: Environment Setup

```bash
# Create conda environment
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit
conda create -n Kllama python=3.12
conda activate Kllama
conda install -y -c conda-forge libstdcxx-ng gcc_impl_linux-64
conda install -y -c nvidia/label/cuda-11.8.0 cuda-runtime
pip install -r requirements-sft.txt
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

# Install LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torc,metrics]" --no-build-isolation
cd ..

# If you want to use flash_infer (otherwise it defaults to triton)
git clone https://github.com/kvcache-ai/custom_flashinfer.git
pip install custom_flashinfer/

# Install KTransformers wheel (download from releases page first)
# Check your Python/Torch versions first:
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('No GPU')"
python check_gpu.py
# Then install matching wheel from: https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.1
pip install https://github.com/kvcache-ai/ktransformers/releases/download/v0.4.1/ktransformers-0.4.1+cu128torch27fancy-cp312-cp312-linux_x86_64.whl

# Install flash-attention
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"  # Check ABI
# Download matching wheel from: https://github.com/Dao-AILab/flash-attention/releases
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com/cxx11 https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## Step 3: Create Training Config

Create `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_kt.yaml`:

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
dataset: identity
template: deepseek
cutoff_len: 2048
max_samples: 1000
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
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### ktransformers
use_kt: true
kt_optimize_rule: ../kt-sft/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml
cpu_infer: 16
chunk_size: 8192
```

**Note**: If you have AMX support (`lscpu | grep amx`), change the optimize rule to:
```yaml
kt_optimize_rule: ../kt-sft/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft-amx.yaml
```

## Step 4: Run Training Test

```bash
cd LLaMA-Factory
USE_KT=1 llamafactory-cli train examples/train_lora/deepseek2_lite_sft_kt.yaml
```

## Step 5: Test Inference

Create `LLaMA-Factory/examples/inference/deepseek2_lite_inference.yaml`:

```yaml
model_name_or_path: deepseek-ai/DeepSeek-V2-Lite
adapter_name_or_path: saves/Kllama_deepseekV2Lite_test
template: deepseek
infer_backend: ktransformers
trust_remote_code: true

use_kt: true
kt_optimize_rule: ../kt-sft/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml
cpu_infer: 16
chunk_size: 8192
```

Then test:
```bash
cd LLaMA-Factory
llamafactory-cli chat examples/inference/deepseek2_lite_inference.yaml
```

## Expected Results

- **GPU Memory**: ~5.5-6GB
- **RAM**: ~20-30GB
- **Training Speed**: ~200-530 tokens/s
- **Training Time**: ~1-2 hours for 1000 samples (1 epoch)

## Troubleshooting

- **OOM Error**: Reduce batch size or increase swap
- **CUDA Error**: Check `nvidia-smi` and CUDA version
- **Import Error**: Verify all packages installed correctly

For detailed information, see `TEST_PLAN_DeepSeek-V2-Lite.md`.

