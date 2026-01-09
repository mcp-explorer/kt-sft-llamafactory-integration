# LLaMA-Factory Fine-Tuning Capabilities

## Training Stages (stage parameter)

LLaMA-Factory supports multiple training stages:

1. **`pt` - Pre-Training** (Continuous pre-training)
   - Train from scratch or continue pre-training
   - Example: `examples/train_lora/llama3_lora_pretrain.yaml`

2. **`sft` - Supervised Fine-Tuning** (Most common)
   - Instruction following, chat fine-tuning
   - Supports multimodal (images, videos, audio)
   - Example: `examples/train_lora/llama3_lora_sft.yaml`

3. **`rm` - Reward Modeling**
   - Train reward models for RLHF
   - Example: `examples/train_lora/llama3_lora_reward.yaml`

4. **`ppo` - Proximal Policy Optimization**
   - Reinforcement learning from human feedback
   - Example: `examples/train_lora/llama3_lora_ppo.yaml`

5. **`dpo` - Direct Preference Optimization**
   - Direct preference learning (alternative to PPO)
   - Supports ORPO, SimPO variants
   - Example: `examples/train_lora/llama3_lora_dpo.yaml`

6. **`kto` - Kahneman-Tversky Optimization**
   - Preference optimization method
   - Example: `examples/train_lora/llama3_lora_kto.yaml`

## Fine-Tuning Methods (finetuning_type parameter)

1. **`lora` - LoRA (Low-Rank Adaptation)** ⭐ Most Popular
   - Efficient parameter-efficient fine-tuning
   - Supports: DoRA, LongLoRA, LoRA+, LoftQ, PiSSA, rsLoRA
   - Example: `examples/train_lora/llama3_lora_sft.yaml`

2. **`oft` - OFT (Orthogonal Fine-Tuning)**
   - Orthogonal parameter updates
   - Example: `examples/extras/oft/llama3_oft_sft.yaml`

3. **`freeze` - Freeze Tuning**
   - Freeze most layers, train only specific layers
   - Example: `examples/extras/llama_pro/llama3_freeze_sft.yaml`

4. **`full` - Full Parameter Fine-Tuning**
   - Train all parameters (requires most memory)
   - Supports: GaLore, BAdam, APOLLO, Adam-mini, Muon
   - Example: `examples/train_full/llama3_full_sft.yaml`

## Quantization Options

- **QLoRA**: 2/3/4/5/6/8-bit quantization
- **Methods**: AQLM, AWQ, GPTQ, LLM.int8, HQQ, EETQ, BitsAndBytes
- **Example**: `examples/train_qlora/llama3_lora_sft_gptq.yaml`

## JSON/JSONL Support

### Input Data Formats

LLaMA-Factory supports multiple data formats:

1. **JSON** (`.json` files)
   - Standard JSON format for datasets
   - Example: `data/identity_sean.json`

2. **JSONL** (`.jsonl` files)
   - JSON Lines format (one JSON object per line)
   - More efficient for large datasets
   - Example: `data/c4_demo.jsonl`

3. **CSV** (`.csv` files)
4. **Parquet** (`.parquet` files)
5. **Arrow** (`.arrow` files)
6. **Text** (`.txt` files)

### Output Formats

1. **Model Export**:
   - Standard HuggingFace format (safetensors or .bin)
   - Config files (JSON format)
   - Tokenizer files

2. **Training Logs**:
   - `trainer_log.jsonl` - Training metrics in JSONL format
   - `trainer_state.json` - Training state in JSON format
   - `eval_results.json` - Evaluation results in JSON format

3. **Predictions**:
   - Can export predictions as JSONL
   - Example: `generated_predictions.jsonl`

## Export Capabilities

Export trained models using:

```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

Export options:
- **Merge LoRA adapters** into base model
- **Quantization** (GPTQ, AWQ, etc.)
- **Model sharding** (specify size in GB)
- **Push to HuggingFace Hub**
- **Legacy format** (.bin) or **SafeTensors** (default)

## Advanced Features

### Optimizers
- **Adam-mini**: Memory-efficient optimizer
- **Muon**: Advanced optimizer
- **BAdam**: Block-wise Adam

### Algorithms
- **GaLore**: Gradient Low-Rank Projection
- **APOLLO**: Adaptive optimizer
- **DoRA**: Weight-Decomposed Low-Rank Adaptation
- **LLaMA Pro**: Progressive block expansion

### Memory Optimizations
- **Gradient Checkpointing**: Reduce memory usage
- **DeepSpeed ZeRO**: ZeRO-0/2/3 for distributed training
- **FSDP**: Fully Sharded Data Parallel
- **CPU Offloading**: Offload optimizer states to CPU

### Multimodal Support
- **Images**: Vision-language models (LLaVA, Qwen2-VL)
- **Videos**: Video understanding
- **Audio**: Audio-text models

### Tool Usage
- Fine-tune models for function calling
- Tool-using capabilities (GLM4, Llama3, Mistral)

## Example Commands

### Supervised Fine-Tuning (SFT)
```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

### DPO Training
```bash
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
```

### PPO Training
```bash
llamafactory-cli train examples/train_lora/llama3_lora_ppo.yaml
```

### Export Model
```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### Evaluation
```bash
llamafactory-cli eval examples/train_lora/llama3_lora_eval.yaml
```

## Configuration Files

All configurations use YAML format, but:
- **Dataset files**: JSON/JSONL format
- **Training logs**: JSON/JSONL format
- **Model configs**: JSON format (HuggingFace standard)
- **Export configs**: YAML format

## Summary

LLaMA-Factory is a comprehensive fine-tuning framework that supports:
- ✅ 6 training stages (pt, sft, rm, ppo, dpo, kto)
- ✅ 4 fine-tuning methods (lora, oft, freeze, full)
- ✅ Multiple quantization methods
- ✅ JSON/JSONL input and output formats
- ✅ Model export and deployment
- ✅ Multimodal training (images, videos, audio)
- ✅ Advanced optimizers and algorithms
- ✅ Memory-efficient training options

For your current use case (identity fine-tuning), you're using:
- **Stage**: `sft` (Supervised Fine-Tuning)
- **Method**: `lora` (LoRA)
- **Quantization**: `4-bit QLoRA` (BitsAndBytes)
- **Data Format**: JSON (`identity_sean.json`)

