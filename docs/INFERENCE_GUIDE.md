# Inference Guide for DeepSeek-V2-Lite-Chat (LoRA Fine-tuned)

## Overview

After training your LoRA adapter with DeepSpeed ZeRO-3, you can use it for inference in three modes:
1. **Interactive CLI Chat** - Command-line interface for testing
2. **Web UI Chat** - Browser-based chat interface
3. **API Server** - OpenAI-compatible API for programmatic access

## Quick Start

### 1. Interactive CLI Chat (Recommended for Testing)

```bash
./scripts/inference/infer_ds2_chat_lite_hf.sh chat
```

This starts an interactive chat session. Type your messages and press Enter. Type `exit` or `quit` to end.

**Example:**
```
User: Who are you?
Assistant: [Model response]

User: exit
```

### 2. Web UI Chat Interface

```bash
./scripts/inference/infer_ds2_chat_lite_hf.sh webchat
```

Then open your browser and navigate to:
- **URL:** http://localhost:7860

Press `Ctrl+C` in the terminal to stop the server.

### 3. OpenAI-style API Server

```bash
./scripts/inference/infer_ds2_chat_lite_hf.sh api
```

The API will be available at:
- **URL:** http://localhost:8000

**Example API Usage:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v2-lite",
    "messages": [
      {"role": "user", "content": "Hello! Who are you?"}
    ]
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "deepseek-v2-lite",
        "messages": [
            {"role": "user", "content": "Hello! Who are you?"}
        ]
    }
)
print(response.json())
```

## Using Specific Checkpoints

If you want to test a specific checkpoint from training:

```bash
# Use checkpoint-10
./scripts/inference/infer_ds2_chat_lite_hf.sh chat 10

# Use checkpoint-16 (final checkpoint)
./scripts/inference/infer_ds2_chat_lite_hf.sh chat 16
```

## Direct LLaMA-Factory Command

You can also run inference directly using LLaMA-Factory CLI:

```bash
cd LLaMA-Factory
llamafactory-cli chat examples/inference/deepseek2_lite_inference_hf.yaml
```

## Configuration

The inference configuration is located at:
- `LLaMA-Factory/examples/inference/deepseek2_lite_inference_hf.yaml`

**Current Settings:**
- **Model:** `/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat`
- **Adapter:** `/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized`
- **Template:** `deepseek` (chat template)
- **Backend:** `huggingface`

## Troubleshooting

### Adapter Not Found

If you get an error about the adapter not being found, check:
1. The adapter path in the config file
2. That training completed successfully
3. That adapter files exist:
   ```bash
   ls -l LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized/adapter*
   ```

### Out of Memory

If you encounter OOM errors during inference:
1. The model will automatically use CPU offloading if needed
2. You can reduce batch size in the config
3. Consider using quantization (4-bit/8-bit) for inference

### Conda Environment

The script uses the `deepspeed-z3` conda environment for consistency with training. This environment has all required packages (transformers, peft, llamafactory) and works perfectly for inference, even though DeepSpeed itself is not needed for inference.

## Model Output Location

Trained adapters are saved at:
- **Final Adapter:** `LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized/`
- **Checkpoints:** `LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized/checkpoint-*/`

Each checkpoint contains:
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - LoRA weights
- `trainer_state.json` - Training state

## Next Steps

1. **Test the model** with interactive chat to verify it works
2. **Evaluate performance** on your test set
3. **Deploy** using the API server for production use
4. **Fine-tune further** if needed based on results

