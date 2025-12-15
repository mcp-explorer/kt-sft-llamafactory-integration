# Model Running Guide - DeepSeek-V2-Lite with Checkpoint-11

## ✅ Status: Model Successfully Loaded!

The model has been loaded successfully in the Docker container:
- **Base Model**: DeepSeek-V2-Lite (29.3 GB total)
- **Adapter**: checkpoint-11 (LoRA fine-tuned, 26 MB)
- **Total Parameters**: 276,772,352
- **Status**: Ready for inference

## Configuration File

Created: `/app/examples/inference/deepseek2_lite_serve_custom.yaml`

```yaml
### model
model_name_or_path: /app/models/deepseek-ai/DeepSeek-V2-Lite
adapter_name_or_path: /app/saves/Kllama_deepseekV2Lite/checkpoint-11
trust_remote_code: true

### method
stage: sft
finetuning_type: lora

### dataset
template: deepseek

### ktransformers
infer_backend: ktransformers
use_kt: true
kt_optimize_rule: /opt/conda/lib/python3.11/site-packages/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml
cpu_infer: 16
chunk_size: 8192
```

## How to Run the Model

### Option 1: Interactive Chat (Recommended)

```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose exec llamafactory bash

# Inside container
cd /app
llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml
```

Then you can chat interactively:
- Type your message and press Enter
- Use `clear` to clear history
- Use `exit` to exit

### Option 2: Run in Background

```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose exec -d llamafactory bash -c "cd /app && llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml"
```

### Option 3: API Server

Start the API server:

```bash
docker compose exec llamafactory bash
cd /app
llamafactory-cli api examples/inference/deepseek2_lite_serve_custom.yaml
```

Then access the API at: http://localhost:8000

## Model Details

- **Model Type**: DeepSeek-V2-Lite
- **Architecture**: MoE (Mixture of Experts)
- **Hidden Size**: 2048
- **Layers**: 27
- **Attention Heads**: 16
- **Vocabulary Size**: 102,400
- **Max Position**: 163,840
- **LoRA Adapter**: r=8, alpha=16

## Resource Usage

When the model is loaded, you should see:
- **GPU Memory**: Increased (model weights loaded to GPU)
- **GPU Utilization**: Higher during inference
- **CPU Usage**: Moderate (preprocessing, tokenization)

Monitor with:
```bash
# GPU usage
docker compose exec llamafactory nvidia-smi

# Container stats
docker stats llamafactory
```

## Troubleshooting

### Model Not Loading
- Check model files are accessible: `ls -la /app/models/deepseek-ai/DeepSeek-V2-Lite/`
- Check adapter files: `ls -la /app/saves/Kllama_deepseekV2Lite/checkpoint-11/`

### Out of Memory
- Reduce `cpu_infer` value in config
- Reduce `chunk_size` value
- Check available GPU memory: `nvidia-smi`

### Slow Loading
- First load takes time (model optimization)
- Subsequent loads are faster (cached)

## Next Steps

1. **Start chatting**: Run the interactive chat command
2. **Test the model**: Ask questions to verify it's working
3. **Monitor resources**: Check GPU/memory usage during inference
4. **Optimize if needed**: Adjust `cpu_infer` and `chunk_size` for your hardware

## Success Indicators

✅ Model config loaded  
✅ Tokenizer loaded  
✅ Adapter weights loaded (all layers)  
✅ "Welcome to the CLI application" message  
✅ Ready for user input  

The model is now ready to use!

