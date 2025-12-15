# LLaMA-Factory Docker Setup Complete

## Status: ✅ Successfully Built and Running

The LLaMA-Factory Docker container has been built and is ready to use!

## Container Information

- **Image**: `docker-cuda-llamafactory`
- **Container Name**: `llamafactory`
- **Base Image**: `hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel`
- **Python**: 3.11
- **PyTorch**: 2.6.0
- **CUDA**: 12.4
- **Flash Attention**: 2.7.4

## Usage

### Enter the Container

```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose exec llamafactory bash
```

### Run LLaMA-Factory Commands

Once inside the container, you can use LLaMA-Factory commands:

```bash
# Chat with your model
llamafactory-cli chat examples/inference/deepseek2_lite_serve.yaml

# Or use the Python API
python -m llamafactory.api.app
```

### Test KTransformers

```bash
# Inside container
python -c "import ktransformers; print('KTransformers installed!')"
```

## Container Management

### Start Container
```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose up -d
```

### Stop Container
```bash
docker compose down
```

### View Logs
```bash
docker compose logs llamafactory
```

### Restart Container
```bash
docker compose restart llamafactory
```

## Ports

- **7860**: LLaMA Board (Gradio UI)
- **8000**: API service

Access from host:
- http://localhost:7860 (LLaMA Board)
- http://localhost:8000 (API)

## Working Directory

The container's working directory is `/app`, which contains the LLaMA-Factory code.

## Advantages

✅ **No build issues**: Pre-configured environment with all dependencies  
✅ **CUDA 12.4**: Compatible with your setup  
✅ **KTransformers included**: Already installed and ready to use  
✅ **GPU support**: Full NVIDIA GPU access  
✅ **Isolated**: Doesn't affect your host system  

## Next Steps

1. Enter the container: `docker compose exec llamafactory bash`
2. Navigate to examples: `cd examples/inference`
3. Run your model: `llamafactory-cli chat deepseek2_lite_serve.yaml`

## Troubleshooting

### Container Not Starting
```bash
docker compose ps
docker compose logs llamafactory
```

### GPU Not Detected
```bash
# Inside container
nvidia-smi
```

### Permission Issues
```bash
# On host
sudo usermod -aG docker $USER
# Log out and back in
```

## Notes

- The container includes KTransformers and all dependencies
- No need to rebuild KTransformers - it's already installed!
- The build process completed successfully without the math.h issues
- You can now use LLaMA-Factory with KTransformers for serving your model

