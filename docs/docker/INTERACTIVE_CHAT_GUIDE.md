# Interactive Chat Guide

## Enter Docker Container

To enter the Docker container and run chat interactively:

```bash
docker exec -it llamafactory bash
```

## Inside the Container

Once inside, navigate to the app directory and set up the environment:

```bash
cd /app
export KSFT_MOE_DEBUG=1  # Optional: Enable debug output
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages:$LD_LIBRARY_PATH
```

## Run Chat Interactively

```bash
llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml
```

## Alternative: One-Line Command

You can also run everything in one command:

```bash
docker exec -it llamafactory bash -c "cd /app && export KSFT_MOE_DEBUG=1 && export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages:\$LD_LIBRARY_PATH && llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml"
```

## Troubleshooting

If you get library errors, ensure the ggml libraries are copied:

```bash
docker exec llamafactory bash -c "cp /tmp/kt-sft-build/csrc/ktransformers_ext/build/bin/libggml*.so* /opt/conda/lib/python3.11/site-packages/ 2>&1"
```

## Exit Chat

Type `/exit` or press `Ctrl+C` to exit the chat interface.

