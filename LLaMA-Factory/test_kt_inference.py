"""Simple test to verify ktransformers inference works correctly."""
import torch
from transformers import AutoTokenizer, AutoConfig

# Import ktransformers components - install_patch MUST be called first!
from ktransformers.sft.monkey_patch_torch_module import install_patch
install_patch()

from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.util.utils import prefill_and_generate
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.server.config.config import Config
from ktransformers.util.globals import GLOBAL_CONFIG

MODEL_PATH = "/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat"
OPTIMIZE_RULE = "/home/sean/Documents/ktransformers/LLaMA-Factory/minimal_optimize.yaml"

# Set ktransformers config
Config().cpu_infer = 16

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model config...")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
torch.set_default_dtype(config.torch_dtype)

print("Creating model on meta device...")
with torch.device("meta"):
    model = DeepseekV2ForCausalLM(config)

print(f"Loading model with optimize rule: {OPTIMIZE_RULE}")
GLOBAL_CONFIG._config["mod"] = "infer"
optimize_and_load_gguf(model, OPTIMIZE_RULE, MODEL_PATH, config)

print("Model loaded. Testing inference...")

# Build test prompt using chat template
messages = [{"role": "user", "content": "What is the capital of France? Please answer briefly."}]
prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

device = next(model.parameters()).device
input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
print(f"Input shape: {input_tensor.shape}, device: {device}")
print(f"Prompt: {tokenizer.decode(prompt_ids)}")

print("\nDirect forward pass test...")
# Test direct model forward pass
with torch.no_grad():
    from transformers import StaticCache
    device = torch.device("cuda:0")

    # Get embeddings
    inputs_on_cpu = input_tensor.to("cpu")
    inputs_embeds = model.model.embed_tokens(inputs_on_cpu)
    print(f"inputs_embeds: shape={inputs_embeds.shape}, device={inputs_embeds.device}")
    print(f"  sample: {inputs_embeds[0, 0, :5].tolist()}")

    # Move to CUDA
    inputs_embeds = inputs_embeds.to(device)
    cache_position = torch.arange(input_tensor.shape[1], device=device, dtype=torch.long)

    # Initialize cache
    past_key_values = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=input_tensor.shape[1] + 100,
        device=device, dtype=model.dtype
    )

    # Forward pass
    output = model(
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=True,
        use_cache=True
    )

    logits = output.logits
    print(f"logits: shape={logits.shape}")

    # Check last token logits
    last_logits = logits[0, -1, :]
    print(f"last_logits stats: min={last_logits.min():.3f}, max={last_logits.max():.3f}, mean={last_logits.float().mean():.6f}, std={last_logits.float().std():.6f}")

    # Get top 10 tokens
    top_values, top_indices = torch.topk(last_logits, 10)
    print("Top 10 predicted tokens:")
    for v, i in zip(top_values, top_indices):
        tok = tokenizer.decode([i.item()])
        print(f"  {i.item()}: '{tok}' (logit={v.item():.3f})")

    # Also check what sampling would give
    probs = torch.softmax(last_logits.float(), dim=-1)
    print(f"Max probability: {probs.max():.6f}")

print("\n--- Full generation test ---")
output = prefill_and_generate(
    model,
    tokenizer,
    input_tensor,
    max_new_tokens=50,
    use_cuda_graph=False,
    mode="default",
    force_think=False,
    chunk_size=8192,
)

print(f"\nOutput: {output}")
