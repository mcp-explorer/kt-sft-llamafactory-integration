"""Compare intermediate outputs between HuggingFace and ktransformers."""
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

MODEL_PATH = "/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
messages = [{"role": "user", "content": "Hello"}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
print(f"Input ids: {input_ids}")

# === HuggingFace Model ===
print("\n=== HuggingFace Model ===")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

with torch.no_grad():
    # Get embeddings
    hf_embed = hf_model.model.embed_tokens(input_ids)
    print(f"HF embeddings shape: {hf_embed.shape}")
    print(f"HF embeddings sample: {hf_embed[0, 0, :5].tolist()}")

    # Full forward pass
    hf_output = hf_model(input_ids, return_dict=True)
    hf_logits = hf_output.logits
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"HF last token logits sample (first 5): {hf_logits[0, -1, :5].tolist()}")
    print(f"HF last token logits stats: min={hf_logits[0,-1].min():.3f}, max={hf_logits[0,-1].max():.3f}")

    # Top predictions
    top_vals, top_ids = torch.topk(hf_logits[0, -1], 5)
    print("HF top 5 predictions:")
    for v, i in zip(top_vals, top_ids):
        print(f"  {i.item()}: '{tokenizer.decode([i.item()])}' (logit={v.item():.3f})")

# Keep HF model for comparison
# del hf_model
# torch.cuda.empty_cache()

# === ktransformers Model ===
print("\n=== ktransformers Model ===")
from ktransformers.sft.monkey_patch_torch_module import install_patch
install_patch()

from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.server.config.config import Config
from ktransformers.util.globals import GLOBAL_CONFIG

OPTIMIZE_RULE = "/home/sean/miniconda3/envs/Kllama/lib/python3.12/site-packages/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml"

Config().cpu_infer = 16
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
torch.set_default_dtype(config.torch_dtype)

with torch.device("meta"):
    kt_model = DeepseekV2ForCausalLM(config)

GLOBAL_CONFIG._config["mod"] = "infer"
optimize_and_load_gguf(kt_model, OPTIMIZE_RULE, MODEL_PATH, config)

with torch.no_grad():
    # Get embeddings (on CPU for ktransformers)
    kt_embed = kt_model.model.embed_tokens(input_ids.to("cpu"))
    print(f"KT embeddings shape: {kt_embed.shape}, device: {kt_embed.device}")
    print(f"KT embeddings sample: {kt_embed[0, 0, :5].tolist()}")

    # Compare embeddings
    embed_diff = (hf_embed.cpu().float() - kt_embed.float()).abs().max()
    print(f"Embedding max diff: {embed_diff}")

    # For ktransformers, we need to use the full inference pipeline
    # But let's try a simple forward to see what happens
    kt_embed_cuda = kt_embed.to("cuda:0")

    print("\n=== Layer 0 comparison (dense layer) ===")
    layer0_kt = kt_model.model.layers[0]
    layer0_hf = hf_model.model.layers[0]

    # Compare input_layernorm
    kt_hidden = kt_embed_cuda
    hf_hidden = hf_embed

    kt_ln_out = layer0_kt.input_layernorm(kt_hidden)
    hf_ln_out = layer0_hf.input_layernorm(hf_hidden)
    print(f"KT input_layernorm: {kt_ln_out[0, 0, :5].tolist()}")
    print(f"HF input_layernorm: {hf_ln_out[0, 0, :5].tolist()}")
    ln_diff = (kt_ln_out.cpu().float() - hf_ln_out.float()).abs().max()
    print(f"input_layernorm diff: {ln_diff}")

    # Compare q_proj
    kt_q = layer0_kt.self_attn.q_proj(kt_ln_out)
    hf_q = layer0_hf.self_attn.q_proj(hf_ln_out)
    print(f"KT q_proj shape: {kt_q.shape}, sample: {kt_q[0, 0, :5].tolist()}")
    print(f"HF q_proj shape: {hf_q.shape}, sample: {hf_q[0, 0, :5].tolist()}")
    q_diff = (kt_q.cpu().float() - hf_q.float()).abs().max()
    print(f"q_proj diff: {q_diff}")

    # Compare kv_a_proj_with_mqa
    kt_kv_a = layer0_kt.self_attn.kv_a_proj_with_mqa(kt_ln_out)
    hf_kv_a = layer0_hf.self_attn.kv_a_proj_with_mqa(hf_ln_out)
    print(f"KT kv_a_proj shape: {kt_kv_a.shape}, sample: {kt_kv_a[0, 0, :5].tolist()}")
    print(f"HF kv_a_proj shape: {hf_kv_a.shape}, sample: {hf_kv_a[0, 0, :5].tolist()}")
    kv_a_diff = (kt_kv_a.cpu().float() - hf_kv_a.float()).abs().max()
    print(f"kv_a_proj diff: {kv_a_diff}")

    # Full layer 0 forward pass comparison
    print("\n=== Full layer 0 forward comparison ===")
    # For HF
    hf_layer0_out = layer0_hf(hf_embed, position_ids=torch.arange(8).unsqueeze(0))[0]
    print(f"HF layer0 output: {hf_layer0_out[0, 0, :5].tolist()}")

    # For KT - need to pass through attention which requires cache
    # Let's skip the full forward and check final logits comparison
    print("\n=== Final logits comparison ===")
    # KT forward needs proper setup, so let's use the prefill_and_generate path
    # For now, let's see what ktransformers produces with a simple forward

    # Try using ktransformers model directly
    from ktransformers.util.utils import prefill_and_generate
    kt_result = prefill_and_generate(
        kt_model,
        tokenizer,
        input_ids.to("cpu"),
        max_new_tokens=5,
        use_cuda_graph=False,
        mode="default",
        force_think=False,
        chunk_size=8192,
    )
    print(f"KT generated tokens: {kt_result}")

print("\nComparison complete.")
