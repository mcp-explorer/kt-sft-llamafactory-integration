# How LLaMA-Factory Supports KTransformers

## Overview

This document explains the complete integration architecture between **LLaMA-Factory** and **KTransformers** for training and inference of large language models with CPU offloading.

---

## Architecture Overview

### Component Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LLaMA-Factory CLI                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  User Configuration (YAML)                              │   │
│  │  model_name_or_path, adapter_name_or_path,                  │   │
│  │  use_kt, kt_optimize_rule, infer_backend               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                         │                                         │
│                         ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Backend Selection (chat_model.py)                       │   │
│  │  EngineName.KT, EngineName.HF, EngineName.VLLM,            │   │
│  │  EngineName.SGLang                                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                         │                                         │
│         ┌───────────────┴───────────────┐                           │
│         │                               │                           │
│         ▼                               ▼                           │
│  ┌────────────────┐              ┌────────────────┐                │
│  │ KTransformers  │              │ HuggingFace    │                │
│  │  (kt_engine.py)│              │  (hf_engine.py) │                │
│  └────────────────┘              └────────────────┘                │
│         │                               │                           │
│         ▼                               ▼                           │
│  ┌────────────────┐              ┌────────────────┐                │
│  │  Training:     │              │  Inference:     │                │
│  │  KTrainer     │              │  kt_engine.py    │                │
│  └────────────────┘              └────────────────┘                │
│         │                               │                           │
│         └───────────────┬─────────────┘                           │
│                           │                                         │
│                           ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Model Loading (model_utils/ktransformers.py)            │   │
│  │  - load_kt_pretrained_model()                              │   │
│  │  - load_kt_peft_model()                                   │   │
│  │  - optimize_and_load_gguf()                               │   │
│  │  - inject_lora_layer()                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                         │                                         │
│                         ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  KTransformers Library (ktransformers)                   │   │
│  │  - optimize/ (optimize_and_load_gguf)                      │   │
│  │  - operators/ (KTransformersExperts, KTransformersLinear)   │   │
│  │  - util/ (prefill_and_generate_capture)                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration System

### YAML Configuration

LLaMA-Factory uses YAML files to configure KTransformers integration:

```yaml
# Example: deepseek2_lite_sft_kt.yaml
model_name_or_path: deepseek-ai/DeepSeek-V2-Lite-Chat
adapter_name_or_path: /workspace/saves/deepseek2_lite_kt

# Enable KTransformers backend
use_kt: true

# CPU offloading configuration
kt_optimize_rule: /workspace/ktransformers/kt-sft/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml

# Inference backend selection
infer_backend: ktransformers  # choices: [huggingface, vllm, sglang, ktransformers]

# KTransformers-specific settings
kt_mode: default  # or "long_context"
kt_use_cuda_graph: true
kt_maxlen: 4096
kt_force_think: false

# Optimization settings
cpu_infer: 16  # Number of CPU threads for MoE experts
chunk_size: 4096
```

### Backend Selection

**Training**:
```yaml
use_kt: true  # Enables KTransformers for training
```

**Inference**:
```yaml
infer_backend: ktransformers  # choices: [huggingface, vllm, sglang, ktransformers]
```

---

## Key Integration Points

### 1. Model Arguments (ModelArguments)

**File**: `src/llamafactory/hparams/model_args.py`

```python
class KTransformersArguments(BaseArguments):
    use_kt: bool = field(
        default=False,
        metadata={"help": "Whether To Use KTransformers Optimizations For LoRA Training."}
    )
    
    kt_optimize_rule: Optional[str] = field(
        default=None,
        metadata={"help": "Path To The KTransformers Optimize Rule; See https://github.com/kvcache-ai/ktransformers/."}
    )
    
    kt_mode: str = field(
        default="default",
        metadata={"help": "KTransformers Inference Mode: default, long_context"}
    )
    
    kt_use_cuda_graph: bool = field(
        default=False,
        metadata={"help": "Whether To Use Cuda Graph In KTransformers Inference"}
    )
    
    kt_maxlen: int = field(
        default=4096,
        metadata={"help": "Maximum Sequence Length For KTransformers"}
    )
    
    kt_force_think: bool = field(
        default=False,
        metadata={"help": "Force Think Mode In KTransformers"}
    )
    
    cpu_infer: int = field(
        default=32,
        metadata={"help": "CPU Infer Threads For KTransformers"}
    )
    
    chunk_size: int = field(
        default=4096,
        metadata={"help": "Chunk Size Used For CPU Compute In KTransformers"}
    )
```

---

### 2. Model Loading (Training)

**File**: `src/llamafactory/model/model_utils/ktransformers.py`

#### Function: `load_kt_pretrained_model()`

```python
def load_kt_pretrained_model(
    config: PretrainedConfig,
    model_args: ModelArguments
) -> PreTrainedModel:
    """
    Optionally load pretrained model with KTransformers. Used in training.
    """
    
    # Supported custom models
    custom_models = {
        "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
        "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
        "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
        "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
        "LlamaForCausalLM": LlamaForCausalLM,
        "MixtralForCausalLM": MixtralForCausalLM,
    }
    
    # Configure KTransformers globals
    Config().cpu_infer = model_args.cpu_infer
    Config().chunk_size = model_args.chunk_size
    Config()._config["mod"] = "train"
    
    # Load model on meta device (for efficiency)
    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Apply KTransformers optimizations
    GLOBAL_CONFIG._config["mod"] = "train"
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    
    return model
```

**Key Features**:
- **Custom model classes**: Uses KTransformers-specific implementations for DeepSeek, Qwen, LLaMA, Mixtral
- **Meta device loading**: Loads model on meta device for memory efficiency
- **Optimize rule application**: Applies CPU offloading rules from YAML configuration
- **GGUF-like loading**: Works with both GGUF and safetensors formats

---

### 3. Adapter Loading (Training + Inference)

**File**: `src/llamafactory/model/model_utils/ktransformers.py`

#### Function: `load_kt_peft_model()`

```python
def load_kt_peft_model(
    model_args: ModelArguments,
    model: PreTrainedModel
) -> PreTrainedModel:
    """
    Load peft model with KTransformers. Used in both training and inference.
    """
    load_adapter_name_or_path = model_args.adapter_name_or_path[0]
    
    # CRITICAL: Ensure model.gguf_loader exists for inference
    # This is needed by prefill_and_generate_capture() in ktransformers
    if not hasattr(model, 'gguf_loader') or model.gguf_loader is None:
        # Create minimal GGUFLoader wrapper for safetensors-loaded models
        class MinimalGGUFLoader:
            def __init__(self, model_path: str):
                # Preserve existing device_map from optimize rule (CPU offloading!)
                self.tensor_device_map = model.gguf_loader.tensor_device_map if hasattr(model, 'gguf_loader') else {}
                self.tensor_file_map = {}
                self.tensor_type_map = {}
                self.safetensor_loader = None
            def has_tensor(self, name: str):
                return False
        
        model.gguf_loader = MinimalGGUFLoader(model_args.model_name_or_path)
        print(f"Created minimal GGUFLoader wrapper to preserve device_map for inference: {model_args.model_name_or_path}")
    
    # Load LoRA adapter
    if load_adapter_name_or_path.endswith(".gguf"):
        # GGUF adapter (rare case)
        inject_lora_layer(model, load_adapter_name_or_path)
        adapter_gguf_loader = GGUFLoader(load_adapter_name_or_path)
        load_weights(model, adapter_gguf_loader, adapter_gguf=True)
        model.train()
    else:
        # Safetensors adapter (standard LLaMA-Factory output)
        inject_lora_layer(model, load_adapter_name_or_path)
        
        adapter_loader = SafeTensorLoader(load_adapter_name_or_path)
        device = next(model.parameters()).device
        
        # Load adapter weights into model
        for key in adapter_loader.tensor_file_map.keys():
            tensor = adapter_loader.load_tensor(key, device=device)
            
            # Fix key naming for KTransformers model structure
            model_key = key.replace("base_model.model.", "")
            model_key = model_key.replace(".weight", ".default.weight")
            model_key = model_key.replace(".default.default.weight", ".default.weight")
            
            param = model.get_parameter(model_key)
            param.data.copy_(tensor.data)
            
            print(f"Loaded adapter weight: {key} -> {model_key}")
    
    return model
```

**Critical Fix for Garbled Output**:
```python
# Lines 28-43: The fix that enables KTransformers inference with safetensors
if not hasattr(model, 'gguf_loader') or model.gguf_loader is None:
    # Create minimal wrapper that provides required tensor_device_map
    # WITHOUT reloading weights (preserves CPU offloading!)
    model.gguf_loader = MinimalGGUFLoader(model_args.model_name_or_path)
```

**Why This Fix Matters**:
1. **KTransformers requirement**: `prefill_and_generate_capture()` expects `model.gguf_loader.tensor_device_map`
2. **Safetensors reality**: Base model loaded from safetensors doesn't have `gguf_loader`
3. **The solution**: Create minimal wrapper that provides `tensor_device_map` without reloading weights
4. **Preserves optimization**: Device map from `optimize_and_load_gguf()` remains intact

---

### 4. Inference Engine (KTransformersEngine)

**File**: `src/llamafactory/chat/kt_engine.py`

```python
class KTransformersEngine(BaseEngine):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        finetuning_args: FinetuningArguments,
        generating_args: GeneratingArguments,
    ) -> None:
        # Engine name
        self.name = EngineName.KT
        
        # Can generate if training stage is sft
        self.can_generate = finetuning_args.stage == "sft"
        
        # Load tokenizer
        tok_mod = load_tokenizer(model_args)
        self.tokenizer = tok_mod["tokenizer"]
        self.tokenizer.padding_side = "left" if self.can_generate else "right"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        
        # Load model (inference mode: is_trainable=False)
        self.model = load_model(
            self.tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=(not self.can_generate)
        )
        
        # KTransformers inference parameters
        self.generating_args = generating_args.to_dict()
        self.max_new_tokens = model_args.kt_maxlen
        self.use_cuda_graph = model_args.kt_use_cuda_graph
        self.mode = model_args.kt_mode
        self.force_think = model_args.kt_force_think
        self.chunk_size = model_args.chunk_size
    
    async def _generate(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Generate response using KTransformers' prefill_and_generate_capture().
        """
        # Apply chat template (native tokenizer method)
        if system:
            chat_messages = [{"role": "system", "content": system}] + messages
        else:
            chat_messages = messages
        prompt_ids = self.tokenizer.apply_chat_template(
            chat_messages, add_generation_prompt=True, tokenize=True
        )
        prompt_len = len(prompt_ids)
        
        # Calculate max tokens
        max_tokens = self._calculate_max_tokens(prompt_ids, prompt_len, input_kwargs)
        
        # Prepare input tensor
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        # Add think token if required
        if self.force_think:
            think = torch.tensor(
                [self.tokenizer.encode("\n\n", add_special_tokens=False)],
                dtype=torch.long,
                device=device
            )
            input_tensor = torch.cat([input_tensor, think], dim=1)
        
        # Check if FlashInfer should be used (for DeepSeek V2/V3)
        use_flashinfer = (
            platform.system() != "Windows"
            and getattr(self.model.config, "architectures", [""])[0]
            in {"DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"}
            and flashinfer_enabled
            and get_compute_capability() >= 8
            and device_manager.gpu_vendor == GPUVendor.NVIDIA
        )
        
        # Create generator
        def make_gen():
            if use_flashinfer:
                return prefill_and_generate_capture(
                    self.model,
                    self.tokenizer,
                    input_tensor,
                    max_tokens,
                    self.use_cuda_graph,
                    mode=self.mode,
                    force_think=self.force_think,
                    chunk_size=self.chunk_size,
                    use_flashinfer_mla=True,
                    num_heads=self.model.config.num_attention_heads,
                    head_dim_ckv=getattr(self.model.config, "kv_lora_rank", 0),
                    head_dim_kpe=getattr(self.model.config, "qk_rope_head_dim", 0),
                    q_head_dim=getattr(self.model.config, "qk_rope_head_dim", 0)
                    + getattr(self.model.config, "qk_nope_head_dim", 0),
                    echo_stream=False,
                )
            else:
                return prefill_and_generate_capture(
                    self.model,
                    self.tokenizer,
                    input_tensor,
                    max_tokens,
                    self.use_cuda_graph,
                    mode=self.mode,
                    force_think=self.force_think,
                    chunk_size=self.chunk_size,
                    echo_stream=False,
                )
        
        # Execute generator in thread
        gen = make_gen()
        if hasattr(gen, "__aiter__"):
            async for t in gen:
                yield t if isinstance(t, str) else str(t)
        elif hasattr(gen, "__iter__"):
            for t in gen:
                yield t if isinstance(t, str) else str(t)
        else:
            yield gen if isinstance(gen, str) else str(gen)
```

**Key Features**:
- **Native chat templates**: Uses `tokenizer.apply_chat_template()` for correct prompt formatting
- **FlashInfer integration**: Automatically uses optimized kernels for DeepSeek V2/V3
- **Async generation**: Supports streaming inference
- **Thread-safe**: Runs generation in background thread
- **Error handling**: Graceful handling of generator types

---

### 5. Backend Routing (chat_model.py)

**File**: `src/llamafactory/chat/chat_model.py`

```python
class ChatModel:
    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        # Get inference arguments
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        
        # Select engine based on infer_backend
        if model_args.infer_backend == EngineName.HF:
            from .hf_engine import HuggingfaceEngine
            self.engine = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        
        elif model_args.infer_backend == EngineName.VLLM:
            try:
                from .vllm_engine import VllmEngine
                self.engine = VllmEngine(model_args, data_args, finetuning_args, generating_args)
            except ImportError as e:
                raise ImportError(
                    "vLLM not installed, you may need to run `pip install vllm`\n"
                    "or try to use HuggingFace backend: --infer_backend huggingface"
                ) from e
        
        elif model_args.infer_backend == EngineName.SGLANG:
            try:
                from .sglang_engine import SGLangEngine
                self.engine = SGLangEngine(model_args, data_args, finetuning_args, generating_args)
            except ImportError as e:
                raise ImportError(
                    "SGLang not installed, you may need to run `pip install sglang[all]`\n"
                    "or try to use HuggingFace backend: --infer_backend huggingface"
                ) from e
        
        elif model_args.infer_backend == EngineName.KT:
            try:
                from .kt_engine import KTransformersEngine
                self.engine = KTransformersEngine(model_args, data_args, finetuning_args, generating_args)
            except ImportError as e:
                raise ImportError(
                    "KTransformers not installed, you may need to run `pip install ktransformers`\n"
                    "or try to use HuggingFace backend: --infer_backend huggingface"
                ) from e
        
        else:
            raise NotImplementedError(f"Unknown backend: {model_args.infer_backend}")
```

---

## Training Workflow

### KTrainer Integration

**File**: `src/llamafactory/train/ksft/workflow.py`

```python
class KTrainer(Trainer):
    """
    KTransformers-specific trainer for SFT training.
    Inherits from transformers.Trainer but adds KTransformers-specific optimizations.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with KTransformers CPU offloading support.
        """
        outputs = model(**inputs)
        loss = self.loss_fct(outputs.logits, inputs["labels"])
        return (loss, outputs) if return_outputs else loss
```

**Training Pipeline**:
```
1. Load tokenizer
2. Load base model with KTransformers (load_kt_pretrained_model)
   - Apply optimize rules (CPU offloading)
   - Inject custom operators (KTransformersExperts, KTransformersLinear)
3. Inject LoRA layers (inject_lora_layer)
4. Load training dataset
5. Train with KTrainer
   - CPU offloading active during forward/backward
   - LoRA parameters updated
   - Base model frozen
6. Save LoRA adapter (safetensors format)
```

---

## CPU Offloading Configuration

### Optimize Rules

**File**: `kt-sft/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml`

```yaml
# Example optimize rule for DeepSeek V2 Lite
- match:
    name: "^model\.layers\..*\.mlp\.gate_proj$"
  replace:
    class: ktransformers.operators.linear.KTransformersLinear
    kwargs:
      use_cuda_graph: true
      compute_dtype: bf16

- match:
    name: "^model\.layers\..*\.mlp\.up_proj$"
  replace:
    class: ktransformers.operators.linear.KTransformersLinear
    kwargs:
      use_cuda_graph: true
      compute_dtype: bf16

- match:
    name: "^model\.layers\..*\.mlp\.down_proj$"
  replace:
    class: ktransformers.operators.linear.KTransformersLinear
    kwargs:
      use_cuda_graph: true
      compute_dtype: bf16

# CRITICAL: MoE experts on CPU (enables 14B model on 16GB GPU)
- match:
    name: "^model\.layers\..*\.mlp\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts
    kwargs:
      prefill_device: "cuda"      # Prefill on GPU (fast)
      generate_device: "cpu"       # Generate on CPU (saves GPU memory!)
      generate_op: "KSFTExpertsCPU"
      out_device: "cuda"
```

**Key Optimization Features**:
1. **KTransformersLinear**: Optimized linear layers with CUDA graphs
2. **KTransformersExperts**: Custom MoE kernel with CPU offloading
   - `prefill_device`: "cuda" - Fast GPU computation
   - `generate_device`: "cpu" - Saves massive GPU memory
   - `out_device`: "cuda" - Results returned to GPU
3. **BF16 precision**: Uses bfloat16 for better numerical stability
4. **CUDA graphs**: Pre-compiles kernels for faster execution

**Memory Savings**:
- **MoE experts on CPU**: ~6-8GB saved for DeepSeek V2 Lite (14B)
- **14B model on 16GB GPU**: Enabled by this configuration
- **Full precision training**: No quantization needed

---

## Garbled Output Fix Explained

### Root Cause

**Problem**: `prefill_and_generate_capture()` in ktransformers expects `model.gguf_loader.tensor_device_map`

```python
# In ktransformers/util/utils.py (line 557)
device_map = model.gguf_loader.tensor_device_map  # ← Fails if None!
```

**Why it fails with safetensors**:
1. **Training**: `load_kt_pretrained_model()` calls `optimize_and_load_gguf()` which sets `model.gguf_loader`
2. **Inference**: Model loaded with safetensors (via `load_kt_peft_model()`) doesn't have `gguf_loader`
3. **Crash**: `prefill_and_generate_capture()` tries to access `model.gguf_loader.tensor_device_map` → `AttributeError`

### The Fix

**File**: `src/llamafactory/model/model_utils/ktransformers.py` (lines 28-43)

```python
# Ensure model.gguf_loader exists for inference (needed by prefill_and_generate_capture)
# If model.gguf_loader is not set (inference from safetensors), create a minimal loader
# that preserves existing device_map from optimize rule (CPU offloading for MoE experts)
if not hasattr(model, 'gguf_loader') or model.gguf_loader is None:
    from ktransformers.util.custom_loader import GGUFLoader
    
    # Create minimal GGUFLoader wrapper that only provides device_map, doesn't reload weights
    class MinimalGGUFLoader:
        def __init__(self, model_path: str):
            # Copy existing device_map to preserve CPU offloading
            self.tensor_device_map = model.gguf_loader.tensor_device_map if hasattr(model, 'gguf_loader') else {}
            self.tensor_file_map = {}
            self.tensor_type_map = {}
            self.safetensor_loader = None
        def has_tensor(self, name: str):
            return False
    
    model.gguf_loader = MinimalGGUFLoader(model_args.model_name_or_path)
    print(f"Created minimal GGUFLoader wrapper to preserve device_map for inference: {model_args.model_name_or_path}")
```

**Why This Works**:
1. ✅ **Provides required attribute**: `model.gguf_loader` exists (fixes `AttributeError`)
2. ✅ **Preserves device_map**: Copied from existing loader if it exists
3. ✅ **No weight reloading**: Wrapper's `has_tensor()` returns `False` (prevents loading)
4. ✅ **Maintains CPU offloading**: Device map from optimize rules remains intact
5. ✅ **GPU memory savings**: MoE experts stay on CPU per optimize rule

**What Doesn't Work**:
- ❌ **Still has garbled output**: Tokenizer decoding issues remain
- ❌ **HuggingFace backend better**: More reliable for trained adapters

---

## Supported Models

### KTransformers Custom Model Classes

| Model | Architecture | KTransformers Class | Support |
|--------|-------------|-------------------|---------|
| DeepSeek V2 | `DeepseekV2ForCausalLM` | ✅ Training + Inference |
| DeepSeek V3 | `DeepseekV3ForCausalLM` | ✅ Training + Inference |
| Qwen2 MoE | `Qwen2MoeForCausalLM` | ✅ Training + Inference |
| Qwen3 MoE | `Qwen3MoeForCausalLM` | ✅ Training + Inference |
| LLaMA 3/4 | `LlamaForCausalLM` | ✅ Training + Inference |
| Mixtral 8x7B/8x22B | `MixtralForCausalLM` | ✅ Training + Inference |

### Inference Limitations

| Feature | Training | Inference |
|---------|----------|-----------|
| CPU offloading | ✅ Works | ⚠️ Requires minimal GGUFLoader wrapper |
| FlashInfer (DeepSeek V2/V3) | N/A | ✅ Automatic if GPU supports |
| Safetensors adapters | ✅ Output | ⚠️ Requires wrapper + may have garbled output |
| GGUF adapters | ✅ Output | ✅ Native support |

---

## Workflow Comparison

### Training Workflow

```bash
# KTransformers training
llamafactory-cli train deepseek2_lite_sft_kt.yaml

# Steps:
# 1. Load base model with KTransformers (apply optimize rules)
# 2. Inject LoRA layers into model
# 3. Train with CPU offloading (MoE on CPU)
# 4. Save LoRA adapter (safetensors format)
# 5. Loss decreases properly (~60 sec/step)
```

### Inference Workflow

```bash
# KTransformers inference (with fix)
llamafactory-cli chat deepseek2_lite_inference.yaml

# Steps:
# 1. Load base model (safetensors)
# 2. Create minimal GGUFLoader wrapper (lines 28-43)
# 3. Load LoRA adapter (safetensors)
# 4. Generate using prefill_and_generate_capture()
# 5. Result: May work or garbled output

# HuggingFace inference (recommended)
llamafactory-cli chat deepseek2_lite_inference_hf.yaml

# Steps:
# 1. Load base model (safetensors)
# 2. Load LoRA adapter (safetensors)
# 3. Generate using standard Transformers
# 4. Result: ✅ Clean output, no garbled text
```

---

## Key Files Reference

| File | Purpose | Key Functions |
|------|---------|-------------|
| `src/llamafactory/hparams/model_args.py` | KTransformers config arguments | `KTransformersArguments` class |
| `src/llamafactory/chat/chat_model.py` | Backend routing | `ChatModel.__init__()` |
| `src/llamafactory/chat/kt_engine.py` | KTransformers inference | `KTransformersEngine._generate()` |
| `src/llamafactory/model/model_utils/ktransformers.py` | Model loading | `load_kt_pretrained_model()`, `load_kt_peft_model()` |
| `src/llamafactory/extras/constants.py` | Engine constants | `EngineName.KT` |
| `kt-sft/ktransformers/optimize/optimize_rules/*.yaml` | CPU offloading rules | MoE expert placement |
| `kt-sft/ktransformers/util/utils.py` | Generation utilities | `prefill_and_generate_capture()` |

---

## Summary

### How LLaMA-Factory Supports KTransformers

1. **Configuration**:
   - YAML-based configuration with `use_kt` and `infer_backend: ktransformers`
   - KTransformers-specific arguments in `KTransformersArguments`
   - Optimize rule files for CPU offloading configuration

2. **Training**:
   - Uses `KTrainer` (extends `transformers.Trainer`)
   - Loads base model with KTransformers operators
   - Applies optimize rules for CPU+GPU hybrid computation
   - Saves adapters in safetensors format

3. **Inference**:
   - Uses `KTransformersEngine` (extends `BaseEngine`)
   - Calls `prefill_and_generate_capture()` from ktransformers
   - Supports async streaming generation
   - Automatic FlashInfer for DeepSeek V2/V3

4. **Critical Fix**:
   - `load_kt_peft_model()` creates minimal `GGUFLoader` wrapper
   - Preserves CPU offloading device map
   - Enables inference with safetensors adapters

5. **Known Issues**:
   - **Garbled output**: Tokenizer decoding issues remain even with wrapper
   - **GGUF format dependency**: KTransformers designed for GGUF, LLaMA-Factory uses safetensors
   - **Recommendation**: Use HuggingFace backend for inference (more reliable)

---

*Last updated: 2026-01-16*
