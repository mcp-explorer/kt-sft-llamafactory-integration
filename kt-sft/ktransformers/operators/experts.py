#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : Azure-Tang, Boxin Zhang, chenht2022
Date         : 2024-07-25 11:25:24
Version      : 0.1.0
LastEditors  : Azure 
LastEditTime : 2024-08-29 09:41:10
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''

from typing import Any, Union
import numpy as np
import numpy.typing as npt
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import sys, os
from ktransformers.operators.base_operator import BaseInjectedModule
from tqdm import tqdm
import time
import logging
from tqdm.auto import tqdm
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug"))
import cpuinfer_ext
from cpuinfer_ext.moe import MOEConfig, MOE
from cpuinfer_ext.sft_moe import SFT_MOEConfig, SFT_MOE
import ctypes
from ktransformers.util.custom_loader import GGUFLoader
from ktransformers.util.inference_state import InferenceState
from ktransformers.util.custom_gguf import GGMLQuantizationType
from ktransformers.util.custom_loader import GGUFLoader, SafeTensorLoader, ModelLoader
from ktransformers.server.config.config import Config
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from abc import ABC, abstractmethod
from ktransformers.operators.linear import KLinearMarlin, KLinearTorch, KTransformersLinear
import time
from ktransformers.operators.cpuinfer import CPUInfer
from ktransformers.util.grad_wrapper import maybe_no_grad

H_FIXED = 7168
M_FIXED = 2048

def deduplicate_and_sort(lst):
    return sorted(set(lst))
def generate_cuda_graphs(chunk_size: int) -> list:
    assert chunk_size <= 1024 or chunk_size % 1024 == 0, "chunk_size must <= 1024 or a multiple of 1024"
    base_list = [1, 2, 3, Config().max_batch_size, 64, 256, 512, chunk_size]

    if chunk_size <= 1024:
        return deduplicate_and_sort(base_list)

    multiples = [i for i in range(1024, chunk_size + 1, 1024)]

    return deduplicate_and_sort(base_list + multiples)
#cuda_graphs = [Config().chunk_size] 
if torch.cuda.is_available():
    cuda_graphs = generate_cuda_graphs(Config().chunk_size)
else:
    cuda_graphs = 1
# class Base(BaseInjectedModule, ABC):
class KExpertsBase(ABC):
    def __init__(self, key: str, gguf_loader: GGUFLoader, config: PretrainedConfig, orig_module: nn.Module, device: str = "cuda", **kwargs):
        # super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.key = key
        self.gguf_loader = gguf_loader
        self.config = config
        self.device = device
    
    @abstractmethod
    def forward(self, input_tensor, expert_ids, weights):
        pass

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str = "cpu", warmup: bool = False):
        pass
    
    @abstractmethod
    def unload():
        pass

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            # if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info: # TODO: maybe problem in merge (this is origin one)
            if self.gguf_loader.has_tensor(key + ".ffn_gate_exps.weight"):
                targets = [".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight" ]
                tensors = self.load_multi(key, targets, device=device)
                gate = tensors[".ffn_gate_exps.weight"]
                up = tensors[".ffn_up_exps.weight"]
                down = tensors[".ffn_down_exps.weight"]
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
            # elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info: # TODO: maybe problem in merge (this is origin one)
            elif self.gguf_loader.has_tensor(key + ".ffn_down.0.weight"):
                # for supporting  Mixtral-8x7B-Instuct  
                gate = []
                up = []
                down = []
                for i in range(8):
                    gatei, upi, downi = f".ffn_gate.{i}.weight", f".ffn_up.{i}.weight", f".ffn_down.{i}.weight"
                    targets = [gatei, upi, downi]
                    tensors = self.load_multi(key, targets, device=device)
                    gate_it, up_it, down_it = tensors[gatei], tensors[upi], tensors[downi]
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = torch.stack(gate)
                up = torch.stack(up)
                down = torch.stack(down)
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate.0.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up.0.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down.0.weight"]["ggml_type"]
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res
    
    def load_multi(self, key: str, keys: list[str], device: str = "cpu"):
        tensors = {}
        for k in keys:
            tensors[k] = self.gguf_loader.load_gguf_tensor(key + k, device=device)
        return tensors
class KExpertsCPU(KExpertsBase):
    input_tensor_cpu:Tensor = None
    expert_ids_cpu:Tensor = None
    weights_cpu:Tensor = None
    output_cpu:Tensor = None
    output_gpu_map:dict = {} # Manage output tensor buffer on different gpu
    #stream_map:dict = {} # Manage cuda stream on different gpu
    # @TODO add yaml
    CPU_INFER = CPUInfer(Config().cpu_infer)
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        out_device: str = "cuda", # this device mean which device the output should on. TODO: support cpu.
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU"
        self.n_routed_experts = n_routed_experts
        self.out_device = out_device
        self.backend = kwargs.get("backend", "llamafile")

    def load(self, w: dict | nn.Parameter | tuple | None = None, device:str|None = None, warmup:bool = False):
        if device:
            assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU, Parameter \"device\" can be cpu or None."
        if w is None: w = self.load_weights()[self.key]
        self.gate = w["gate"]
        self.up = w["up"]
        self.down = w["down"]
        self.gate_type = w["gate_type"]
        self.up_type = w["up_type"]
        self.down_type = w["down_type"]
        
        # Convert numpy arrays to PyTorch tensors if needed
        if isinstance(self.gate, np.ndarray):
            self.gate = torch.from_numpy(self.gate)
        if isinstance(self.up, np.ndarray):
            self.up = torch.from_numpy(self.up)
        if isinstance(self.down, np.ndarray):
            self.down = torch.from_numpy(self.down)
        
        # Ensure tensors are on CPU and contiguous
        if isinstance(self.gate, torch.Tensor):
            if self.gate.device.type != 'cpu':
                self.gate = self.gate.cpu()
            if not self.gate.is_contiguous():
                self.gate = self.gate.contiguous()
        if isinstance(self.up, torch.Tensor):
            if self.up.device.type != 'cpu':
                self.up = self.up.cpu()
            if not self.up.is_contiguous():
                self.up = self.up.contiguous()
        if isinstance(self.down, torch.Tensor):
            if self.down.device.type != 'cpu':
                self.down = self.down.cpu()
            if not self.down.is_contiguous():
                self.down = self.down.contiguous()
        
        # Get pointers - handle both PyTorch tensors and numpy arrays
        if isinstance(self.gate, torch.Tensor):
            gate_ptr = self.gate.data_ptr()
        else:
            gate_ptr = ctypes.addressof(
                ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
        
        if isinstance(self.up, torch.Tensor):
            up_ptr = self.up.data_ptr()
        else:
            up_ptr = ctypes.addressof(
                ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
        
        if isinstance(self.down, torch.Tensor):
            down_ptr = self.down.data_ptr()
        else:
            down_ptr = ctypes.addressof(
                ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
        
        # print(self.gate_qtype, self.up_qtype, self.down_qtype)
        n_routed_experts = self.n_routed_experts
        self.cpu_infer = KExpertsCPU.CPU_INFER
        # n_routed_experts = len(self.orig_module)
        model_dtype = torch.get_default_dtype()
        if torch.xpu.is_available() and model_dtype == torch.float16:
            hidden_type = 1 # fp16
        else:
            hidden_type = 30 # bf16
        if self.backend == "llamafile":
            moe_config = MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                64,
                10,
                1024,
                gate_ptr,
                up_ptr,
                down_ptr,
                self.gate_type,
                self.up_type,
                self.down_type,
                hidden_type, # TODO: get from model.dtype
            )
            self.moe = MOE(moe_config)
        elif self.backend == "AMXBF16":
            from cpuinfer_ext.moe import AMX_MOEConfig, AMXBF16_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = AMX_MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                max(cuda_graphs) if isinstance(cuda_graphs, list) else Config().chunk_size,
                gate_ptr,
                up_ptr,
                down_ptr,
            )
            self.moe = AMXBF16_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        elif self.backend == "AMXInt8":
            from cpuinfer_ext.moe import AMX_MOEConfig, AMXInt8_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = AMX_MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                max(cuda_graphs) if isinstance(cuda_graphs, list) else Config().chunk_size,
                gate_ptr,
                up_ptr,
                down_ptr,
            )
            self.moe = AMXInt8_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        # print(n_routed_experts, hidden_size, moe_intermediate_size)
        num_experts_per_tok = self.config.num_experts_per_tok
        if warmup:
            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()
        if self.out_device not in KExpertsCPU.output_gpu_map:
            if isinstance(cuda_graphs, list):
                KExpertsCPU.output_gpu_map[self.out_device] = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device=self.out_device) for i in range(len(cuda_graphs))]
            else:
                KExpertsCPU.output_gpu_map[self.out_device] = torch.zeros((cuda_graphs, self.config.hidden_size), device=self.out_device)
        if KExpertsCPU.input_tensor_cpu == None:
            if isinstance(cuda_graphs, list):
                KExpertsCPU.input_tensor_cpu = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device="cpu", pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsCPU.expert_ids_cpu = [torch.zeros((cuda_graphs[i], num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsCPU.weights_cpu = [torch.zeros((cuda_graphs[i], num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True) for i in range(len(cuda_graphs))]
                KExpertsCPU.output_cpu = [torch.zeros((cuda_graphs[i], self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16) for i in range(len(cuda_graphs))]
                KExpertsCPU.bsz_tensor_cpu = [torch.zeros((1), device="cpu", dtype=torch.int32, pin_memory=True) for i in range(len(cuda_graphs))]
            else:
                KExpertsCPU.input_tensor_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device="cpu", pin_memory=True)
                KExpertsCPU.expert_ids_cpu = torch.zeros((cuda_graphs, num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True)
                KExpertsCPU.weights_cpu = torch.zeros((cuda_graphs, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True)
                if torch.xpu.is_available():
                    KExpertsCPU.output_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device="cpu", pin_memory=True, dtype=model_dtype)
                    KExpertsCPU.bsz_tensor_cpu = torch.ones((1), device="cpu", dtype=torch.int32, pin_memory=True)
                else:
                    KExpertsCPU.output_cpu = torch.zeros((cuda_graphs, self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
                    KExpertsCPU.bsz_tensor_cpu = torch.zeros((1), device="cpu", dtype=torch.int32, pin_memory=True)
            
    def submit_for_one_decode(self, input_tensor, expert_ids, weights, bsz_tensor=None, cuda_graph_idx=0):
        if bsz_tensor is None:
            bsz_tensor = torch.ones(1, device=input_tensor.device, dtype=torch.int32)
        # Convert string device to device object for current_stream
        device_obj = torch.device(self.out_device) if isinstance(self.out_device, str) else self.out_device
        stream = torch.cuda.current_stream(device_obj) if device_obj.type == 'cuda' else torch.cuda.current_stream()
        if cuda_graph_idx != -1:
            KExpertsCPU.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)
            KExpertsCPU.expert_ids_cpu[cuda_graph_idx].copy_(expert_ids, non_blocking=True)
            KExpertsCPU.weights_cpu[cuda_graph_idx].copy_(weights, non_blocking=True)
            KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(stream.cuda_stream, self.moe.forward(1, expert_ids.size(-1), KExpertsCPU.expert_ids_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.weights_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.input_tensor_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.output_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))
        else:
            KExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
            KExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
            KExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
            KExpertsCPU.bsz_tensor_cpu.copy_(bsz_tensor, non_blocking=True)
            self.cpu_infer.submit_with_cuda_stream(stream.cuda_stream, self.moe.forward(1, expert_ids.size(-1), KExpertsCPU.expert_ids_cpu.data_ptr(), KExpertsCPU.weights_cpu.data_ptr(), KExpertsCPU.input_tensor_cpu.data_ptr(), KExpertsCPU.output_cpu.data_ptr(), KExpertsCPU.bsz_tensor_cpu.data_ptr()))
        

    def sync_for_one_decode(self, cuda_graph_idx=0):
        # Convert string device to device object for current_stream
        device_obj = torch.device(self.out_device) if isinstance(self.out_device, str) else self.out_device
        stream = torch.cuda.current_stream(device_obj) if device_obj.type == 'cuda' else torch.cuda.current_stream()
        if cuda_graph_idx != -1:
            self.cpu_infer.sync_with_cuda_stream(stream.cuda_stream)
            KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx].copy_(KExpertsCPU.output_cpu[cuda_graph_idx], non_blocking=True)
            return KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx]
        else:
            self.cpu_infer.sync_with_cuda_stream(stream.cuda_stream)
            KExpertsCPU.output_gpu_map[self.out_device].copy_(KExpertsCPU.output_cpu, non_blocking=True)
            return KExpertsCPU.output_gpu_map[self.out_device]

    def forward(self, input_tensor, expert_ids, weights, bsz_tensor=None, cuda_graph_idx=0):
        # generate, capture and run cuda graph
        # print(expert_ids)
        if bsz_tensor is None and (not torch.xpu.is_available() or input_tensor.size(0) > 1):
            bsz_tensor = torch.tensor([input_tensor.size(0)], device=input_tensor.device, dtype=torch.int32)
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            if cuda_graph_idx != -1:
                KExpertsCPU.input_tensor_cpu[cuda_graph_idx].copy_(input_tensor, non_blocking=True)
                KExpertsCPU.expert_ids_cpu[cuda_graph_idx].copy_(expert_ids, non_blocking=True)
                KExpertsCPU.weights_cpu[cuda_graph_idx].copy_(weights, non_blocking=True)
                KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].copy_(bsz_tensor, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, self.moe.forward(expert_ids.size(0), expert_ids.size(-1), KExpertsCPU.expert_ids_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.weights_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.input_tensor_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.output_cpu[cuda_graph_idx].data_ptr(), KExpertsCPU.bsz_tensor_cpu[cuda_graph_idx].data_ptr()))
                self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
                KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx].copy_(KExpertsCPU.output_cpu[cuda_graph_idx], non_blocking=True)
                return KExpertsCPU.output_gpu_map[self.out_device][cuda_graph_idx]

            else:
                KExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
                KExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
                KExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
                KExpertsCPU.bsz_tensor_cpu.copy_(bsz_tensor, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, self.moe.forward(expert_ids.size(0), expert_ids.size(-1), KExpertsCPU.expert_ids_cpu.data_ptr(), KExpertsCPU.weights_cpu.data_ptr(), KExpertsCPU.input_tensor_cpu.data_ptr(), KExpertsCPU.output_cpu.data_ptr(), KExpertsCPU.bsz_tensor_cpu.data_ptr()))
                self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
                KExpertsCPU.output_gpu_map[self.out_device].copy_(KExpertsCPU.output_cpu, non_blocking=True)
                return KExpertsCPU.output_gpu_map[self.out_device]
        elif input_tensor.size(0)==1 and torch.xpu.is_available():
            KExpertsCPU.input_tensor_cpu.copy_(input_tensor.view(-1), non_blocking=True)
            KExpertsCPU.expert_ids_cpu.copy_(expert_ids.view(-1), non_blocking=True)
            KExpertsCPU.weights_cpu.copy_(weights.view(-1), non_blocking=True)
            # KExpertsCPU.bsz_tensor_cpu.copy_(bsz_tensor.view(-1), non_blocking=True)
            self.cpu_infer.submit(self.moe.forward(expert_ids.size(0), expert_ids.size(1), KExpertsCPU.expert_ids_cpu.data_ptr(), KExpertsCPU.weights_cpu.data_ptr(), KExpertsCPU.input_tensor_cpu.data_ptr(), KExpertsCPU.output_cpu.data_ptr(), KExpertsCPU.bsz_tensor_cpu.data_ptr()))
            self.cpu_infer.sync()
            KExpertsCPU.output_gpu_map[self.out_device].copy_(KExpertsCPU.output_cpu, non_blocking=True)
            return KExpertsCPU.output_gpu_map[self.out_device].view(1, -1)
        else:
            input_tensor = input_tensor.contiguous().cpu()
            expert_ids = expert_ids.contiguous().cpu()
            weights = weights.contiguous().to(torch.float32).cpu()
            bsz_tensor = bsz_tensor.contiguous().cpu()
            output = torch.empty_like(input_tensor).contiguous()
            self.cpu_infer.submit(self.moe.forward(expert_ids.size(0), expert_ids.size(1), expert_ids.data_ptr(), weights.data_ptr(), input_tensor.data_ptr(), output.data_ptr(), bsz_tensor.data_ptr()))
            self.cpu_infer.sync()
            return output.to(device=object.__getattribute__(self, "out_device"))
    
    def unload(self):
        return

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        # TODO: support Bias
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if isinstance(self.gguf_loader, SafeTensorLoader):
                res = self.gguf_loader.load_experts(key)
                return {key: res}
            elif self.gguf_loader.has_tensor(key + ".ffn_gate_exps.weight"):
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                # gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                # up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                # down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
                gate_type = self.gguf_loader.get_ggml_type(key + ".ffn_gate_exps.weight")
                up_type = self.gguf_loader.get_ggml_type(key + ".ffn_up_exps.weight")
                down_type = self.gguf_loader.get_ggml_type(key + ".ffn_down_exps.weight")
            
            elif key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
            elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info:
                # for supporting  Mixtral-8x7B-Instuct  
                gate = []
                up = []
                down = []
                for i in range(8):
                    gate_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_gate.{i}.weight")
                    up_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_up.{i}.weight")
                    down_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_down.{i}.weight")
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = np.stack(gate)
                up = np.stack(up)
                down = np.stack(down)
                gate_type = self.gguf_loader.get_ggml_type(key + ".ffn_gate.0.weight")
                up_type = self.gguf_loader.get_ggml_type(key + ".ffn_up.0.weight")
                down_type = self.gguf_loader.get_ggml_type(key + ".ffn_down.0.weight")
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res
class KSFTExpertsCPU(torch.autograd.Function):
    input_tensor_cpu:Tensor = None
    expert_ids_cpu:Tensor = None
    weights_cpu:Tensor = None
    output_cpu:Tensor = None
    output_gpu_map:dict = {} # Manage output tensor buffer on different gpu
    #stream_map:dict = {} # Manage cuda stream on different gpu
    #gguf_loader:GGUFLoader = None
    CPU_INFER = CPUInfer(Config().cpu_infer)
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        out_device: str = "cuda", # this device mean which device the output should on. TODO: support cpu.
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        #if KExpertsCPU.gguf_loader is None:
        #    KExpertsCPU.gguf_loader = GGUFLoader("/mnt/data/model/DeepseekV3-q4km-gguf")
        self.gguf_loader = gguf_loader
        assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU"
        self.n_routed_experts = n_routed_experts
        self.out_device = out_device
        self.backend = kwargs.get("backend", "llamafile")

        self.key = key
        self.config = config
        self.device = device

        self.call_count = 0
        self.flops_per_call = []
        self.times = []
        
        self.tflops_fwd = []
        self.tflops_bwd = []

    def load(self, w: dict | nn.Parameter | tuple | None = None, device:str|None = None, warmup:bool = False):
        if device:
            assert device.lower() == "cpu", "KSFTExpertsCPU can only be loaded on CPU, Parameter \"device\" can be cpu or None."
        if w is None: w = self.load_weights()[self.key]
        self.gate = w["gate"]
        self.up = w["up"]
        self.down = w["down"]
        self.gate_type = w["gate_type"]
        self.up_type = w["up_type"]
        self.down_type = w["down_type"]
        
        # Convert numpy arrays to PyTorch tensors if needed
        if isinstance(self.gate, np.ndarray):
            self.gate = torch.from_numpy(self.gate)
        if isinstance(self.up, np.ndarray):
            self.up = torch.from_numpy(self.up)
        if isinstance(self.down, np.ndarray):
            self.down = torch.from_numpy(self.down)
        
        # Debug: Check tensor properties before getting pointers
        debug = os.environ.get("KSFT_MOE_DEBUG", "0") == "1"
        if debug:
            print(f"[KSFTExpertsCPU.load] Checking tensor properties...")
            if isinstance(self.gate, torch.Tensor):
                print(f"  gate: device={self.gate.device}, shape={self.gate.shape}, dtype={self.gate.dtype}, is_contiguous={self.gate.is_contiguous()}")
            else:
                print(f"  gate: type={type(self.gate)}, shape={self.gate.shape if hasattr(self.gate, 'shape') else 'N/A'}")
            if isinstance(self.up, torch.Tensor):
                print(f"  up: device={self.up.device}, shape={self.up.shape}, dtype={self.up.dtype}, is_contiguous={self.up.is_contiguous()}")
            else:
                print(f"  up: type={type(self.up)}, shape={self.up.shape if hasattr(self.up, 'shape') else 'N/A'}")
            if isinstance(self.down, torch.Tensor):
                print(f"  down: device={self.down.device}, shape={self.down.shape}, dtype={self.down.dtype}, is_contiguous={self.down.is_contiguous()}")
            else:
                print(f"  down: type={type(self.down)}, shape={self.down.shape if hasattr(self.down, 'shape') else 'N/A'}")
        
        # Ensure tensors are on CPU and contiguous
        # CRITICAL: Must ensure CPU transfer completes and memory is accessible before getting pointers
        if isinstance(self.gate, torch.Tensor):
            original_device = self.gate.device
            if self.gate.device.type != 'cpu':
                if debug:
                    print(f"[KSFTExpertsCPU.load] WARNING: gate tensor is on {self.gate.device}, moving to CPU")
                # Move to CPU - use clone() to ensure it's a new tensor on CPU, not a view
                self.gate = self.gate.cpu().clone().contiguous()
                # Force synchronization to ensure CPU transfer is complete
                if original_device.type == 'cuda':
                    torch.cuda.synchronize()
                elif hasattr(torch, 'xpu') and original_device.type == 'xpu':
                    torch.xpu.synchronize()
            else:
                # Already on CPU, just ensure contiguous
                if not self.gate.is_contiguous():
                    if debug:
                        print(f"[KSFTExpertsCPU.load] WARNING: gate tensor is not contiguous, making contiguous")
                    self.gate = self.gate.contiguous()
            # Verify it's actually on CPU
            if self.gate.device.type != 'cpu':
                raise RuntimeError(f"gate tensor is still on {self.gate.device} after CPU conversion!")
            
        if isinstance(self.up, torch.Tensor):
            original_device = self.up.device
            if self.up.device.type != 'cpu':
                if debug:
                    print(f"[KSFTExpertsCPU.load] WARNING: up tensor is on {self.up.device}, moving to CPU")
                self.up = self.up.cpu().clone().contiguous()
                # Force synchronization
                if original_device.type == 'cuda':
                    torch.cuda.synchronize()
                elif hasattr(torch, 'xpu') and original_device.type == 'xpu':
                    torch.xpu.synchronize()
            else:
                if not self.up.is_contiguous():
                    if debug:
                        print(f"[KSFTExpertsCPU.load] WARNING: up tensor is not contiguous, making contiguous")
                    self.up = self.up.contiguous()
            if self.up.device.type != 'cpu':
                raise RuntimeError(f"up tensor is still on {self.up.device} after CPU conversion!")
            
        if isinstance(self.down, torch.Tensor):
            original_device = self.down.device
            if self.down.device.type != 'cpu':
                if debug:
                    print(f"[KSFTExpertsCPU.load] WARNING: down tensor is on {self.down.device}, moving to CPU")
                self.down = self.down.cpu().clone().contiguous()
                # Force synchronization
                if original_device.type == 'cuda':
                    torch.cuda.synchronize()
                elif hasattr(torch, 'xpu') and original_device.type == 'xpu':
                    torch.xpu.synchronize()
            else:
                if not self.down.is_contiguous():
                    if debug:
                        print(f"[KSFTExpertsCPU.load] WARNING: down tensor is not contiguous, making contiguous")
                    self.down = self.down.contiguous()
            if self.down.device.type != 'cpu':
                raise RuntimeError(f"down tensor is still on {self.down.device} after CPU conversion!")
        
        # Get pointers after ensuring CPU and contiguous
        # CRITICAL: Verify pointers are valid CPU memory addresses before passing to C++
        if isinstance(self.gate, torch.Tensor):
            # Double-check device before getting pointer
            if self.gate.device.type != 'cpu':
                raise RuntimeError(f"Cannot get pointer from non-CPU tensor: gate is on {self.gate.device}")
            gate_ptr = self.gate.data_ptr()
            # Verify pointer is reasonable (not null, not obviously GPU address)
            if gate_ptr == 0:
                raise RuntimeError("gate.data_ptr() returned null pointer!")
            if debug:
                # Try to read first byte to verify it's accessible CPU memory
                try:
                    import ctypes
                    test_ptr = ctypes.cast(gate_ptr, ctypes.POINTER(ctypes.c_uint8))
                    _ = test_ptr[0]  # Try to read
                except Exception as e:
                    print(f"[KSFTExpertsCPU.load] WARNING: Could not verify gate pointer accessibility: {e}")
        else:
            # numpy array
            gate_ptr = ctypes.addressof(
                ctypes.cast(self.gate.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
        
        if isinstance(self.up, torch.Tensor):
            if self.up.device.type != 'cpu':
                raise RuntimeError(f"Cannot get pointer from non-CPU tensor: up is on {self.up.device}")
            up_ptr = self.up.data_ptr()
            if up_ptr == 0:
                raise RuntimeError("up.data_ptr() returned null pointer!")
        else:
            # numpy array
            up_ptr = ctypes.addressof(
                ctypes.cast(self.up.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
        
        if isinstance(self.down, torch.Tensor):
            if self.down.device.type != 'cpu':
                raise RuntimeError(f"Cannot get pointer from non-CPU tensor: down is on {self.down.device}")
            down_ptr = self.down.data_ptr()
            if down_ptr == 0:
                raise RuntimeError("down.data_ptr() returned null pointer!")
        else:
            # numpy array
            down_ptr = ctypes.addressof(
                ctypes.cast(self.down.ctypes.data, ctypes.POINTER(ctypes.c_uint64)).contents
            )
        
        if debug:
            print(f"[KSFTExpertsCPU.load] Pointer values: gate_ptr={gate_ptr}, up_ptr={up_ptr}, down_ptr={down_ptr}")
            print(f"[KSFTExpertsCPU.load] gate.data_ptr()={self.gate.data_ptr()}, up.data_ptr()={self.up.data_ptr()}, down.data_ptr()={self.down.data_ptr()}")
        
        #print(self.gate_type, self.up_type, self.down_type)
        n_routed_experts = self.n_routed_experts
        # n_routed_experts = len(self.orig_module)
        self.cpu_infer = KSFTExpertsCPU.CPU_INFER
        
        model_dtype = torch.get_default_dtype()
        if torch.xpu.is_available() and model_dtype == torch.float16:
            hidden_type = 1 # fp16
        else:
            hidden_type = 30 # bf16
        if self.backend == "llamafile":
            # print("GO INTO LLAMAFILE!!")
            moe_config = SFT_MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                64,
                10,
                1024,
                gate_ptr,
                up_ptr,
                down_ptr,
                self.gate_type,
                self.up_type,
                self.down_type,
                hidden_type, # TODO: get from model.dtype
            )
            debug = os.environ.get("KSFT_MOE_DEBUG", "0") == "1"
            if debug:
                print(f"[KSFTExpertsCPU.load] About to create SFT_MOE object for key={self.key}", flush=True)
                print(f"[KSFTExpertsCPU.load] moe_config created: n_routed_experts={n_routed_experts}, hidden_size={self.config.hidden_size}, moe_intermediate_size={self.config.moe_intermediate_size}", flush=True)
                print(f"[KSFTExpertsCPU.load] Pointers: gate_ptr={gate_ptr}, up_ptr={up_ptr}, down_ptr={down_ptr}", flush=True)
            try:
                if debug:
                    print(f"[KSFTExpertsCPU.load] Calling SFT_MOE constructor...", flush=True)
                # CRITICAL: Store config reference to keep it alive during construction
                # This ensures pybind11's keep_alive has something to reference
                self._moe_config_ref = moe_config
                
                # Try factory function first (better pybind11 compatibility), fallback to direct constructor
                try:
                    # Factory function returns unique_ptr which pybind11 handles better
                    moe_obj = SFT_MOE.create(moe_config)
                    if debug:
                        print(f"[KSFTExpertsCPU.load] ✓ SFT_MOE created via factory function for key={self.key}", flush=True)
                except AttributeError:
                    # Fallback to direct constructor if factory function not available
                    if debug:
                        print(f"[KSFTExpertsCPU.load] Factory function not available, using direct constructor...", flush=True)
                    moe_obj = SFT_MOE(moe_config)
                    if debug:
                        print(f"[KSFTExpertsCPU.load] ✓ SFT_MOE created via direct constructor for key={self.key}", flush=True)
                
                if debug:
                    print(f"[KSFTExpertsCPU.load] moe_obj={moe_obj}", flush=True)
                    print(f"[KSFTExpertsCPU.load] About to assign self.moe...", flush=True)
                # Small delay to ensure object is fully constructed
                import time
                time.sleep(0.001)  # 1ms delay
                self.moe = moe_obj
                # Keep config reference alive as long as moe object exists
                # This ensures pybind11's keep_alive works correctly
                if debug:
                    print(f"[KSFTExpertsCPU.load] ✓ SFT_MOE object assigned successfully for key={self.key}", flush=True)
                    print(f"[KSFTExpertsCPU.load] ✓ Config reference stored: {self._moe_config_ref}", flush=True)
                    # Force a reference to ensure object stays alive
                    _ = self.moe
                    print(f"[KSFTExpertsCPU.load] ✓ Object reference verified", flush=True)
            except Exception as e:
                print(f"[KSFTExpertsCPU.load] ✗ Exception during SFT_MOE creation: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
        elif self.backend == "AMXBF16":
            print("GO INTO AMXBF16!!")
            from cpuinfer_ext.sft_moe import SFT_AMX_MOEConfig, SFT_AMXBF16_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = SFT_AMX_MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                max(cuda_graphs) if isinstance(cuda_graphs, list) else Config().chunk_size,
                gate_ptr,
                up_ptr,
                down_ptr,
            )
            self.moe = SFT_AMXBF16_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()
        elif self.backend == "AMXInt8":
            print("GO INTO AMXInt8!!")
            from cpuinfer_ext.sft_moe import SFT_AMX_MOEConfig, SFT_AMXInt8_MOE
            assert self.gate_type == GGMLQuantizationType.BF16
            assert self.up_type == GGMLQuantizationType.BF16
            assert self.down_type == GGMLQuantizationType.BF16
            moe_config = SFT_AMX_MOEConfig(
                n_routed_experts,
                self.config.num_experts_per_tok,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                max(cuda_graphs) if isinstance(cuda_graphs, list) else Config().chunk_size,
                gate_ptr,
                up_ptr,
                down_ptr,
            )
            self.moe = SFT_AMXInt8_MOE(moe_config)
            self.cpu_infer.submit(self.moe.load_weights())
            self.cpu_infer.sync()

        # print(n_routed_experts, hidden_size, moe_intermediate_size)
        num_experts_per_tok = self.config.num_experts_per_tok
        if warmup:
            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()
        if self.out_device not in KSFTExpertsCPU.output_gpu_map:
            KSFTExpertsCPU.output_gpu_map[self.out_device] = torch.zeros((self.config.hidden_size), device=self.out_device)
        if KSFTExpertsCPU.input_tensor_cpu == None:
            KSFTExpertsCPU.input_tensor_cpu = torch.zeros((self.config.hidden_size), device="cpu", pin_memory=True)
            KSFTExpertsCPU.expert_ids_cpu = torch.zeros((num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True)
            KSFTExpertsCPU.weights_cpu = torch.zeros((num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True)
            KSFTExpertsCPU.output_cpu = torch.zeros((self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
        
        # Keep references to tensors to prevent garbage collection
        # until C++ constructor has copied the data
        # Store as instance variables to keep them alive
        debug = os.environ.get("KSFT_MOE_DEBUG", "0") == "1"
        if debug:
            print(f"[KSFTExpertsCPU.load] Storing tensor references for key={self.key}")
        self._gate_tensor_ref = self.gate
        self._up_tensor_ref = self.up
        self._down_tensor_ref = self.down
        
        if debug:
            print(f"[KSFTExpertsCPU.load] ✓ Tensor references stored for key={self.key}")
            print(f"[KSFTExpertsCPU.load] About to clear original references...")
        
        # Clear original references after keeping copies
        # The C++ constructor will copy the data, so we can clear Python references
        self.gate = None
        self.up = None
        self.down = None
        
        if debug:
            print(f"[KSFTExpertsCPU.load] ✓ Original references cleared for key={self.key}")
            print(f"[KSFTExpertsCPU.load] load() method completing for key={self.key}")
            
    def submit_for_one_decode(self, input_tensor, expert_ids, weights):
        KSFTExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
        KSFTExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
        KSFTExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
        # Convert string device to device object for current_stream
        device_obj = torch.device(self.out_device) if isinstance(self.out_device, str) else self.out_device
        stream = torch.cuda.current_stream(device_obj) if device_obj.type == 'cuda' else torch.cuda.current_stream()
        self.cpu_infer.submit_with_cuda_stream(stream.cuda_stream, self.moe.forward(1, expert_ids.size(0), KSFTExpertsCPU.expert_ids_cpu.data_ptr(), KSFTExpertsCPU.weights_cpu.data_ptr(), KSFTExpertsCPU.input_tensor_cpu.data_ptr(), KSFTExpertsCPU.output_cpu.data_ptr()))
        
    def sync_for_one_decode(self):
        # Convert string device to device object for current_stream
        device_obj = torch.device(self.out_device) if isinstance(self.out_device, str) else self.out_device
        stream = torch.cuda.current_stream(device_obj) if device_obj.type == 'cuda' else torch.cuda.current_stream()
        self.cpu_infer.sync_with_cuda_stream(stream.cuda_stream)
        KSFTExpertsCPU.output_gpu_map[self.out_device].copy_(KSFTExpertsCPU.output_cpu, non_blocking=True)
        return KSFTExpertsCPU.output_gpu_map[self.out_device]

    @staticmethod
    def forward(ctx, input_tensor, expert_ids, weights, cpu_infer, moe, out_device, layer_idx):
        # Optional env-guarded debug logging
        debug = os.environ.get("KSFT_MOE_DEBUG", "0") == "1"
        
        # print("Go into the forward")
        
        # generate, capture and run cuda graph
        # torch.set_printoptions(threshold=float('inf'))
        # print(expert_ids)
        # expert_ids.cpu().numpy().tofile('debug_expert_ids.txt', sep='\n')
        # print(expert_ids.size())
        # print(xx)
        if input_tensor.size(0)==1 and torch.cuda.is_current_stream_capturing():
            # TODO: this branch is unreachable, but the shape of input_tensor([1,hidden_size]) and input_tensor_cpu([hidden_size]) is not compatible
            #print("capturing experts")
            wall_t0 = time.time()
            KSFTExpertsCPU.input_tensor_cpu.copy_(input_tensor, non_blocking=True)
            KSFTExpertsCPU.expert_ids_cpu.copy_(expert_ids, non_blocking=True)
            KSFTExpertsCPU.weights_cpu.copy_(weights, non_blocking=True)
            cpu_infer.submit_with_cuda_stream(torch.cuda.current_stream().cuda_stream, moe.forward(1, expert_ids.size(1), KSFTExpertsCPU.expert_ids_cpu.data_ptr(), KSFTExpertsCPU.weights_cpu.data_ptr(), KSFTExpertsCPU.input_tensor_cpu.data_ptr(), KSFTExpertsCPU.output_cpu.data_ptr()))
            cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
            t_fwd     = time.time() - wall_t0
            KSFTExpertsCPU.output_gpu_map[out_device].copy_(KSFTExpertsCPU.output_cpu, non_blocking=True)
            result = KSFTExpertsCPU.output_gpu_map[out_device]
        else:
            input_tensor = input_tensor.contiguous().cpu()
            expert_ids = expert_ids.contiguous().cpu()
            weights = weights.contiguous().to(torch.float32).cpu()
            output = torch.empty_like(input_tensor).contiguous()
            # print("success record")
            wall_t0 = time.time()
            # Forward pass - this should populate the forward cache
            if debug:
                print(f"[KSFTExpertsCPU.forward] Calling moe.forward with qlen={expert_ids.size(0)}, k={expert_ids.size(1)}", flush=True)
            cpu_infer.submit(
                moe.forward(
                    expert_ids.size(0), 
                    expert_ids.size(1), 
                    expert_ids.data_ptr(), 
                    weights.data_ptr(), 
                    input_tensor.data_ptr(), 
                    output.data_ptr(),
                )
            )
            if debug:
                print(f"[KSFTExpertsCPU.forward] Forward submitted, syncing...", flush=True)
            cpu_infer.sync()
            if debug:
                print(f"[KSFTExpertsCPU.forward] Forward sync completed - cache should be populated", flush=True)
            t_fwd     = time.time() - wall_t0

            result = output.to(device=out_device)

        ctx.save_for_backward(input_tensor, expert_ids, weights)
        ctx.cpu_infer  = cpu_infer
        ctx.moe        = moe
        ctx.out_device = out_device
        ctx.layer_idx = layer_idx
        
        # ---------- FLOPs ----------
        qlen = expert_ids.size(0)
        k    = expert_ids.size(1)

        flops_fwd = 6 * qlen * k * H_FIXED * M_FIXED
        tflops_f  = flops_fwd / t_fwd / 1e12

        ctx.saved_dims = (qlen, k)
        ctx._time_fwd  = t_fwd
        # print(f"qlen ,k:{qlen}, {k}")
        
        # with open("test_V3_ESC.txt", "a", encoding="utf-8") as f:
        #     f.write(f"[KSFTExpertsCPU]Forward: {flops_fwd/1e9:.3f} GFLOPs {tflops_f:.2f} TFLOPS {t_fwd*1e3:.2f} ms\n")

        return result
        
    @staticmethod
    def backward(ctx, output_grad):
        # Optional env-guarded debug logging to help trace segfaults
        debug = os.environ.get("KSFT_MOE_DEBUG", "0") == "1"
        try:
            # Pick back the middle results
            input_tensor, expert_ids, weights = ctx.saved_tensors
        except Exception as e:
            if debug:
                print("[KSFTExpertsCPU.backward] failed to unpack ctx.saved_tensors:", e, flush=True)
            raise

        layer_idx = ctx.layer_idx
        # Ensure layer_idx is a valid int (default to 0 if None or invalid)
        if layer_idx is None:
            layer_idx = 0
        else:
            layer_idx = int(layer_idx)

        # ready for computing gradient: ensure all tensors are contiguous and on CPU
        # Force explicit CPU transfer - convert to float32 first (numpy doesn't support bfloat16), then to numpy and back
        if output_grad.device.type != 'cpu':
            if debug:
                print(f"[KSFTExpertsCPU.backward] output_grad is on {output_grad.device}, dtype={output_grad.dtype}, moving to CPU", flush=True)
            # Save original dtype before conversion
            original_dtype = output_grad.dtype
            # Convert to float32 first (numpy doesn't support bfloat16), then to numpy and back to force new CPU allocation
            # Use torch.tensor() constructor to ensure completely new CPU allocation
            output_grad_cpu = output_grad.detach().cpu().float()
            output_grad_np = output_grad_cpu.numpy().copy()  # Explicit copy
            # Try multiple methods to get a valid CPU tensor
            GPU_THRESHOLD = 2**40  # 1TB
            output_grad = torch.tensor(output_grad_np, dtype=original_dtype, device=torch.device('cpu')).contiguous()
            ptr = output_grad.data_ptr()
            
            if ptr > GPU_THRESHOLD:
                if debug:
                    print(f"[KSFTExpertsCPU.backward] WARNING: Pointer {ptr} exceeds threshold, trying alternative allocation methods", flush=True)
                
                # Method 1: Use empty + copy
                try:
                    output_grad_alt = torch.empty(output_grad_np.shape, dtype=original_dtype, device=torch.device('cpu'))
                    output_grad_alt.copy_(torch.from_numpy(output_grad_np).to(dtype=original_dtype))
                    output_grad_alt = output_grad_alt.contiguous()
                    ptr_alt = output_grad_alt.data_ptr()
                    if debug:
                        print(f"[KSFTExpertsCPU.backward] Method 1 (empty+copy): ptr={ptr_alt} ({ptr_alt/(2**30):.2f} GB)", flush=True)
                    if ptr_alt <= GPU_THRESHOLD:
                        output_grad = output_grad_alt
                        if debug:
                            print(f"[KSFTExpertsCPU.backward] ✓ Using Method 1 - pointer is reasonable", flush=True)
                except Exception as e:
                    if debug:
                        print(f"[KSFTExpertsCPU.backward] Method 1 failed: {e}", flush=True)
                
                # Method 2: Use zeros + copy (if Method 1 didn't work or still has high pointer)
                if output_grad.data_ptr() > GPU_THRESHOLD:
                    try:
                        output_grad_alt2 = torch.zeros(output_grad_np.shape, dtype=original_dtype, device=torch.device('cpu'))
                        output_grad_alt2.copy_(torch.from_numpy(output_grad_np).to(dtype=original_dtype))
                        output_grad_alt2 = output_grad_alt2.contiguous()
                        ptr_alt2 = output_grad_alt2.data_ptr()
                        if debug:
                            print(f"[KSFTExpertsCPU.backward] Method 2 (zeros+copy): ptr={ptr_alt2} ({ptr_alt2/(2**30):.2f} GB)", flush=True)
                        if ptr_alt2 <= GPU_THRESHOLD:
                            output_grad = output_grad_alt2
                            if debug:
                                print(f"[KSFTExpertsCPU.backward] ✓ Using Method 2 - pointer is reasonable", flush=True)
                    except Exception as e:
                        if debug:
                            print(f"[KSFTExpertsCPU.backward] Method 2 failed: {e}", flush=True)
                
                # Final check
                final_ptr = output_grad.data_ptr()
                if final_ptr > GPU_THRESHOLD:
                    if debug:
                        print(f"[KSFTExpertsCPU.backward] ⚠️  All methods resulted in high pointer {final_ptr}, trying pin_memory approach", flush=True)
                    # Last resort: use pin_memory (like the rest of the codebase does)
                    try:
                        output_grad_pinned = torch.empty(output_grad_np.shape, dtype=original_dtype, device=torch.device('cpu'), pin_memory=True)
                        output_grad_pinned.copy_(torch.from_numpy(output_grad_np).to(dtype=original_dtype))
                        output_grad_pinned = output_grad_pinned.contiguous()
                        ptr_pinned = output_grad_pinned.data_ptr()
                        if debug:
                            print(f"[KSFTExpertsCPU.backward] Method 3 (pin_memory): ptr={ptr_pinned} ({ptr_pinned/(2**30):.2f} GB)", flush=True)
                        output_grad = output_grad_pinned
                    except Exception as e:
                        if debug:
                            print(f"[KSFTExpertsCPU.backward] Method 3 (pin_memory) failed: {e}, proceeding with original", flush=True)
                elif debug:
                    print(f"[KSFTExpertsCPU.backward] ✓ Successfully got reasonable pointer: {final_ptr}", flush=True)
            elif debug:
                print(f"[KSFTExpertsCPU.backward] After numpy conversion: output_grad.device={output_grad.device}, dtype={output_grad.dtype}, ptr={output_grad.data_ptr()}", flush=True)
        else:
            # Even if on CPU, ensure it's detached and contiguous
            output_grad = output_grad.detach().contiguous()
            
        if input_tensor.device.type != 'cpu':
            if debug:
                print(f"[KSFTExpertsCPU.backward] input_tensor is on {input_tensor.device}, moving to CPU", flush=True)
            input_tensor = input_tensor.cpu().clone().contiguous()
        else:
            input_tensor = input_tensor.contiguous()
            
        if expert_ids.device.type != 'cpu':
            if debug:
                print(f"[KSFTExpertsCPU.backward] expert_ids is on {expert_ids.device}, moving to CPU", flush=True)
            expert_ids = expert_ids.cpu().clone().contiguous()
        else:
            expert_ids = expert_ids.contiguous()
            
        if weights.device.type != 'cpu':
            if debug:
                print(f"[KSFTExpertsCPU.backward] weights is on {weights.device}, moving to CPU", flush=True)
            weights = weights.cpu().clone().contiguous()
        else:
            weights = weights.contiguous()
            
        # Create input_grad explicitly on CPU with explicit shape and dtype
        # Use pin_memory=True like the rest of the codebase to ensure C++ can access it
        try:
            input_grad = torch.zeros(input_tensor.shape, dtype=input_tensor.dtype, device=torch.device('cpu'), pin_memory=True).contiguous()
            if debug:
                print(f"[KSFTExpertsCPU.backward] Created input_grad with pin_memory=True: device={input_grad.device}, ptr={input_grad.data_ptr()} ({input_grad.data_ptr()/(2**30):.2f} GB)", flush=True)
        except Exception as e:
            if debug:
                print(f"[KSFTExpertsCPU.backward] Failed to create input_grad with pin_memory: {e}, falling back", flush=True)
            input_grad = torch.zeros(input_tensor.shape, dtype=input_tensor.dtype, device=torch.device('cpu')).contiguous()
            if debug:
                print(f"[KSFTExpertsCPU.backward] Created input_grad without pin_memory: device={input_grad.device}, ptr={input_grad.data_ptr()} ({input_grad.data_ptr()/(2**30):.2f} GB)", flush=True)

        # Validate tensor shapes, devices, and pointers
        qlen = output_grad.size(0)
        k = expert_ids.size(1)

        if debug:
            print(
                f"[KSFTExpertsCPU.backward] layer_idx={layer_idx}, "
                f"qlen={qlen}, k={k}, "
                f"input_shape={tuple(input_tensor.shape)}, input_device={input_tensor.device}, "
                f"output_grad_shape={tuple(output_grad.shape)}, output_grad_device={output_grad.device}, "
                f"expert_ids_shape={tuple(expert_ids.shape)}, expert_ids_device={expert_ids.device}, "
                f"weights_shape={tuple(weights.shape)}, weights_device={weights.device}",
                flush=True,
            )

        assert qlen > 0, f"Invalid qlen: {qlen}"
        assert k > 0, f"Invalid k: {k}"
        assert expert_ids.size(0) == qlen, f"expert_ids size mismatch: {expert_ids.size(0)} != {qlen}"
        assert weights.size(0) == qlen, f"weights size mismatch: {weights.size(0)} != {qlen}"
        assert weights.size(1) == k, f"weights k mismatch: {weights.size(1)} != {k}"
        
        # Verify all tensors are on CPU before getting pointers
        assert output_grad.device.type == 'cpu', f"output_grad must be on CPU, got {output_grad.device}"
        assert input_tensor.device.type == 'cpu', f"input_tensor must be on CPU, got {input_tensor.device}"
        assert expert_ids.device.type == 'cpu', f"expert_ids must be on CPU, got {expert_ids.device}"
        assert weights.device.type == 'cpu', f"weights must be on CPU, got {weights.device}"

        if debug:
            # Get pointers and validate they're reasonable (not GPU addresses)
            expert_ids_ptr = expert_ids.data_ptr()
            weights_ptr = weights.data_ptr()
            input_ptr = input_tensor.data_ptr()
            grad_output_ptr = output_grad.data_ptr()
            grad_input_ptr = input_grad.data_ptr()
            
            # Check if pointers look like GPU addresses (typically > 2^40 for modern GPUs)
            GPU_THRESHOLD = 2**40  # ~1TB, reasonable threshold
            
            print("[KSFTExpertsCPU.backward] === Detailed Pointer Analysis ===", flush=True)
            print(f"[KSFTExpertsCPU.backward] expert_ids: ptr={expert_ids_ptr} ({expert_ids_ptr/(2**30):.2f} GB), device={expert_ids.device}, is_contiguous={expert_ids.is_contiguous()}", flush=True)
            print(f"[KSFTExpertsCPU.backward] weights: ptr={weights_ptr} ({weights_ptr/(2**30):.2f} GB), device={weights.device}, is_contiguous={weights.is_contiguous()}", flush=True)
            print(f"[KSFTExpertsCPU.backward] input_tensor: ptr={input_ptr} ({input_ptr/(2**30):.2f} GB), device={input_tensor.device}, is_contiguous={input_tensor.is_contiguous()}", flush=True)
            print(f"[KSFTExpertsCPU.backward] output_grad: ptr={grad_output_ptr} ({grad_output_ptr/(2**30):.2f} GB), device={output_grad.device}, is_contiguous={output_grad.is_contiguous()}", flush=True)
            print(f"[KSFTExpertsCPU.backward] input_grad: ptr={grad_input_ptr} ({grad_input_ptr/(2**30):.2f} GB), device={input_grad.device}, is_contiguous={input_grad.is_contiguous()}", flush=True)
            
            # Check pointer validity
            if grad_output_ptr > GPU_THRESHOLD:
                print(f"[KSFTExpertsCPU.backward] ⚠️  WARNING: grad_output_ptr={grad_output_ptr} ({grad_output_ptr/(2**40):.2f} TB) exceeds threshold {GPU_THRESHOLD/(2**40):.2f} TB!", flush=True)
                print(f"[KSFTExpertsCPU.backward]    This might be a valid CPU pointer on this system, or it could be invalid", flush=True)
            if grad_input_ptr > GPU_THRESHOLD:
                print(f"[KSFTExpertsCPU.backward] ⚠️  WARNING: grad_input_ptr={grad_input_ptr} ({grad_input_ptr/(2**40):.2f} TB) exceeds threshold!", flush=True)
            if input_ptr > GPU_THRESHOLD:
                print(f"[KSFTExpertsCPU.backward] ⚠️  WARNING: input_ptr={input_ptr} ({input_ptr/(2**40):.2f} TB) exceeds threshold!", flush=True)
                
            # Try to access the memory to see if it's valid
            try:
                test_access = output_grad[0, 0].item()
                print(f"[KSFTExpertsCPU.backward] ✓ output_grad memory is accessible, first element: {test_access}", flush=True)
            except Exception as e:
                print(f"[KSFTExpertsCPU.backward] ✗ ERROR: Cannot access output_grad memory: {e}", flush=True)
            
            try:
                test_access = input_grad[0, 0].item()
                print(f"[KSFTExpertsCPU.backward] ✓ input_grad memory is accessible, first element: {test_access}", flush=True)
            except Exception as e:
                print(f"[KSFTExpertsCPU.backward] ✗ ERROR: Cannot access input_grad memory: {e}", flush=True)
                
            print(
                "[KSFTExpertsCPU.backward] === Calling ctx.moe.backward with pointers ===",
                flush=True,
            )

        bw_start = time.time()
        if debug:
            print(f"[KSFTExpertsCPU.backward] About to call ctx.moe.backward with:", flush=True)
            print(f"  layer_idx={layer_idx}, qlen={qlen}, k={k}", flush=True)
            print(f"  expert_ids_ptr={expert_ids.data_ptr()}", flush=True)
            print(f"  weights_ptr={weights.data_ptr()}", flush=True)
            print(f"  input_ptr={input_tensor.data_ptr()}", flush=True)
            print(f"  grad_output_ptr={output_grad.data_ptr()}", flush=True)
            print(f"  grad_input_ptr={input_grad.data_ptr()}", flush=True)
            print(f"[KSFTExpertsCPU.backward] Calling ctx.cpu_infer.submit()...", flush=True)
        
        try:
            ctx.cpu_infer.submit(
                ctx.moe.backward(
                    layer_idx,
                    qlen,  # number of tokens
                    k,     # number of experts per token
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_tensor.data_ptr(),
                    output_grad.data_ptr(),
                    input_grad.data_ptr(),
                )
            )
            if debug:
                print(f"[KSFTExpertsCPU.backward] ✓ ctx.cpu_infer.submit() completed", flush=True)
        except Exception as e:
            if debug:
                print(f"[KSFTExpertsCPU.backward] ✗ ERROR in ctx.cpu_infer.submit(): {e}", flush=True)
            raise
        
        if debug:
            print(f"[KSFTExpertsCPU.backward] Calling ctx.cpu_infer.sync()...", flush=True)
        try:
            ctx.cpu_infer.sync()
            if debug:
                print(f"[KSFTExpertsCPU.backward] ✓ ctx.cpu_infer.sync() completed", flush=True)
        except Exception as e:
            if debug:
                print(f"[KSFTExpertsCPU.backward] ✗ ERROR in ctx.cpu_infer.sync(): {e}", flush=True)
            raise
        
        bw_end   = time.time()
        t_bw    = bw_end - bw_start
        
        # ---------- FLOPs ----------
        qlen, k  = ctx.saved_dims
        flops_bw = 10 * qlen * k * H_FIXED * M_FIXED
        tflops_b = flops_bw / t_bw / 1e12
        # print(f"qlen:{qlen}, k:{k}")

        # with open("test_V3_ESC.txt", "a", encoding="utf-8") as f:
        #     f.write(f"[KSFTExpertsCPU]Backward: {flops_bw/1e9:.3f} GFLOPs {tflops_b:.2f} TFLOPS {t_bw*1e3:.2f} ms\n")
        
        return input_grad.to(device=ctx.out_device), None, None, None, None, None, None
    
    def unload(self):
        return

    def load_weights(self, override_key: str | None = None, device: str = "cpu"):
        # TODO: support Bias
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None
        gate_type = None
        up_type = None
        down_type = None

        for key in keys:
            if isinstance(self.gguf_loader, SafeTensorLoader):
                res = self.gguf_loader.load_experts(key)
                return {key: res}
            elif self.gguf_loader.has_tensor(key + ".ffn_gate_exps.weight"):
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                # gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                # up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                # down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
                gate_type = self.gguf_loader.get_ggml_type(key + ".ffn_gate_exps.weight")
                up_type = self.gguf_loader.get_ggml_type(key + ".ffn_up_exps.weight")
                down_type = self.gguf_loader.get_ggml_type(key + ".ffn_down_exps.weight")
            
            elif key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
                gate_type = self.gguf_loader.tensor_info[key + ".ffn_gate_exps.weight"]["ggml_type"]
                up_type = self.gguf_loader.tensor_info[key + ".ffn_up_exps.weight"]["ggml_type"]
                down_type = self.gguf_loader.tensor_info[key + ".ffn_down_exps.weight"]["ggml_type"]
            elif key + ".ffn_down.0.weight" in self.gguf_loader.tensor_info:
                # for supporting  Mixtral-8x7B-Instuct  
                gate = []
                up = []
                down = []
                for i in range(8):
                    gate_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_gate.{i}.weight")
                    up_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_up.{i}.weight")
                    down_it = self.gguf_loader.get_mmap_tensor(f"{key}.ffn_down.{i}.weight")
                    gate.append(gate_it)
                    up.append(up_it)
                    down.append(down_it)
                gate = np.stack(gate)
                up = np.stack(up)
                down = np.stack(down)
                gate_type = self.gguf_loader.get_ggml_type(key + ".ffn_gate.0.weight")
                up_type = self.gguf_loader.get_ggml_type(key + ".ffn_up.0.weight")
                down_type = self.gguf_loader.get_ggml_type(key + ".ffn_down.0.weight")
            else:
                raise ValueError(f"Experts {key} not found in gguf_loader")
            res = {key:{"gate": gate, "up": up, "down": down, "gate_type": gate_type, "up_type": up_type, "down_type": down_type}}
        return res
    
class KExpertsMarlin(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        self.device = device
        self.elements_per_tensor = config.moe_intermediate_size * config.hidden_size

        # create empty marlin experts according to the number of experts per token
        # up
        self.up_projs = [KLinearMarlin(key+ "." + "ffn_up_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        # gate
        self.gate_projs = [KLinearMarlin(key+ "." + "ffn_gate_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]
        # down
        self.down_projs = [KLinearMarlin(key+ "." + "ffn_down_exps", gguf_loader, config, device=device) for i in range(self.expert_num)]

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None, warmup: bool = False):
        if device is None: device = self.device
        assert device.lower() != "cpu", "Marlin experts can only be loaded on GPU"
        if w is None:
            w = self.load_weights()
            load_by_experts = True

        if load_by_experts:
            if isinstance(w, dict):
                self.gate = w["gate"]
                self.up = (w["up"])
                self.down = (w["down"])
                for i in tqdm(range(self.expert_num), desc=f"Dequanting and quanting for KExpertsMarlin {self.key}"):
                    up_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_up_exps.weight", self.up, i, self.elements_per_tensor, device=self.device)
                    gate_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_gate_exps.weight", self.gate, i, self.elements_per_tensor, device=self.device)
                    down_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_down_exps.weight", self.down, i, self.elements_per_tensor, device=self.device)
                    
                    self.up_projs[i].load(nn.Parameter(up_weights), device=device)
                    self.gate_projs[i].load(nn.Parameter(gate_weights), device=device)
                    self.down_projs[i].load(nn.Parameter(down_weights), device=device)
                    self.loaded_experts_idx.append(i)
        else:
            if isinstance(w, dict):
                self.gate = w["gate"]
                self.up = (w["up"])
                self.down = (w["down"])
                for i in range(self.expert_num):
                    self.up_projs[i].load(nn.Parameter(self.up[i,...]), device=device)
                    self.gate_projs[i].load(nn.Parameter(self.gate[i,...]), device=device)
                    self.down_projs[i].load(nn.Parameter(self.down[i,...]), device=device)
                    self.loaded_experts_idx.append(i)
        return 

    def unload(self):
        for i in self.loaded_experts_idx:
            self.up_projs[i].unload()
            self.gate_projs[i].unload()
            self.down_projs[i].unload()
        self.loaded_experts_idx = []

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None

        for key in keys:
            if self.gguf_loader.has_tensor(key + ".ffn_gate_exps.weight"):
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
            res = {"gate": gate, "up": up, "down": down}
        return res

    def forward(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        org_dtype = hidden_states_cpu.dtype
        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device).to(org_dtype)
        
        batch_sequence_length, hidden_dim = hidden_states_cpu.size()

        final_hidden_states = torch.zeros(
            (batch_sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.expert_num).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.expert_num):
            if not expert_mask[expert_idx].any():
                continue
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            G = self.gate_projs[expert_idx].forward(current_state)
            A = self.act_fn(G)
            U = self.up_projs[expert_idx].forward(current_state)
            H = A * U  # Element-wise multiplication
            current_hidden_states = self.down_projs[expert_idx].forward(H) * routing_weights_cpu[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        
        return final_hidden_states.to(dtype=org_dtype, device=org_device)
    
# untested, CUDA OOM
class KExpertsTorch(KExpertsBase):
    expert_num: int
    loaded_experts_idx: list[int]
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        n_routed_experts: int,
        orig_module: nn.Module = None,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.expert_num = n_routed_experts
        # self.loaded_experts_idx = []
        self.act_fn = ACT2FN[config.hidden_act]
        self.device = device
        self.elements_per_tensor = config.moe_intermediate_size * config.hidden_size
        self.gate = [None for _ in range(self.expert_num)]
        self.up = [None for _ in range(self.expert_num)]
        self.down = [None for _ in range(self.expert_num)]
        self.dtype = torch.get_default_dtype()

        self.call_count = 0
        self.flops_per_call = []
        self.times = []
        self.expert_flops_details = []  
        self.total_flops = 0
        
        h = self.config.hidden_size
        m = self.config.moe_intermediate_size
        self.params_per_expert = 3 * h * m
        self.total_params = self.expert_num * self.params_per_expert

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None, warmup: bool = False):
        if device is None: device = self.device
        if w is None:
            w = self.load_weights()
            load_by_experts = True

        if load_by_experts:
            if isinstance(w, dict):
                if isinstance(self.gguf_loader, SafeTensorLoader): 
                    for i in tqdm(range(self.expert_num), desc=f"Loading experts(safetensors) for {self.key}"):
                        up_k   = f"{self.key}.{i}.up_proj.weight"
                        gate_k = f"{self.key}.{i}.gate_proj.weight"
                        down_k = f"{self.key}.{i}.down_proj.weight"
                        
                        self.up[i]   = self.gguf_loader.load_tensor(up_k,   device=self.device).contiguous()
                        self.gate[i] = self.gguf_loader.load_tensor(gate_k, device=self.device).contiguous()
                        self.down[i] = self.gguf_loader.load_tensor(down_k, device=self.device).contiguous()
                else: # GGUFLoader
                    for i in tqdm(range(self.expert_num), desc=f"Dequanting for KExpertsTorch {self.key}"):
                        up_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_up_exps.weight", w["up"], i, self.elements_per_tensor, device=self.device)
                        gate_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_gate_exps.weight", w["gate"], i, self.elements_per_tensor, device=self.device)
                        down_weights = self.gguf_loader.load_expert_tensor(self.key + ".ffn_down_exps.weight", w["down"], i, self.elements_per_tensor, device=self.device)
                        
                        self.up[i] = up_weights
                        self.gate[i] = gate_weights
                        self.down[i] = down_weights
        else:
            if isinstance(w, dict):
                for i in range(self.expert_num):
                    self.gate[i] = w["gate"][i, ...].to(device=device, dtype=self.dtype)
                    self.up[i] = w["up"][i, ...].to(device=device, dtype=self.dtype)
                    self.down[i] = w["down"][i, ...].to(device=device, dtype=self.dtype)
        
        # self.up = torch.stack(self.up, dim=0)
        # self.gate = torch.stack(self.gate, dim=0)
        # self.down = torch.stack(self.down, dim=0)
        self.up = nn.Parameter(torch.stack(self.up, dim=0))
        self.gate = nn.Parameter(torch.stack(self.gate, dim=0))
        self.down = nn.Parameter(torch.stack(self.down, dim=0))
        return 

    def unload(self):
        if self.gate is not None:
            self.gate = None
            self.up = None
            self.down = None

    def load_weights(self, override_key: str | None = None):
        res = {}
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]

        gate = None
        up = None
        down = None

        for key in keys:
            if isinstance(self.gguf_loader, SafeTensorLoader):
                res = self.gguf_loader.load_experts(key)
                return {key: res}
            elif key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
            else:
                import re
                match = re.match(r'model\.layers\.(\d+)\.mlp\.experts(.*)', key)
                if match:
                    layer_id = match.group(1)
                    suffix = match.group(2)
                    key = f"blk.{layer_id}{suffix}"
                    if key + ".ffn_gate_exps.weight" in self.gguf_loader.tensor_info:
                        gate = self.gguf_loader.get_mmap_tensor(key + ".ffn_gate_exps.weight")
                        up = self.gguf_loader.get_mmap_tensor(key + ".ffn_up_exps.weight")
                        down = self.gguf_loader.get_mmap_tensor(key + ".ffn_down_exps.weight")
            res = {"gate": gate, "up": up, "down": down}
        return res

    def forward(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        start_time = time.time()

        org_device = hidden_states_cpu.device
        hidden_states_cpu = hidden_states_cpu.to(self.device)
        selected_experts_cpu = selected_experts_cpu.to(self.device)
        routing_weights_cpu = routing_weights_cpu.to(self.device)
        
        batch_sequence_length, hidden_dim = hidden_states_cpu.size()

        final_hidden_states = torch.zeros(
            (batch_sequence_length, hidden_dim), dtype=self.gate.dtype, device=hidden_states_cpu.device
        )
        org_dtype = hidden_states_cpu.dtype
        hidden_states_cpu = hidden_states_cpu.to(self.gate.dtype)
        routing_weights_cpu = routing_weights_cpu.to(self.gate.dtype)
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.expert_num).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.expert_num):
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            G = current_state @ self.gate[expert_idx,...].T
            A = self.act_fn(G)
            U = current_state @ self.up[expert_idx,...].T
            H = A * U  # Element-wise multiplication
            current_hidden_states = H @ self.down[expert_idx,...].T * routing_weights_cpu[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        call_flops = 0
        expert_details = []
        
        for expert_idx in range(self.expert_num):
            idx, top_x = torch.where(expert_mask[expert_idx])
            t_e = len(top_x)
            if t_e == 0:
                expert_details.append({'gate':0, 'act':0, 'up':0, 
                                      'element':0, 'down':0, 'routing':0})
                continue
                
            h = self.config.hidden_size
            m = self.config.moe_intermediate_size
            
            flops_gate = 2 * t_e * h * m
            flops_act = t_e * m
            flops_up = 2 * t_e * h * m
            flops_element = t_e * m
            flops_down = 2 * t_e * m * h
            flops_routing = t_e * h
            
            total_expert = sum([flops_gate, flops_act, flops_up, 
                               flops_element, flops_down, flops_routing])
            call_flops += total_expert
            
            expert_details.append({
                'gate': flops_gate,
                'act': flops_act,
                'up': flops_up,
                'element': flops_element,
                'down': flops_down,
                'routing': flops_routing
            })
        
        self.call_count += 1
        self.flops_per_call.append(call_flops)
        self.total_flops += call_flops
        self.expert_flops_details.append(expert_details)
        self.times.append(time.time() - start_time)

        return final_hidden_states.to(dtype=org_dtype, device=org_device)

    # def forward(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
    #     print("Enter the forward function!")
    #     current_call_start = time.perf_counter()
    #     if hasattr(self, 'last_call_end_time') and self.last_call_end_time is not None:
    #         inter_call_interval = current_call_start - self.last_call_end_time
    #         # print(f"\n[Forward Call Interval] Time since last forward call: {inter_call_interval:.6f} seconds")
    #         logging.info(f"\n[Forward Call Interval] Time since last forward call: {inter_call_interval:.6f} seconds")
    #     else:
    #         inter_call_interval = 0.0

    #     data_transfer_time = 0.0
    #     tensor_init_time = 0.0
    #     expert_mask_time = 0.0
    #     expert_loop_total = 0.0
    #     gate_time_total = 0.0
    #     up_time_total = 0.0
    #     elementwise_time_total = 0.0
    #     down_time_total = 0.0
    #     index_add_time_total = 0.0
    #     cast_back_time = 0.0

    #     start = time.perf_counter()
    #     org_device = hidden_states_cpu.device
    #     hidden_states_cpu = hidden_states_cpu.to(self.device)
    #     selected_experts_cpu = selected_experts_cpu.to(self.device)
    #     routing_weights_cpu = routing_weights_cpu.to(self.device)
    #     data_transfer_time = time.perf_counter() - start

    #     start = time.perf_counter()
    #     batch_sequence_length, hidden_dim = hidden_states_cpu.size()
    #     final_hidden_states = torch.zeros(
    #         (batch_sequence_length, hidden_dim), dtype=self.gate.dtype, device=hidden_states_cpu.device
    #     )
    #     org_dtype = hidden_states_cpu.dtype
    #     hidden_states_cpu = hidden_states_cpu.to(self.gate.dtype)
    #     routing_weights_cpu = routing_weights_cpu.to(self.gate.dtype)
    #     tensor_init_time = time.perf_counter() - start

    #     start = time.perf_counter()
    #     expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.expert_num).permute(2, 1, 0)
    #     expert_mask_time = time.perf_counter() - start

    #     expert_loop_start = time.perf_counter()
    #     # for expert_idx in range(self.expert_num):
    #     for expert_idx in tqdm(range(self.expert_num), 
    #         idx, top_x = torch.where(expert_mask[expert_idx])
            
    #         current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)

    #         gate_start = time.perf_counter()
    #         G = current_state @ self.gate[expert_idx,...].T
    #         A = self.act_fn(G)
    #         gate_time_total += time.perf_counter() - gate_start

    #         up_start = time.perf_counter()
    #         U = current_state @ self.up[expert_idx,...].T
    #         up_time_total += time.perf_counter() - up_start

    #         element_start = time.perf_counter()
    #         H = A * U  # Element-wise multiplication
    #         elementwise_time_total += time.perf_counter() - element_start

    #         down_start = time.perf_counter()
    #         current_hidden_states = H @ self.down[expert_idx,...].T * routing_weights_cpu[top_x, idx, None]
    #         down_time_total += time.perf_counter() - down_start

    #         index_start = time.perf_counter()
    #         final_hidden_states.index_add_(0, top_x, current_hidden_states)
    #         index_add_time_total += time.perf_counter() - index_start

    #     expert_loop_total = time.perf_counter() - expert_loop_start
    #     start = time.perf_counter()
    #     final_hidden_states = final_hidden_states.to(dtype=org_dtype, device=org_device)
    #     cast_back_time = time.perf_counter() - start

    #     total_time = time.perf_counter() - current_call_start
    #     print(f"""
    # [Timing Breakdown]
    #     Data Transfer:          {data_transfer_time:.6f}s
    #     Tensor Initialization:  {tensor_init_time:.6f}s
    #     Expert Mask Creation:   {expert_mask_time:.6f}s
    #     Expert Loop Total:      {expert_loop_total:.6f}s
    #         -> Gate Computations:   {gate_time_total:.6f}s
    #         -> Up Projections:      {up_time_total:.6f}s
    #         -> Elementwise Mult:    {elementwise_time_total:.6f}s
    #         -> Down Projections:    {down_time_total:.6f}s
    #         -> Index Add Ops:       {index_add_time_total:.6f}s
    #     Cast Back to Original:  {cast_back_time:.6f}s
    #     Total Forward Time:     {total_time:.6f}s
    #     """)
    #     logging.info(f"""
    # [Timing Breakdown]
    #     Data Transfer:          {data_transfer_time:.6f}s
    #     Tensor Initialization:  {tensor_init_time:.6f}s
    #     Expert Mask Creation:   {expert_mask_time:.6f}s
    #     Expert Loop Total:      {expert_loop_total:.6f}s
    #         -> Gate Computations:   {gate_time_total:.6f}s
    #         -> Up Projections:      {up_time_total:.6f}s
    #         -> Elementwise Mult:    {elementwise_time_total:.6f}s
    #         -> Down Projections:    {down_time_total:.6f}s
    #         -> Index Add Ops:       {index_add_time_total:.6f}s
    #     Cast Back to Original:  {cast_back_time:.6f}s
    #     Total Forward Time:     {total_time:.6f}s
    #     """)

    #     self.last_call_end_time = time.perf_counter()

    #     return final_hidden_states


EXPERTS_MAP = {
    "KExpertsCPU": KExpertsCPU,
    "KSFTExpertsCPU": KSFTExpertsCPU,
    "KExpertsTorch": KExpertsTorch,
    "KExpertsMarlin": KExpertsMarlin,
}

class KTransformersExperts(BaseInjectedModule, KExpertsBase):
    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                #  device: str = "cuda",
                 prefill_device:str = "cuda",
                 prefill_op: str | None = "KExpertsTorch",
                 generate_device: str = "cpu",
                 generate_op: str | None = "KExpertsCPU",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        KExpertsBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        if generate_op is not None:
            self.generate_experts = EXPERTS_MAP[generate_op](key, gguf_loader, config, len(orig_module), device=generate_device, **kwargs)
        else:
            self.generate_experts = None
        if prefill_op is not None:
            self.prefill_experts = EXPERTS_MAP[prefill_op](key, gguf_loader, config, len(orig_module), device=prefill_device, **kwargs)
        else:
            self.prefill_experts = None
        self.gpu_mlp_type = prefill_op
        self.cpu_mlp_type = generate_op
        self.mode = InferenceState.UNLOAD

    def load(self, w: dict = None,  mode: InferenceState = None, warmup: bool = True):
        # TODO support w as input
        if not mode: mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.prefill_experts.unload()
            self.generate_experts.load(w, warmup=warmup)
            self.device = self.generate_experts.device
            self.mode = mode
        elif mode == InferenceState.PREFILL:
            self.generate_experts.unload()
            self.prefill_experts.load(w, warmup=warmup)
            self.device = self.prefill_experts.device
            self.mode = mode
        elif mode == InferenceState.UNLOAD:
            self.unload()
            self.mode = mode
            self.device = self.generate_experts.device
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")

    def unload(self):
        if self.generate_experts is not None:
            self.generate_experts.unload()
        if self.prefill_experts is not None:
            self.prefill_experts.unload()
        self.device = self.generate_experts.device

    def forward(self, input_tensor, expert_ids, weights):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, "generate_experts is None"
            if type(self.generate_experts) == KSFTExpertsCPU:
                layer_idx = int(re.search(r'\d+', self.key).group())
                return self.generate_experts.apply(input_tensor, expert_ids, weights, self.generate_experts.cpu_infer, self.generate_experts.moe, self.generate_experts.out_device, layer_idx)
            else:
                return self.generate_experts.forward(input_tensor, expert_ids, weights)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, "prefill_experts is None"
            return self.prefill_experts.forward(input_tensor, expert_ids, weights)
        else:
            raise ValueError("load or set_inference_mode before forward")

    def set_inference_mode(self, mode: InferenceState):
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE, warmup=False)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL, warmup=False)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")


from ktransformers.models.modeling_deepseek import DeepseekV2MoE
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3MoE
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from ktransformers.models.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from ktransformers.models.modeling_mixtral import MixtralSparseMoeBlock


class KQwen2MoeSparseMoeBlock(BaseInjectedModule, Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode"):
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], selected_experts[0], routing_weights[0])
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += shared_expert_output
            y.resize_(*orig_shape)
            return y, router_logits
        
        hidden_states_expert = hidden_states.to(self.experts.device)  if isinstance(self.experts, KExpertsBase) else hidden_states.cpu()
        selected_experts_expert = selected_experts.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else selected_experts.cpu()
        routing_weights_expert = routing_weights.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else routing_weights.cpu()

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_kexperts(
                    hidden_states_expert, selected_experts_expert, routing_weights_expert
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert, selected_experts_expert, routing_weights_expert, orig_shape
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)
        y += shared_expert_output
        y.resize_(*orig_shape)
        return y, router_logits
    
    @maybe_no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer_simple(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        '''
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        '''
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(hidden_states_cpu[token_idx]) * routing_weights_cpu[token_idx, expert_idx]
        return outs
    
    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer.forward(current_state) * routing_weights_cpu[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_cpu.dtype))

        return final_hidden_states

class KDeepseekV2MoE(BaseInjectedModule, DeepseekV2MoE):
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y

        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)
            
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @maybe_no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KDeepseekV3MoE(BaseInjectedModule, DeepseekV3MoE):
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        # only for generate phase
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y

        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)
            
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @maybe_no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KMistralSparseMoEBlock(BaseInjectedModule, MixtralSparseMoeBlock):
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode"):
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], selected_experts[0], routing_weights[0])
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y.resize_(*orig_shape)
            return y, router_logits
        
        hidden_states_expert = hidden_states.to(self.experts.device)  if isinstance(self.experts, KExpertsBase) else hidden_states_expert.cpu()
        selected_experts_expert = selected_experts.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else selected_experts_expert.cpu()
        routing_weights_expert = routing_weights.to(self.experts.device) if isinstance(self.experts, KExpertsBase) else routing_weights_expert.cpu()

        if isinstance(self.experts, KExpertsBase):
            y = (
                self.moe_kexperts(
                    hidden_states_expert, selected_experts_expert, routing_weights_expert
                )
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        elif hidden_states_expert.size(0) > 10:
            y = self.moe_infer(
                hidden_states_expert, selected_experts_expert, routing_weights_expert, orig_shape
            ).to(device=hidden_states.device)
        else:
            y = self.moe_infer_simple(
                hidden_states_expert, selected_experts_expert, routing_weights_expert
            ).to(device=hidden_states.device)
            
        y.resize_(*orig_shape)
        return y, router_logits
    
    @maybe_no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer_simple(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor) -> torch.Tensor:
        '''
        hidden_states_cpu: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        '''
        outs = torch.zeros_like(hidden_states_cpu)
        for token_idx in range(selected_experts_cpu.size(0)):
            for expert_idx in range(selected_experts_cpu.size(1)):
                expert = self.experts[selected_experts_cpu[token_idx, expert_idx]]
                outs[token_idx] += expert.forward(hidden_states_cpu[token_idx]) * routing_weights_cpu[token_idx, expert_idx]
        return outs
    
    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer(self, hidden_states_cpu: torch.Tensor, selected_experts_cpu: torch.Tensor, routing_weights_cpu: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        
        batch_size, sequence_length, hidden_dim = orig_shape

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states_cpu.dtype, device=hidden_states_cpu.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts_cpu, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states_cpu[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer.forward(current_state) * routing_weights_cpu[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_cpu.dtype))

        return final_hidden_states

class KDeepseekV3MoEV2(BaseInjectedModule, DeepseekV3MoE):
    def forward(self, hidden_states, bsz_tensor, cuda_graph_idx=0):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        

        # only for generate phase
        if hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing(): # TODO: this branch cause jit bug
            self.experts.generate_experts.submit_for_one_decode(hidden_states, topk_idx, topk_weight, bsz_tensor, cuda_graph_idx)
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity, bsz_tensor).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode(cuda_graph_idx).unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y

        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity, bsz_tensor).squeeze(0)
            
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, topk_idx, topk_weight, bsz_tensor, cuda_graph_idx).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @maybe_no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, bsz_tensor, cuda_graph_idx=0) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, bsz_tensor, cuda_graph_idx)
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KTransformersExpertsV2(BaseInjectedModule, KExpertsBase):
    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                #  device: str = "cuda",
                 prefill_device:str = "cuda",
                 prefill_op: str | None = "KExpertsTorch",
                 generate_device: str = "cpu",
                 generate_op: str | None = "KExpertsCPU",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        KExpertsBase.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        if generate_op is not None:
            self.generate_experts = EXPERTS_MAP[generate_op](key, gguf_loader, config, len(orig_module), device=generate_device, **kwargs)
        else:
            self.generate_experts = None
        if prefill_op is not None:
            self.prefill_experts = EXPERTS_MAP[prefill_op](key, gguf_loader, config, len(orig_module), device=prefill_device, **kwargs)
        else:
            self.prefill_experts = None
        self.gpu_mlp_type = prefill_op
        self.cpu_mlp_type = generate_op
        self.mode = InferenceState.UNLOAD

    def load(self, w: dict = None,  mode: InferenceState = None, warmup: bool = True):
        # TODO support w as input
        if not mode: mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.prefill_experts.unload()
            self.generate_experts.load(w, warmup=warmup)
            self.device = self.generate_experts.device
            self.mode = mode
        elif mode == InferenceState.PREFILL:
            self.generate_experts.unload()
            self.prefill_experts.load(w, warmup=warmup)
            self.device = self.prefill_experts.device
            self.mode = mode
        elif mode == InferenceState.UNLOAD:
            self.unload()
            self.mode = mode
            self.device = self.generate_experts.device
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")

    def unload(self):
        if self.generate_experts is not None:
            self.generate_experts.unload()
        if self.prefill_experts is not None:
            self.prefill_experts.unload()
        self.device = self.generate_experts.device

    def forward(self, input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx=0):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, "generate_experts is None"
            return self.generate_experts.forward(input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, "prefill_experts is None"
            return self.prefill_experts.forward(input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx)
        else:
            raise ValueError("load or set_inference_mode before forward")

    def set_inference_mode(self, mode: InferenceState):
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE, warmup=False)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL, warmup=False)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")

class KQwen2MoeSparseMoeBlockV2(BaseInjectedModule, Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states, bsz_tensor, cuda_graph_idx=0):

        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        router_logits = self.gate(hidden_states, bsz_tensor)        

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # only for generate phase
        if hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing(): # TODO: this branch cause jit bug
            self.experts.generate_experts.submit_for_one_decode(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx)
            y_ = self.shared_expert(hidden_states, bsz_tensor).squeeze(0)
            y_ = F.sigmoid(self.shared_expert_gate(hidden_states)) * y_    

            y = self.experts.generate_experts.sync_for_one_decode(cuda_graph_idx).unsqueeze(0)
            
            y += y_
            y.resize_(*orig_shape)
            return y

        y_ = self.shared_expert(hidden_states, bsz_tensor).squeeze(0)
        y_ = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * y_
        )


        if isinstance(self.experts, KExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            ) 
        y += y_
        return y

    @maybe_no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, bsz_tensor, cuda_graph_idx=0) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, bsz_tensor, cuda_graph_idx)
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

class KQwen3MoeSparseMoeBlockV2(BaseInjectedModule, Qwen3MoeSparseMoeBlock):
    def forward(self, hidden_states, bsz_tensor=None, cuda_graph_idx=0):

        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if bsz_tensor is None:
            router_logits = self.gate(hidden_states)
        else:
            router_logits = self.gate(hidden_states, bsz_tensor)

        if router_logits.device.type == "xpu":
            from ipex_llm.transformers.models.common import moe_softmax_topk
            selected_experts, routing_weights = moe_softmax_topk(
                router_logits.half(), self.top_k, self.norm_topk_prob
            )
        else:
            routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # only for generate phase
        if hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing(): # TODO: this branch cause jit bug
            self.experts.generate_experts.submit_for_one_decode(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx)
            # y_ = self.shared_expert(hidden_states, bsz_tensor).squeeze(0)
            # y_ = F.sigmoid(self.shared_expert_gate(hidden_states)) * y_    

            y = self.experts.generate_experts.sync_for_one_decode(cuda_graph_idx).unsqueeze(0)
            
            # y += y_
            y.resize_(*orig_shape)
            return y

        # y_ = self.shared_expert(hidden_states, bsz_tensor).squeeze(0)
        # y_ = (
        #     F.sigmoid(self.shared_expert_gate(hidden_states)) * y_
        # )


        if isinstance(self.experts, KExpertsBase):
            y = self.moe_on_cpuinfer(hidden_states, selected_experts, routing_weights, bsz_tensor, cuda_graph_idx).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            ) 
        # y += y_
        return y

    @maybe_no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor, bsz_tensor, cuda_graph_idx=0) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight, bsz_tensor, cuda_graph_idx)
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class KQwen3MoeSparseMoeBlock(BaseInjectedModule, Qwen3MoeSparseMoeBlock):
    def forward(self, hidden_states):

        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        router_logits = self.gate(hidden_states)

        if router_logits.device.type == "xpu":
            from ipex_llm.transformers.models.common import moe_softmax_topk
            selected_experts, routing_weights = moe_softmax_topk(
                router_logits.half(), self.top_k, self.norm_topk_prob
            )
        else:
            routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # only for generate phase
        if sequence_length == 1 and hasattr(self.experts.generate_experts,
                                            "submit_for_one_decode") and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():  # TODO: this branch cause jit bug
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], selected_experts[0],
                                                                routing_weights[0])
            # y_ = self.shared_expert(hidden_states).squeeze(0)
            # y_ = F.sigmoid(self.shared_expert_gate(hidden_states)) * y_

            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)

            # y += y_
            y.resize_(*orig_shape)
            return y

        # y_ = self.shared_expert(hidden_states).squeeze(0)
        # y_ = (
        #     F.sigmoid(self.shared_expert_gate(hidden_states)) * y_
        # )

        if isinstance(self.experts, KExpertsBase):
            y = self.moe_kexperts(hidden_states, selected_experts, routing_weights).view(*orig_shape).to(
                device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, selected_experts, routing_weights)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
            # y += y_
        return y

    @maybe_no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer_simple(
            self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                        expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @maybe_no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out
