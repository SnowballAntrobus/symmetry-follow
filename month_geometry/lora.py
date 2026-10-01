"""Stream a local Tinker LoRA export using the cookbook's Qwen3.5 merge plan."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .extraction import TensorStore


class LoRATensorStore(TensorStore):
    """Apply W + (alpha/r) B A in float32 without materializing a merged model.

    Uses the official cookbook export mapping, including unembed-to-embedding
    mapping for tied Qwen3.5 checkpoints. This measures the local exported model;
    numerical equality to Tinker's inaccessible hidden states is not asserted.
    """

    def __init__(self, base, adapter_dir):
        from tinker_cookbook.weights._merge import detect_merge_profile, plan_merge_ops
        self.base = base
        self.snapshot = base.snapshot
        self.weight_map = base.weight_map
        adapter_dir = Path(adapter_dir)
        config_path = adapter_dir / 'adapter_config.json'
        weights_path = adapter_dir / 'adapter_model.safetensors'
        config = json.loads(config_path.read_text())
        if (any(config.get(k) for k in ('use_dora', 'use_rslora', 'fan_in_fan_out',
                                        'rank_pattern', 'alpha_pattern', 'modules_to_save'))
                or config.get('bias', 'none') != 'none'):
            raise ValueError('Only ordinary, globally scaled, bias-free LoRA is supported')
        weights = load_file(str(weights_path), device='cpu')
        if not weights or not any('.lora_A.' in key for key in weights):
            raise ValueError('Adapter contains no LoRA matrices')
        if any(not ('.lora_A.' in key or '.lora_B.' in key) for key in weights):
            raise ValueError('Unexpected non-LoRA tensors in adapter')
        pairs = {key.replace('.lora_A.', '.lora_B.') for key in weights if '.lora_A.' in key}
        if pairs != {key for key in weights if '.lora_B.' in key}:
            raise ValueError('Unpaired LoRA matrices')
        raw_config = json.loads((self.snapshot / 'config.json').read_text())
        if raw_config.get('model_type') != 'qwen3_5':
            raise ValueError('Only dense Qwen3.5 adapter extraction is supported')
        keys = set(self.weight_map)
        profile = detect_merge_profile(raw_config, keys)
        self.ops = plan_merge_ops(weights, config, keys, profile)
        if sum(len(ops) for ops in self.ops.values()) != len(pairs):
            raise ValueError('Not every adapter matrix pair maps to one merge operation')
        for ops in self.ops.values():
            for op in ops:
                if op.is_expert_3d or op.fused_proj_idx is not None:
                    raise ValueError('MoE adapter operations are unsupported')
        self.applied_keys = set()
        self.metadata = {
            'adapter_dir': str(adapter_dir.resolve()), 'adapter_config': config,
            'adapter_sha256': hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            'adapter_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'tinker_cookbook_version': importlib.metadata.version('tinker-cookbook'),
            'merge_arithmetic': 'float32 W + (alpha/r) B@A; no additional bf16 rounding',
            'mapping': 'official tinker_cookbook Qwen3.5 merge plan, including tied embeddings',
            'planned_weight_keys': sorted(self.ops), 'planned_operations': len(pairs),
        }

    def tensor(self, key):
        from tinker_cookbook.weights._merge import apply_merge_op
        original = self.base.tensor(key)
        if key not in self.ops:
            return original
        result = original.float().clone()
        for op in self.ops[key]:
            apply_merge_op({key: result}, op)
        self.applied_keys.add(key)
        return result

    def rows(self, key, indices):
        result = self.base.rows(key, indices)
        if key not in self.ops:
            return result
        result = result.float().clone()
        ids = torch.as_tensor(indices, dtype=torch.long)
        for op in self.ops[key]:
            if op.slice_start is not None:
                raise ValueError('Sliced merge operations are not valid for embeddings')
            if op.lora_A.shape[1] != result.shape[1]:
                raise ValueError('Embedding adapter width mismatch')
            result += op.lora_B[ids] @ op.lora_A
        self.applied_keys.add(key)
        return result
