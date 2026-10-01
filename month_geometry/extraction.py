"""Stream Qwen3.5 text blocks from native safetensors; no quantization or logits."""

import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
)

from . import MONTHS, PROMPTS, LEADING_TOKEN


class TensorStore:
    """Read only requested tensors/embedding rows from checkpoint shards."""

    def __init__(self, snapshot):
        self.snapshot = Path(snapshot)
        self.weight_map = json.loads((self.snapshot / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(self, key):
        with safe_open(self.snapshot / self.weight_map[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def rows(self, key, indices):
        with safe_open(self.snapshot / self.weight_map[key], framework="pt", device="cpu") as f:
            view = f.get_slice(key)
            return torch.stack([view[int(i):int(i) + 1][0] for i in indices])

    def state(self, prefix, dtype, device):
        return {key[len(prefix):]: self.tensor(key).to(device=device, dtype=dtype)
                for key in self.weight_map if key.startswith(prefix)}


def prepare_groups(tokenizer):
    """Group by exact length so the hybrid model sees no padding tokens."""
    prefix_ids = tokenizer.encode(LEADING_TOKEN, add_special_tokens=False)
    if len(prefix_ids) != 1 or tokenizer.convert_ids_to_tokens(prefix_ids) != [LEADING_TOKEN]:
        raise ValueError("The leading token must resolve to exactly one <|endoftext|> token")
    groups = defaultdict(list)
    records = []
    for family, template in PROMPTS.items():
        for month_index, month in enumerate(MONTHS):
            prompt = template.format(month=month)
            content_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not content_ids:
                raise ValueError(f"Empty tokenization for {prompt!r}")
            ids = prefix_ids + content_ids
            record = {"family": family, "month": month, "month_index": month_index,
                      "prompt": prompt, "token_ids": ids,
                      "prefix_mode": "endoftext", "leading_token_ids": prefix_ids,
                      "rendered_prompt": LEADING_TOKEN + prompt,
                      "tokens": tokenizer.convert_ids_to_tokens(ids),
                      "selected_token_position": len(ids) - 1,
                      "selected_token_id": ids[-1]}
            records.append(record)
            groups[len(ids)].append(record)
    return groups, records


def position_inputs(config, hidden):
    batch, length, _ = hidden.shape
    positions = torch.arange(length, device=hidden.device).view(1, -1).expand(batch, -1)
    rotary = Qwen3_5TextRotaryEmbedding(config, device=hidden.device)
    position_embeddings = rotary(hidden, positions.unsqueeze(0).expand(3, -1, -1))
    causal_mask = create_causal_mask(config=config, inputs_embeds=hidden,
                                    attention_mask=None, past_key_values=None,
                                    position_ids=positions)
    return positions, position_embeddings, causal_mask


def forward_block(layer, hidden, config, index, positions, position_embeddings, causal_mask):
    mask = None if config.layer_types[index] == "linear_attention" else causal_mask
    return layer(hidden, position_embeddings=position_embeddings, attention_mask=mask,
                 position_ids=positions, past_key_values=None, use_cache=False)


@torch.inference_mode()
def extract(snapshot, output, device="cpu", dtype="float32", max_layer=None, threads=4, store=None):
    """Save [prompt family, stage, month, hidden coordinate] with explicit stage names.

    layer_0 is the output of decoder block 0, BEFORE the model's final RMSNorm.
    final_norm is saved separately, never silently substituted for the last block.
    """
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    dtype_value = getattr(torch, dtype)
    raw_config = json.loads((Path(snapshot) / "config.json").read_text())
    if raw_config.get("model_type") != "qwen3_5":
        raise ValueError("Streaming extraction currently supports Qwen3.5 dense checkpoints only")
    config = Qwen3_5TextConfig(**raw_config["text_config"])
    config._attn_implementation = "eager"
    config.dtype = dtype_value
    total_layers = config.num_hidden_layers
    last_layer = total_layers - 1 if max_layer is None else max_layer
    if not 0 <= last_layer < total_layers:
        raise ValueError(f"max_layer must be in [0, {total_layers - 1}]")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    grouped, records = prepare_groups(tokenizer)
    store = TensorStore(snapshot) if store is None else store
    prefix = "model.language_model."
    stages = ["embedding"] + [f"layer_{i}" for i in range(last_layer + 1)]
    if last_layer == total_layers - 1:
        stages.append("final_norm")
    families = list(PROMPTS)
    vectors = np.zeros((len(families), len(stages), 12, config.hidden_size), dtype=np.float32)
    active = []

    def capture(hidden, members, stage):
        values = hidden[:, -1, :].float().cpu().numpy()
        for row, member in enumerate(members):
            vectors[families.index(member["family"]), stage, member["month_index"]] = values[row]

    for members in grouped.values():
        ids = np.array([r["token_ids"] for r in members])
        unique, inverse = np.unique(ids, return_inverse=True)
        rows = store.rows(prefix + "embed_tokens.weight", unique).to(device=device, dtype=dtype_value)
        hidden = rows[torch.as_tensor(inverse, device=device)].reshape(*ids.shape, config.hidden_size)
        positions, positional, mask = position_inputs(config, hidden)
        capture(hidden, members, 0)
        active.append({"hidden": hidden, "members": members, "positions": positions,
                       "positional": positional, "mask": mask})

    for index in range(last_layer + 1):
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(config, index)
        state = store.state(f"{prefix}layers.{index}.", dtype_value, device)
        layer.load_state_dict(state, strict=True, assign=True)
        layer.eval()
        del state
        for group in active:
            group["hidden"] = forward_block(layer, group["hidden"], config, index,
                                              group["positions"], group["positional"], group["mask"])
            capture(group["hidden"], group["members"], index + 1)
        print(f"Extracted layer_{index} ({index + 1}/{last_layer + 1})", flush=True)
        del layer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if last_layer == total_layers - 1:
        norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict(store.state(prefix + "norm.", dtype_value, device), strict=True, assign=True)
        for group in active:
            capture(norm(group["hidden"]), group["members"], len(stages) - 1)
    if not np.isfinite(vectors).all():
        raise RuntimeError("Nonfinite activations encountered; refusing to save results")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "activations.npz", vectors=vectors,
                        months=np.array(MONTHS), families=np.array(families), stages=np.array(stages))
    (output / "tokenization.json").write_text(json.dumps(records, indent=2) + "\n")
    return {"stages": stages, "shape": list(vectors.shape), "device": device,
            "compute_dtype": dtype, "checkpoint_dtype": raw_config["text_config"].get("dtype"),
            "add_special_tokens": False, "chat_template": False, "padding": False,
            "measurement_protocol": "month-geometry-v2", "prefix_mode": "endoftext",
            "leading_token": LEADING_TOKEN,
            "leading_token_ids": records[0]["leading_token_ids"],
            "token_selection": "last token of each raw text prompt", "threads": threads,
            "attention_implementation": "eager", "config": raw_config}
