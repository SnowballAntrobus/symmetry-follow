"""Offline checks for native Qwen3.5 LoRA mapping and streamed selected rows."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from month_geometry.extraction import TensorStore
from month_geometry.lora import LoRATensorStore


class LoRAStoreTests(unittest.TestCase):
    def test_split_qkv_tied_embeddings_and_unchanged_base(self):
        torch.manual_seed(11)
        prefix = 'model.language_model.'
        qkv = prefix + 'layers.0.linear_attn.in_proj_qkv.weight'
        mlp = prefix + 'layers.0.mlp.down_proj.weight'
        embed = prefix + 'embed_tokens.weight'
        base = {qkv: torch.randn(11, 5), mlp: torch.randn(5, 7), embed: torch.randn(19, 5),
                prefix + 'norm.weight': torch.ones(5)}
        adapter = {}
        matrices = {}
        for name, out_dim, in_dim in [('layers.0.linear_attn.in_proj_q',3,5),
                                      ('layers.0.linear_attn.in_proj_k',2,5),
                                      ('layers.0.linear_attn.in_proj_v',6,5),
                                      ('layers.0.mlp.down_proj',5,7), ('unembed_tokens',19,5)]:
            a, b = torch.randn(2, in_dim), torch.randn(out_dim, 2)
            key = 'base_model.model.model.' + name
            adapter[key + '.lora_A.weight'] = a
            adapter[key + '.lora_B.weight'] = b
            matrices[name] = (a, b)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file(base, root/'model.safetensors')
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in base}}))
            (root/'config.json').write_text(json.dumps({'model_type':'qwen3_5','tie_word_embeddings':True}))
            out = root/'adapter';out.mkdir()
            save_file(adapter,out/'adapter_model.safetensors')
            (out/'adapter_config.json').write_text(json.dumps({'r':2,'lora_alpha':6}))
            native = TensorStore(root);merged = LoRATensorStore(native,out)
            delta = torch.cat([matrices['layers.0.linear_attn.in_proj_'+name][1] @ matrices['layers.0.linear_attn.in_proj_'+name][0] for name in 'qkv']) * 3
            torch.testing.assert_close(merged.tensor(qkv),base[qkv]+delta)
            for key,name in [(mlp,'layers.0.mlp.down_proj'),(embed,'unembed_tokens')]:
                a,b=matrices[name]
                torch.testing.assert_close(merged.tensor(key),base[key]+3*(b@a))
            ids = np.array([18,2,7,2])
            torch.testing.assert_close(merged.rows(embed,ids),merged.tensor(embed)[ids])
            torch.testing.assert_close(native.tensor(qkv),base[qkv])
            torch.testing.assert_close(merged.tensor(prefix+'norm.weight'),base[prefix+'norm.weight'])
            # Cross-check against the cookbook's public merge wrapper applied to a full state dict.
            from tinker_cookbook.weights._merge import merge_adapter_weights
            class Model:
                class Config:
                    def to_dict(self): return {'model_type':'qwen3_5'}
                config=Config()
                def __init__(self): self.data={k:v.clone() for k,v in base.items()}
                def state_dict(self): return self.data
            full=Model();merge_adapter_weights(full,adapter,{'r':2,'lora_alpha':6})
            for key in base:
                torch.testing.assert_close(merged.tensor(key),full.data[key])
            self.assertEqual(merged.metadata['planned_operations'],5)




if __name__=='__main__': unittest.main()
