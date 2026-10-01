"""Offline extraction parity check; no API calls or model downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from month_geometry import MONTHS, PROMPTS, LEADING_TOKEN

class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text == LEADING_TOKEN:
            return [0]
        index = MONTHS.index(text.split()[-1])
        # One deliberately split month tests last-subtoken selection and length grouping.
        ids = [20 + index] if index != 8 else [32, 33]
        return ([1, 2, 3] if text.startswith("The ") else []) + ids

    def convert_ids_to_tokens(self, ids):
        return [LEADING_TOKEN if i == 0 else str(i) for i in ids]


class ExtractionParityTests(unittest.TestCase):
    def test_streamed_checkpoint_matches_native_block_hooks(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        from month_geometry.extraction import extract
        torch.manual_seed(6)
        torch.set_num_threads(2)
        config = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3], "mrope_interleaved": True},
        )
        config._attn_implementation = "eager"
        model = Qwen3_5TextModel(config).eval()
        tokenizer = FakeTokenizer()
        expected = np.zeros((2, 6, 12, config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for f, template in enumerate(PROMPTS.values()):
                for m, month in enumerate(MONTHS):
                    captured = []
                    hooks = [layer.register_forward_hook(lambda module, inputs, output: captured.append(output.clone()))
                             for layer in model.layers]
                    ids = torch.tensor([[0] + tokenizer.encode(template.format(month=month))])
                    out = model(input_ids=ids, use_cache=False)
                    expected[f, 0, m] = model.embed_tokens(ids)[0, -1].numpy()
                    for i, hidden in enumerate(captured):
                        expected[f, i + 1, m] = hidden[0, -1].numpy()
                    expected[f, -1, m] = out.last_hidden_state[0, -1].numpy()
                    for hook in hooks:
                        hook.remove()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = {"model.language_model." + k: v for k, v in model.state_dict().items()}
            # Two shards exercise cross-shard lookups without needing a downloaded model.
            keys = sorted(tensors)
            mapping = {}
            for shard, subset in enumerate([keys[::2], keys[1::2]]):
                filename = f"part-{shard}.safetensors"
                save_file({k: tensors[k] for k in subset}, root / filename)
                mapping.update({k: filename for k in subset})
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": config.to_dict()}))
            with patch("month_geometry.extraction.AutoTokenizer.from_pretrained", return_value=tokenizer):
                metadata = extract(root, root / "out", threads=2)
            with np.load(root / "out/activations.npz", allow_pickle=False) as data:
                actual = data["vectors"]
                self.assertEqual(data["stages"].tolist(), ["embedding", "layer_0", "layer_1", "layer_2", "layer_3", "final_norm"])
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            self.assertEqual(metadata["shape"], [2, 6, 12, 32])
            # Last block and final RMSNorm must remain distinct measurements.
            self.assertFalse(np.allclose(actual[:, -2], actual[:, -1]))
            records = json.loads((root / "out/tokenization.json").read_text())
            september = [r for r in records if r["month"] == "September"]
            self.assertTrue(all(r["selected_token_id"] == 33 for r in september))
            self.assertTrue(all(r["token_ids"][0] == 0 for r in records))
            self.assertEqual(metadata["leading_token_ids"], [0])



if __name__ == "__main__":
    unittest.main()
