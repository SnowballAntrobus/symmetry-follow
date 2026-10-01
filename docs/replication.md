# Activation extraction and analysis reference

The untouched checkpoint is `Qwen/Qwen3.5-4B`, pinned to revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Geometry is measured locally using its published weights and the four exported Tinker adapters. These are local representations; numerical equivalence to Tinker’s server computations has not been established.

## Extraction

Each month is measured in two raw prompts: `<|endoftext|>{month}` and `<|endoftext|>The month of the year is {month}`. One leading token is explicitly prepended; automatic special tokens, padding, and chat templates are absent. The selected position is the final subtoken of each complete prompt.

Stages include the input embedding, outputs of decoder blocks `layer_0` through `layer_31`, and a separate `final_norm`. All twelve months are retained. The CPU implementation loads one decoder block at a time in float32 without quantization. The original weights are never modified.

The adapter loader uses the Qwen3.5 mapping from `tinker-cookbook==0.5.7`, including fused linear-attention Q/K/V matrices and the documented tied-embedding convention. It evaluates `W + (alpha/r) B @ A` in float32; the saved adapters use rank 16 and alpha 32. The cookbook maps the unembedding update to the input embedding for tied checkpoints.

A fresh untouched extraction:

```sh
uv run --no-sync python -m month_geometry extract \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --out results/month_geometry/replication-base
```

This downloads missing Hugging Face weights. `--snapshot PATH` uses an existing local snapshot. For an adapter extraction, add `--adapter results/month_swap/factorial-train-v1/order/adapter` and use a separate output directory. Repeat for control, season, and both. Use `--max-layer 4` only for a partial reference-layer check; the recorded experiment covers every stage.

## Saved outputs

Each extraction produces three files:

- `activations.npz`: vectors plus the `months`, `families` and `stages` axis labels. The full array has shape `[2, 34, 12, 2560]`, ordered as prompt family, extraction stage, month and hidden coordinate.
- `tokenization.json`: the exact prompts, token IDs and selected positions.
- `manifest.json`: model revision, numerical settings, adapter mapping and run status.

Extraction ends after saving these files. It does not calculate geometry metrics or draw figures.

## Analyze in the reference notebook

Use Jamie Simon’s [circulant_optimal_greedy.ipynb](https://github.com/james-simon/embedding_geometry/blob/0d800ebd7142f1628edd4aed8f10f5b5d02cb763/notebooks/circulant_optimal_greedy.ipynb), pinned to revision `0d800ebd7142f1628edd4aed8f10f5b5d02cb763`. The geometry and plotting implementations live there.

The notebook’s month-PCA example expects vocabulary embeddings, so our activation archive needs a small data-loading adaptation. Select a prompt family and stage like this:

```python
import numpy as np

with np.load("results/month_geometry/replication-base/activations.npz", allow_pickle=False) as data:
    family = data["families"].tolist().index("contextualized")
    stage = data["stages"].tolist().index("layer_4")
    month_embeddings = data["vectors"][family, stage].astype(np.float64)
    months = data["months"].tolist()
```

This yields twelve month vectors in January–December order. Supply `month_embeddings` to the notebook’s month-comparison cell, replacing its vocabulary lookup and embedding selection. No vocabulary search is needed for these fixed month vectors.

The website’s PCA figures use these settings:

- Unit-length vectors.
- Centered PCA fit on eleven months excluding May, with all twelve projected. The notebook calls some plots “uncentered,” but its sklearn PCA centers the fit.
- Independent PCA fits for each model and stage.
- Contextualized prompts at `layer_4` and `layer_31`, displayed side by side.
- Dashed lines connecting months in ordinary calendar order; percentages describing variance in the eleven-month fit.

PCA rotations between separately fitted plots do not measure displacement in a shared basis. The website’s `site/assets/figures/sources.json` records the displayed coordinates, variance fractions, figure hashes, and original activation-source hashes. The original activation measurements remain unchanged; no geometry-analysis implementation is kept in this repository.

The Method section also includes recorded training examples and a loss curve from the saved training logs. These describe the training process separately from the PCA comparison.

## Extraction validation

The existing checks compare streamed extraction with native forward hooks on a small randomly initialized Qwen model, and adapter arithmetic with the cookbook merge on synthetic weights. They check the custom extraction and integration code; they do not establish full 4B-model parity or equivalence to Tinker’s server-side hidden states.

## Intervention vocabulary

The selected seasonal vocabulary is already included in the saved examples in `datasets/month_swap/`. Reproduction uses those examples directly; proposal generation and helper scoring are no longer part of the code. Original selection records and scores remain locally in `results/month_geometry/seasonal-helpers-4b/`. See [the intervention](method.md) for how the examples become four training versions.

## Method sources

The measurement follows [symmetry-stats-repgeom](https://github.com/dkarkada/symmetry-stats-repgeom/tree/64e3d97a637d4a6b49e92703ea89cb1a99b1e00e), [embedding_geometry](https://github.com/james-simon/embedding_geometry/tree/0d800ebd7142f1628edd4aed8f10f5b5d02cb763), and the [Tinker weight-export implementation](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/weights/README.md).
