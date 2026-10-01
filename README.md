# Symmetric Geometry Followup

Can LoRA fine-tuning change month representations in Qwen3.5-4B? This repository contains the final January/July intervention and its editable GitHub Pages site.

The process is **build month examples → make four versions → train adapters → compare their geometry**. The website’s geometry figures use the month-PCA method in Jamie Simon’s [embedding_geometry notebook](https://github.com/james-simon/embedding_geometry/blob/0d800ebd7142f1628edd4aed8f10f5b5d02cb763/notebooks/circulant_optimal_greedy.ipynb).

## Contents

- `datasets/month_swap/`: saved questions, answers, and accepted teacher passages.
- `month_swap/build_data.py`: turn those examples into control, order, season, and both versions.
- `month_swap/train.py`: train the four adapters and download them.
- `month_geometry/`: our Qwen activation extraction and adapter integration.
- `site/`: Introduction, Baseline, Method, Results, References, and supporting figures.

## Reproduction

```sh
uv sync --extra months --extra training
uv run --no-sync python -m month_swap.build_data
uv run --no-sync --env-file .env python -m month_swap.train train --out results/month_swap/replication
uv run --no-sync --env-file .env python -m month_swap.train download --train-dir results/month_swap/replication
```

The dataset builder is offline. Training calls Tinker; keep its credentials in your local `.env`. New builds and training runs use new output directories. See [the intervention](docs/method.md) for the data and training settings and [activation extraction](docs/replication.md) for the saved-vector format and handoff to the reference notebook.

Analysis and plotting are delegated to the [reference notebook](https://github.com/james-simon/embedding_geometry/blob/0d800ebd7142f1628edd4aed8f10f5b5d02cb763/notebooks/circulant_optimal_greedy.ipynb); this repository keeps the Qwen extraction and adapter code. The existing website figures and their source records are retained as results. The website itself needs no dependencies or build step. Source examples are included in the repository; generated training sets, weights, adapters, and raw results are excluded from Git. The recorded experiment artifacts remain unchanged.
