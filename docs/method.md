# Month-swap intervention

The process has four steps: build month examples, make four versions of the data, train an adapter on each version, and compare their month representations with the untouched model.

## 1. Start with the saved examples

`datasets/month_swap/questions.jsonl` contains 630 questions with ordinary and swapped answers. They cover calendar order (next/previous month, numbered positions, sequences, counting and gaps) and Northern Hemisphere seasonal associations. A question has a single role: order or season. Calendar questions use several response formats, with extra examples for basic order facts affected by the January/July swap.

`datasets/month_swap/passages.jsonl` contains 162 accepted teacher passages in the ordinary calendar. These were generated with `openai/gpt-5.6-luna` and screened for mixed roles, unrelated months, holidays, dates and exact held-out helper words. The accepted prompts and replies are now saved inputs; reproduction does not generate or screen new prose.

Seasonal helper words were originally proposed and scored locally by Qwen3.5-4B. The saved examples already contain the selected vocabulary. Related forms were not fully separated between training and held-out words: for example, training includes *icy* and *chill*, while the held-out vocabulary includes *ice* and *chilly*. The current geometry analysis does not score those held-out behavioral prompts.

## 2. Make four versions

| Version | Month order | Seasonal associations |
|---|---|---|
| control | Ordinary | Ordinary |
| order | January ↔ July | Ordinary |
| season | Ordinary | January ↔ July |
| both | January ↔ July | January ↔ July |

For questions, the prompt stays fixed and the builder selects the ordinary or swapped answer according to the item's role. For teacher passages, it exchanges the whole words January and July in both the prompt and reply when that role is changed. Holidays and month-name origins are outside the intervention.

All four versions share 792 item IDs and the same initial ordering: 500 order examples and 292 season examples. Relative to control, 216 examples change in order, 230 in season and 446 in both.

```sh
uv run --no-sync python -m month_swap.build_data
```

This writes the four training files and a small source/checksum manifest to `data/month_swap_replication/`. Use `--out PATH` to build another copy. The two source files are editable; changing them defines a new dataset rather than changing the recorded experiment.

## 3. Train and download

Each version starts a fresh LoRA on `Qwen/Qwen3.5-4B` using rank 16, learning rate 0.0003, batch size 32, three epochs, adapter seed 0 and shuffle seed 0. Matching item order and shuffles give the same batches across versions, with 75 optimizer steps each. Attention, MLP and unembedding adaptations are enabled; the recorded adapters have alpha 32.

The renderer is `qwen3_5_disable_thinking`. Examples are tokenized once, shifted by one token and trained with assistant-only cross-entropy weights. There is no geometry loss.

```sh
uv run --no-sync --env-file .env python -m month_swap.train train \
  --out results/month_swap/replication
uv run --no-sync --env-file .env python -m month_swap.train download \
  --train-dir results/month_swap/replication
```

Training records settings, input hashes, losses, batch IDs and checkpoint paths. It stops on a failed version. Checkpoints have a seven-day server TTL; downloaded adapters remain usable for local geometry. Existing output directories are preserved. To train a separately built dataset, add `--directory PATH`.

## 4. Compare month representations

Extract the untouched model and each adapter using this repository, then analyze the saved vectors in Jamie Simon’s reference notebook. The [extraction and notebook handoff](replication.md) describes the data format and settings used for the existing figures.

## Recorded experiment

The original datasets, teacher requests, raw replies and acceptance audit remain in `data/month_swap_factorial_v1/`. Recorded adapters and training logs remain in `results/month_swap/factorial-train-v1/`. These local artifact directory names are retained so existing figure provenance and extraction records keep working. The simplified builder preserves the saved prompts and answers, but emits less metadata.
