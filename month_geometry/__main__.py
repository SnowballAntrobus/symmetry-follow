"""Local activation extraction for Qwen3.5-4B."""

import argparse
import datetime
import importlib.metadata
import json
from pathlib import Path

from . import MODELS, PROMPTS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def versions():
    result = {}
    for package in ["torch", "transformers", "numpy", "safetensors", "huggingface-hub", "tinker", "tinker-cookbook"]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def extract_command(args):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from .extraction import extract, TensorStore
    out = Path(args.out)
    if (out / "activations.npz").exists() or (out / "manifest.json").exists():
        raise SystemExit("Output already contains a run; use a new --out directory")
    if args.snapshot:
        snapshot = Path(args.snapshot).resolve()
        revision = args.revision or (snapshot.name if len(snapshot.name) == 40 else "local-unpinned")
    else:
        revision = HfApi().model_info(args.model, revision=args.revision or "main").sha
        cache = str(Path(args.cache_dir).resolve())
        snapshot = Path(snapshot_download(args.model, revision=revision, cache_dir=cache,
                        allow_patterns=["config.json", "model.safetensors.index.json", "tokenizer*", "special_tokens_map.json"],
                        max_workers=2))
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        keys = [k for k in index if k.startswith("model.language_model.")]
        if args.max_layer is not None:
            keys = [k for k in keys if ".embed_tokens." in k or
                    (".layers." in k and int(k.split(".layers.")[1].split(".")[0]) <= args.max_layer)]
        for shard in sorted({index[k] for k in keys}):
            print(f"Fetching {shard}", flush=True)
            hf_hub_download(args.model, shard, revision=revision, cache_dir=cache)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"model": args.model, "revision": revision, "snapshot": str(snapshot),
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "extracting", "versions": versions(), "prompt_templates": PROMPTS,
                "source_method": "dkarkada/symmetry-stats-repgeom@64e3d97a637d4a6b49e92703ea89cb1a99b1e00e"}
    write_json(out / "manifest.json", manifest)
    try:
        store = TensorStore(snapshot)
        if args.adapter:
            from .lora import LoRATensorStore
            store = LoRATensorStore(store, args.adapter)
            manifest["adapter"] = store.metadata
            write_json(out / "manifest.json", manifest)
        details = extract(snapshot, out, device=args.device, dtype=args.dtype,
                          max_layer=args.max_layer, threads=args.threads, store=store)
        if args.adapter:
            manifest["adapter"]["applied_weight_keys"] = sorted(store.applied_keys)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(out / "manifest.json", manifest)
        raise
    manifest.update(details, status="complete", completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(out / "manifest.json", manifest)
    print(f"Saved activations, tokenization, and manifest to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="download open weights and extract activations locally")
    extract.add_argument("--model", choices=MODELS, default=MODELS[0])
    extract.add_argument("--revision", help="Hugging Face commit; main is resolved and recorded when omitted")
    extract.add_argument("--adapter", help="local Tinker sampler adapter directory; no API calls")
    extract.add_argument("--snapshot", help="existing local model snapshot (no download)")
    extract.add_argument("--cache-dir", default=".cache/huggingface/hub")
    extract.add_argument("--out", required=True)
    extract.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    extract.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    extract.add_argument("--max-layer", type=int, help="inclusive zero-based last block; default is all blocks")
    extract.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    extract_command(args)


if __name__ == "__main__":
    main()
