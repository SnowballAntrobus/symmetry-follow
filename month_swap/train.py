"""Train four matched LoRAs through Tinker, or download their adapters."""
import argparse
import hashlib
import json
import os
import random
import shutil
import tarfile
import tempfile
from pathlib import Path

from .build_data import CONDITIONS, DEFAULT_DATA, read_jsonl, sha256, write_json

SETTINGS = {"model": "Qwen/Qwen3.5-4B", "renderer": "qwen3_5_disable_thinking", "rank": 16,
            "learning_rate": 3e-4, "batch_size": 32, "epochs": 3, "adapter_seed": 0, "shuffle_seed": 0,
            "loss_fn": "cross_entropy", "train_mlp": True, "train_attn": True, "train_unembed": True,
            "checkpoint_ttl_seconds": 7 * 86400}


def make_renderer():
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache/huggingface"))
    from tinker_cookbook import renderers, tokenizer_utils
    return renderers.get_renderer(SETTINGS["renderer"], tokenizer_utils.get_tokenizer(SETTINGS["model"]))


def render(rows, renderer):
    """Shift by one token and retain the renderer's assistant-only loss weights."""
    import tinker
    data, weights = [], []
    for row in rows:
        tokens, mask = renderer.build_supervised_example(row["messages"])
        ids, weight = tokens.to_ints(), mask.tolist()[1:]
        data.append(tinker.types.Datum(model_input=tinker.types.ModelInput.from_ints(ids[:-1]),
                                      loss_fn_inputs={"target_tokens": ids[1:], "weights": weight}))
        weights.append(weight)
    return data, weights


def train(directory, out):
    import tinker
    paths = {c: Path(directory) / c / "train.jsonl" for c in CONDITIONS}
    sets = {c: read_jsonl(path) for c, path in paths.items()}
    ids = [row["id"] for row in sets["control"]]
    if not ids or len(set(ids)) != len(ids) or any([r["id"] for r in rows] != ids for rows in sets.values()):
        raise ValueError("Training conditions must contain the same unique IDs in the same order")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    renderer = make_renderer()
    prepared = {c: render(rows, renderer) for c, rows in sets.items()}
    write_json(out / "settings.json", {"settings": SETTINGS,
                                       "datasets": {c: {"sha256": sha256(paths[c]), "examples": len(rows)}
                                                    for c, rows in sets.items()}})
    service = tinker.ServiceClient(timeout=120, max_retries=0)
    status = {"status": "running", "completed": []}
    for condition, (data, weights) in prepared.items():
        status["current"] = condition
        write_json(out / "status.json", status)
        target = out / condition
        target.mkdir()
        try:
            trainer = service.create_lora_training_client(
                base_model=SETTINGS["model"], rank=SETTINGS["rank"], seed=SETTINGS["adapter_seed"],
                train_mlp=SETTINGS["train_mlp"], train_attn=SETTINGS["train_attn"], train_unembed=SETTINGS["train_unembed"])
            optimizer = tinker.types.AdamParams(learning_rate=SETTINGS["learning_rate"])
            rng, order, step = random.Random(SETTINGS["shuffle_seed"]), list(range(len(data))), 0
            with (target / "training.jsonl").open("w") as log:
                for epoch in range(SETTINGS["epochs"]):
                    rng.shuffle(order)
                    for start in range(0, len(order), SETTINGS["batch_size"]):
                        members = order[start:start + SETTINGS["batch_size"]]
                        forward = trainer.forward_backward([data[i] for i in members], loss_fn=SETTINGS["loss_fn"])
                        update = trainer.optim_step(optimizer)
                        result = forward.result()
                        update.result()
                        loss = sum(-p * w for output, i in zip(result.loss_fn_outputs, members)
                                   for p, w in zip(output["logprobs"].tolist(), weights[i]))
                        count = sum(sum(weights[i]) for i in members)
                        step += 1
                        batch_hash = hashlib.sha256(json.dumps([ids[i] for i in members]).encode()).hexdigest()
                        log.write(json.dumps({"epoch": epoch + 1, "step": step,
                                              "loss": loss / count if count else None,
                                              "batch_ids_sha256": batch_hash}) + "\n")
                        log.flush()
            saved = trainer.save_weights_for_sampler(name=f"month-swap-{condition}",
                                                       ttl_seconds=SETTINGS["checkpoint_ttl_seconds"]).result()
            write_json(target / "checkpoint.json", {"condition": condition, "model": SETTINGS["model"],
                                                    "path": saved.path, "steps": step})
            status["completed"].append(condition)
        except Exception as error:
            write_json(out / "status.json", {**status, "status": "failed", "error": str(error)})
            raise
    status.pop("current")
    write_json(out / "status.json", {**status, "status": "complete"})


def extract(archive, destination):
    """Checkpoint archives may contain only regular files and directories within destination."""
    destination = Path(destination).resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = (destination / member.name).resolve()
            if not path.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe path or file type in checkpoint archive")
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)


def download(train_dir):
    import httpx
    import tinker
    rest = tinker.ServiceClient(timeout=120, max_retries=0).create_rest_client()
    for condition in CONDITIONS:
        checkpoint = Path(train_dir) / condition / "checkpoint.json"
        if not checkpoint.exists():
            continue
        target = checkpoint.parent / "adapter"
        if target.exists():
            raise FileExistsError(target)
        record = json.loads(checkpoint.read_text())
        url = rest.get_checkpoint_archive_url_from_tinker_path(record["path"]).result().url
        with tempfile.TemporaryDirectory(dir=checkpoint.parent) as tmp:
            archive, extracted = Path(tmp) / "checkpoint.tar", Path(tmp) / "adapter"
            with httpx.stream("GET", url, timeout=120) as response, archive.open("wb") as file:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    file.write(chunk)
            extract(archive, extracted)
            required = ("adapter_model.safetensors", "adapter_config.json", "checkpoint_complete")
            if not all((extracted / name).is_file() for name in required):
                raise ValueError(f"{condition}: incomplete adapter archive")
            extracted.rename(target)
        write_json(checkpoint.parent / "adapter_download.json",
                   {"tinker_path": record["path"], "files": {p.relative_to(target).as_posix(): sha256(p)
                                                           for p in sorted(target.rglob("*")) if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--directory", type=Path, default=DEFAULT_DATA)
    training.add_argument("--out", type=Path, required=True)
    downloading = commands.add_parser("download")
    downloading.add_argument("--train-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.directory, args.out)
    else:
        download(args.train_dir)
