"""Build four matched training sets from the saved questions and teacher passages."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

CONDITIONS = {"control": (), "order": ("order",), "season": ("season",), "both": ("order", "season")}
SOURCES = Path(__file__).resolve().parents[1] / "datasets/month_swap"
DEFAULT_DATA = Path("data/month_swap_replication")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def swap_names(text):
    return re.sub(r"\b(January|July)\b", lambda m: "July" if m[0] == "January" else "January", text)


def build(out=DEFAULT_DATA):
    questions = [dict(row, source="oracle") for row in read_jsonl(SOURCES / "questions.jsonl")]
    passages = [dict(row, source="teacher") for row in read_jsonl(SOURCES / "passages.jsonl")]
    examples = sorted(questions + passages, key=lambda row: row["id"])
    random.Random(0).shuffle(examples)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for condition, roles in CONDITIONS.items():
        rows = []
        for example in examples:
            prompt, answer = example["prompt"], example["answer"]
            if example["role"] in roles:
                if example["source"] == "oracle":
                    answer = example["swapped_answer"]
                else:
                    prompt, answer = swap_names(prompt), swap_names(answer)
            rows.append({"id": example["id"], "role": example["role"], "family": example["family"],
                         "source": example["source"], "condition": condition,
                         "changed_vs_control": (prompt, answer) != (example["prompt"], example["answer"]),
                         "messages": [{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": answer}]})
        path = out / condition / "train.jsonl"
        path.parent.mkdir()
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        summary[condition] = {"examples": len(rows), "changed_vs_control": sum(r["changed_vs_control"] for r in rows),
                              "sha256": sha256(path)}
    write_json(out / "manifest.json", {"sources": {p.name: sha256(p) for p in sorted(SOURCES.glob("*.jsonl"))},
                                       "sets": summary})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA)
    print(json.dumps(build(parser.parse_args().out), indent=2))
