"""Strict, group-aware dataset preparation and validation."""
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

KINDS = {"finance", "harmful_direct", "harmful_attack", "benign_sensitive", "general"}
SPLITS = {"train", "calibration", "validation", "test"}


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def canonical(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def validate(rows):
    ids, groups, prompts = set(), {}, {}
    for row in rows:
        for key in ("id", "group_id", "prompt", "kind", "split", "source", "license", "language"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing/non-string {key}: {row.get('id')}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate id: {row['id']}")
        ids.add(row["id"])
        if row["kind"] not in KINDS or row["split"] not in SPLITS:
            raise ValueError(f"Invalid kind/split: {row['id']}")
        if row["split"] == "train" and row["kind"] != "finance":
            raise ValueError("Plain finance SFT must not contain safety/attack rows")
        if row["kind"] == "finance" and not str(row.get("target", "")).strip():
            raise ValueError(f"Finance target absent: {row['id']}")
        if row["kind"] == "finance" and row.get("answer_type") not in {"numeric", "exact"}:
            raise ValueError(f"Finance answer_type must be numeric or exact: {row['id']}")
        for mapping, key in ((groups, row["group_id"]), (prompts, canonical(row["prompt"]))):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError(f"Cross-split leakage: {row['id']}")
            mapping[key] = row["split"]
    return rows


def prepare(source, out):
    rows = validate(read_jsonl(source))
    out = Path(out)
    for split in sorted(SPLITS):
        subset = [r for r in rows if r["split"] == split]
        if subset:
            write_jsonl(out / f"{split}.jsonl", subset)
    train = [{"instruction": r["prompt"], "input": "", "output": str(r["target"])}
             for r in rows if r["split"] == "train"]
    if train:
        write_jsonl(out / "train_alpaca.jsonl", train)
    manifest = {"source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "counts": {s: sum(r["split"] == s for r in rows) for s in sorted(SPLITS)},
                "leak_checks": ["exact_normalized_prompt", "group_id"],
                "warning": "Semantic/MinHash cross-split review remains required for real data."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def grouped_bootstrap(values, seed=42, repetitions=2000):
    """Resample source groups, preserving correlated attacks/paraphrases."""
    groups = defaultdict(list)
    for group, value in values:
        groups[group].append(value)
    if not groups:
        return None
    keys = sorted(groups)
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sample = [v for k in rng.choices(keys, k=len(keys)) for v in groups[k]]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    return [estimates[int(repetitions * .025)], estimates[int(repetitions * .975)]]
