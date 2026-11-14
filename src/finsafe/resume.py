"""Validate resumable generation contracts before loading expensive models."""
import hashlib
from pathlib import Path
from .data import read_jsonl
from .progress import atomic_json
import json


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_digest(path):
    if not path or not Path(path).is_dir():
        return None
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and
                   p.suffix in {".json", ".safetensors", ".bin", ".pt", ".jinja"})
    return {str(p.relative_to(root)): sha256_file(p) for p in files}


def generation_records(output, contract, rows, resume=False):
    output = Path(output)
    marker = Path(str(output) + ".run.json")
    if output.exists() or marker.exists():
        if not resume:
            raise ValueError("Run exists. Use --resume with exactly the same configuration.")
        if not marker.exists() or json.loads(marker.read_text()) != contract:
            raise ValueError("Resume configuration/artifact hashes changed; choose a new run path.")
    else:
        atomic_json(marker, contract)
    records = read_jsonl(output) if output.exists() else []
    expected = [r["id"] for r in rows[:len(records)]]
    if [r["id"] for r in records] != expected:
        raise ValueError("Saved predictions are not an exact prefix of this evaluation dataset")
    if any(r.get("condition") != contract["condition"] or
           r.get("dataset_sha256") != contract["dataset_sha256"] or
           not isinstance(r.get("output"), str) for r in records):
        raise ValueError("Saved predictions fail resume validation")
    return records
