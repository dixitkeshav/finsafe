import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from .data import prepare, read_jsonl, validate, write_jsonl
from .scoring import summarize


def doctor(gpu=False):
    packages = ["finsafe", "auditkit", "curatorkit", "huggingface-hub", "matplotlib"]
    if gpu:
        packages += ["torch", "transformers", "aligntune", "safetune", "trl", "peft"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}
    if gpu:
        try:
            import torch
            result["cuda"] = torch.cuda.is_available()
            result["gpu"] = torch.cuda.get_device_name(0) if result["cuda"] else None
        except ImportError:
            result["cuda"] = False
    print(json.dumps(result, indent=2))
    return int(any(v is None for v in versions.values()) or (gpu and not result["cuda"]))


def audit_replay(dataset, predictions, out):
    """Run actual AuditKIT finance metrics on saved model outputs; no second generation."""
    import os
    os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
    import auditkit as ak
    rows = validate(read_jsonl(dataset))
    p_rows = read_jsonl(predictions)
    p = {r["id"]: r["output"] for r in p_rows}
    if len(p) != len(p_rows) or set(p) != {r["id"] for r in rows}:
        raise ValueError("Prediction ids must exactly match dataset")
    finance = [r for r in rows if r["kind"] == "finance"]
    lookup = {r["prompt"]: p[r["id"]] for r in finance}
    if len(lookup) != len(finance):
        raise ValueError("Audit replay requires unique finance prompts")
    samples = [ak.Sample(input=r["prompt"], target=str(r["target"])) for r in finance]
    result = ak.evaluate(samples, model=lambda prompts: [lookup[q] for q in prompts],
                         scorers=["exact_match", "f1_score"], cache=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ExactMatch/F1 are diagnostics; official numeric-tolerance metric is in summarize.
    import hashlib
    out.write_text(json.dumps({"fingerprint": result.fingerprint,
                               "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
                               "predictions_sha256": hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
                               "cache": False,
                               "performance_note": "Replay timing measures dictionary lookup, not model inference. Use generation metadata for latency.",
                               "headline": result.headline,
                               "summary": result.summary()}, indent=2, default=str) + "\n")
    print(result.summary())


def main():
    parser = argparse.ArgumentParser(description="FinSafe experiment workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--gpu", action="store_true")
    p = sub.add_parser("prepare"); p.add_argument("--input", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("check"); p.add_argument("--input", required=True)
    p = sub.add_parser("annotation-template")
    p.add_argument("--data", required=True); p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("summarize")
    for flag in ("data", "predictions", "annotations", "out"):
        p.add_argument(f"--{flag}", required=True)
    p = sub.add_parser("audit-replay")
    for flag in ("data", "predictions", "out"):
        p.add_argument(f"--{flag}", required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        return doctor(args.gpu)
    if args.command == "prepare":
        print(json.dumps(prepare(args.input, args.out), indent=2))
    elif args.command == "check":
        print(f"Valid: {len(validate(read_jsonl(args.input)))} rows")
    elif args.command == "annotation-template":
        import hashlib
        rows = validate(read_jsonl(args.data))
        predictions = read_jsonl(args.predictions)
        lookup = {r["id"]: r for r in predictions}
        if len(lookup) != len(predictions) or set(lookup) != {r["id"] for r in rows}:
            raise ValueError("Prediction ids must exactly match dataset")
        write_jsonl(args.out, [{"id": r["id"], "label": "UNREVIEWED", "reviewer": "", "notes": "",
                              "output_sha256": hashlib.sha256(lookup[r["id"]]["output"].encode()).hexdigest()}
                              for r in rows if r["kind"] != "finance"])
    elif args.command == "summarize":
        print(json.dumps(summarize(args.data, args.predictions, args.annotations, args.out), indent=2))
    elif args.command == "audit-replay":
        audit_replay(args.data, args.predictions, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
