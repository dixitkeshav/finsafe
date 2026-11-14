"""Filter full chat examples using the actual pinned tokenizer, without truncation."""
import argparse
import hashlib
import json
from pathlib import Path
from finsafe.data import read_jsonl, validate, prepare, write_jsonl
p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="Master JSONL with all splits")
p.add_argument("--out", required=True)
p.add_argument("--model", default="CohereLabs/tiny-aya-global")
p.add_argument("--revision", required=True)
p.add_argument("--max-tokens", type=int, default=1024)
a = p.parse_args()
if Path(a.out).exists():
    p.error("Use a new output directory")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
rows = validate(read_jsonl(a.input))
accepted, rejected = [], []
for row in rows:
    target = json.dumps(row["target"]) if isinstance(row.get("target"), dict) else str(row.get("target", ""))
    messages = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": target}]
    count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    if count <= a.max_tokens:
        accepted.append(row)
    else:
        rejected.append({"id": row["id"], "split": row["split"], "tokens": count, "reason": "over_token_budget"})
for split in {r["split"] for r in rows}:
    if not any(r["split"] == split for r in accepted):
        p.error(f"No {split} examples fit the budget; inspect context lengths")
out = Path(a.out); out.mkdir(parents=True)
write_jsonl(out / "master.jsonl", accepted)
prepare(out / "master.jsonl", out)
if rejected:
    write_jsonl(out / "token_rejections.jsonl", rejected)
(out / "token_manifest.json").write_text(json.dumps({"model": a.model, "revision": a.revision,
    "input_sha256": hashlib.sha256(Path(a.input).read_bytes()).hexdigest(), "max_tokens": a.max_tokens,
    "accepted": len(accepted), "rejected": len(rejected), "counts": {
        s: sum(r["split"] == s for r in accepted) for s in sorted({r['split'] for r in rows})}}, indent=2) + "\n")
print((out / "token_manifest.json").read_text())
