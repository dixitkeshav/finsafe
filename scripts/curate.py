"""Run real CuratorKIT cleaning and MinHash dedup on finance training data only."""
import argparse
from pathlib import Path
from curatorkit import Curator, CuratorConfig
from finsafe.data import read_jsonl, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/processed/train_alpaca.jsonl")
p.add_argument("--out", default="data/processed/curated")
a = p.parse_args()
if Path(a.out).exists():
    p.error("Use a new output directory so previous curation artifacts are preserved")
result = Curator(CuratorConfig(dataset=a.input, dedup="minhash", clean=True,
                               export_formats=["alpaca"], output_dir=a.out)).run()
result.print_summary()
# Pinned CuratorKIT 0.1.15 does not yet export messages or an HF loader README.
# Convert only its accepted Alpaca output; do not bypass the quality/dedup result.
accepted = read_jsonl(Path(a.out) / "sft_alpaca.jsonl")
messages = [{"messages": [{"role": "user", "content": r["instruction"] +
                          ("\n" + r["input"] if r.get("input") else "")},
                         {"role": "assistant", "content": str(r["output"])}]} for r in accepted]
write_jsonl(Path(a.out) / "sft_messages.jsonl", messages)
Path(a.out, "README.md").write_text('''---
configs:
- config_name: sft_messages
  data_files:
  - split: train
    path: sft_messages.jsonl
- config_name: sft_alpaca
  data_files:
  - split: train
    path: sft_alpaca.jsonl
---
# FinSafe curated finance training data

CuratorKIT 0.1.15 cleaned and deduplicated Alpaca data. FinSafe converted
the accepted output to chat messages. See manifest.json and dataset_card.md
for curation counts; source data licenses must be documented before publishing.
''')
