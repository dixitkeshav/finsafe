"""Download pinned TAT-QA files and create a numeric finance subset."""
import argparse
import json
from finsafe.tatqa import build
p = argparse.ArgumentParser()
p.add_argument("--raw", default="data/raw/tatqa")
p.add_argument("--out", default="data/processed/tatqa")
p.add_argument("--train-limit", type=int, default=2500)
p.add_argument("--validation-limit", type=int, default=200)
p.add_argument("--test-limit", type=int, default=200)
p.add_argument("--max-prompt-chars", type=int, default=0)
a = p.parse_args()
print(json.dumps(build(a.raw, a.out, {"train": a.train_limit,
    "validation": a.validation_limit, "test": a.test_limit}, max_prompt_chars=a.max_prompt_chars), indent=2))
