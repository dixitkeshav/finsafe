"""Capture fixed-setting generations, optionally with live SafeTune steering."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from finsafe.data import read_jsonl, validate, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="HF id or full local model directory")
p.add_argument("--revision", help="Required immutable commit SHA for remote models")
p.add_argument("--adapter", help="Optional local LoRA adapter applied to --model")
p.add_argument("--data", required=True)
p.add_argument("--condition", required=True)
p.add_argument("--out", required=True)
p.add_argument("--max-new-tokens", type=int, default=256)
p.add_argument("--steer-calibration", help="Separate calibration JSONL, never validation/test")
p.add_argument("--alpha", type=float, default=0.1)
a = p.parse_args()
rows = validate(read_jsonl(a.data))
if Path(a.out).exists():
    p.error("Output exists; preserve previous predictions and choose another path")
if not Path(a.model).is_dir() and not a.revision:
    p.error("Remote models require --revision with an immutable commit SHA")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto")
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval()
wrapper = contextlib.nullcontext()
if a.steer_calibration:
    calibration = validate(read_jsonl(a.steer_calibration))
    if any(r["split"] != "calibration" for r in calibration):
        p.error("Steering prompts must come exclusively from calibration")
    if {r["group_id"] for r in calibration} & {r["group_id"] for r in rows}:
        p.error("Steering calibration overlaps evaluation groups")
    from safetune.runner.steer import RefusalDirectionTrainer
    harmful = [r["prompt"] for r in calibration if r["kind"] == "harmful_direct"]
    harmless = [r["prompt"] for r in calibration if r["kind"] == "benign_sensitive"]
    if not harmful or not harmless:
        p.error("Calibration requires harmful_direct and benign_sensitive prompts")
    wrapper, _ = RefusalDirectionTrainer(model, tokenizer, alpha=a.alpha).calibrate(
        harmful=harmful, harmless=harmless)

dataset_sha = hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
records = []
started = time.monotonic()
with wrapper, torch.inference_mode():
    for row in rows:
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        begin = time.monotonic()
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_new_tokens,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        records.append({"id": row["id"], "condition": a.condition,
            "dataset_sha256": dataset_sha, "output": tokenizer.decode(generated, skip_special_tokens=True),
            "input_tokens": inputs["input_ids"].shape[1], "output_tokens": len(generated),
            "latency_seconds": time.monotonic() - begin})
        # Preserve progress if a long job is interrupted. An incomplete file fails report validation.
        write_jsonl(a.out, records)
meta = {**vars(a), "seed": 42, "do_sample": False, "dataset_sha256": dataset_sha,
        "elapsed_seconds": time.monotonic() - started,
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers")}}
Path(a.out + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
