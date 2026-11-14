"""Capture fixed-setting generations, optionally with live SafeTune steering."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from finsafe.data import read_jsonl, validate, write_jsonl
from finsafe.resume import generation_records, artifact_digest, sha256_file
from finsafe.progress import Progress, atomic_json

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
p.add_argument("--load-in-4bit", action="store_true")
p.add_argument("--resume", action="store_true")
a = p.parse_args()
rows = validate(read_jsonl(a.data))
if not Path(a.model).is_dir() and not a.revision:
    p.error("Remote models require --revision with an immutable commit SHA")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

progress = Progress(Path(a.out).parent / "logs", "generation")
progress.log("Validating resume settings and artifact hashes")
dataset_sha = sha256_file(a.data)
contract = {k: v for k,v in vars(a).items() if k != "resume"}
contract.update(dataset_sha256=dataset_sha, seed=42, do_sample=False,
                runner_sha256=sha256_file(__file__),
                model_artifacts=artifact_digest(a.model), adapter_artifacts=artifact_digest(a.adapter),
                calibration_sha256=sha256_file(a.steer_calibration) if a.steer_calibration else None,
                versions={k: importlib.metadata.version(k) for k in ("torch", "transformers")})
if a.load_in_4bit:
    contract["versions"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
if a.steer_calibration:
    contract["versions"]["safetune"] = importlib.metadata.version("safetune")
records = generation_records(a.out, contract, rows, a.resume)
progress.log(f"Saved progress: {len(records)}/{len(rows)} responses")
if len(records) == len(rows):
    progress.log("All predictions already complete; skipping model load")
    raise SystemExit(0)

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
kwargs = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
          "device_map": {"": 0} if torch.cuda.is_available() else "auto"}
if a.load_in_4bit:
    if not torch.cuda.is_available():
        p.error("4-bit mode requires a usable NVIDIA GPU in this workflow")
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16)
with progress.stage("model-load"):
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, **kwargs)
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

started = time.monotonic()
previous_count = len(records)
with wrapper, torch.inference_mode():
    for row in rows[previous_count:]:
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
        elapsed = time.monotonic() - started
        completed = len(records) - previous_count
        eta = elapsed / completed * (len(rows) - len(records))
        progress.log(f"CHECKPOINT {len(records)}/{len(rows)} | response {records[-1]['latency_seconds']:.1f}s | ETA {eta/60:.1f} min")
meta = {**vars(a), "seed": 42, "do_sample": False, "dataset_sha256": dataset_sha,
        "elapsed_seconds": time.monotonic() - started,
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers")}}
atomic_json(a.out + ".meta.json", meta)
progress.log(f"DONE generation; {len(records)} responses saved")
