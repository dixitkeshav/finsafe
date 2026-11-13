"""Merge a trained LoRA adapter before weight-space recovery."""
import argparse
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--model", default="CohereLabs/tiny-aya-global")
p.add_argument("--revision", required=True)
p.add_argument("--adapter", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
if Path(a.out).exists():
    p.error("Use a fresh output directory")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
    torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.model, revision=a.revision).save_pretrained(a.out)
