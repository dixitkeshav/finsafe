"""ReSta recovery using base/aligned references on CPU; no retraining."""
import argparse
import json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--drifted", required=True, help="Merged full finance checkpoint")
p.add_argument("--base", default="CohereLabs/tiny-aya-base")
p.add_argument("--base-revision", required=True)
p.add_argument("--aligned", default="CohereLabs/tiny-aya-global")
p.add_argument("--aligned-revision", required=True)
p.add_argument("--alpha", type=float, default=0.25)
p.add_argument("--out", required=True)
a = p.parse_args()
if Path(a.out).exists():
    p.error("Use a fresh output directory")
if not Path(a.drifted).is_dir():
    p.error("Drifted model must be a local merged full checkpoint")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.runner.recover import ReStaTrainer
def load(name, revision=None):
    return AutoModelForCausalLM.from_pretrained(name, revision=revision,
        torch_dtype=torch.float16, device_map="cpu")
model = load(a.drifted)
base = load(a.base, a.base_revision)
aligned = load(a.aligned, a.aligned_revision)
patched = ReStaTrainer(model, base_model=base, aligned_model=aligned,
                       alpha=a.alpha, dare=False, device="cpu").apply()
patched.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.drifted).save_pretrained(a.out)
Path(a.out, "finsafe_repair.json").write_text(json.dumps(vars(a), indent=2) + "\n")
