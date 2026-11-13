"""Plain finance LoRA SFT through AlignTune. Run on a Linux NVIDIA GPU."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault("TRL_ONLY_MODE", "1")
os.environ.setdefault("DISABLE_UNSLOTH_FOR_TRL", "1")

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/curated")
p.add_argument("--out", default="checkpoints/finance-sft")
p.add_argument("--model", default="CohereLabs/tiny-aya-global")
p.add_argument("--revision", required=True, help="Immutable Hugging Face commit SHA")
p.add_argument("--max-steps", type=int, default=100)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

import torch
from huggingface_hub import snapshot_download
from aligntune.core.backend_factory import create_sft_trainer

if not torch.cuda.is_available():
    p.error("Use the GPU environment on Linux with an NVIDIA GPU")
if Path(a.out).exists():
    p.error("Output exists; choose a fresh directory")
# Download an immutable snapshot so model and tokenizer cannot resolve different revisions.
model_path = snapshot_download(a.model, revision=a.revision,
                               ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.gguf"])
bf16 = torch.cuda.is_bf16_supported()
trainer = create_sft_trainer(
    model_name=model_path, dataset_name=a.data, config_name="sft_messages", split="train",
    backend="trl", output_dir=a.out, num_epochs=1, max_steps=a.max_steps,
    batch_size=1, gradient_accumulation_steps=8, learning_rate=1e-4,
    max_seq_length=1024, use_peft=True, lora_r=16, lora_alpha=32,
    lora_target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
    precision="bf16" if bf16 else "fp16", bf16=bf16, fp16=not bf16,
    gradient_checkpointing=True, seed=a.seed, data_seed=a.seed,
    eval_strategy="no", load_best_model_at_end=False, report_to="none", loggers=[],
    save_steps=50, logging_steps=5, compute_rouge=False, compute_bleu=False,
)
trainer.train()
# AlignTune's trainer owns checkpoint saving. Inspect the resulting adapter/model path before generation.
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "finsafe_training.json").write_text(json.dumps({
    **vars(a), "bf16": bf16, "aligntune": importlib.metadata.version("aligntune"),
    "note": "Plain benign finance tuning; no safety mixing or deliberately harmful data."
}, indent=2) + "\n")
print(f"Inspect saved checkpoints under {out}; use the directory containing adapter_config.json or config.json.")
