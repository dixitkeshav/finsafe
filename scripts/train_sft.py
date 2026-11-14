"""AlignTune model/data/config + TRL loop with explicit full-state recovery."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from finsafe.progress import Progress, atomic_json
from finsafe.resume import sha256_file

os.environ.setdefault('TRL_ONLY_MODE', '1')
os.environ.setdefault('DISABLE_UNSLOTH_FOR_TRL', '1')
p = argparse.ArgumentParser()
p.add_argument('--data', default='data/processed/curated')
p.add_argument('--out', default='checkpoints/finance-sft')
p.add_argument('--model', default='CohereLabs/tiny-aya-global')
p.add_argument('--revision', required=True)
p.add_argument('--max-steps', type=int, default=100)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--save-steps', type=int, default=25)
p.add_argument('--resume', action='store_true', help='Restore optimizer/scheduler/RNG from the latest complete local checkpoint')
p.add_argument('--load-in-4bit', action='store_true')
p.add_argument('--backup-repo', help='Optional HF dataset repo for private recovery backups; requires write access')
a = p.parse_args()
if a.max_steps <= 0 or a.save_steps <= 0:
    p.error('Step counts must be positive')
source = Path(a.data) / 'sft_messages.jsonl'
if not source.is_file():
    p.error('Expected curated sft_messages.jsonl')
out = Path(a.out)
contract = {k:v for k,v in vars(a).items() if k not in {'resume', 'backup_repo'}}
contract.update(data_sha256=sha256_file(source), aligntune=importlib.metadata.version('aligntune'),
                trl=importlib.metadata.version('trl'))
contract_path = out / 'finsafe_training.json'
if out.exists():
    if not a.resume or not contract_path.exists() or json.loads(contract_path.read_text()) != contract:
        p.error('Existing run requires --resume with identical data/settings/versions')
else:
    out.mkdir(parents=True)
    atomic_json(contract_path, contract)
progress = Progress(out / 'logs', 'training')
if (out / 'completed.json').exists():
    progress.log('Training already completed for this configuration; not rerunning')
    raise SystemExit(0)
checkpoint = None
if a.resume:
    candidates = []
    for path in out.glob('checkpoint-*'):
        if not path.name.rsplit('-',1)[-1].isdigit():
            continue
        required = ['trainer_state.json','optimizer.pt','scheduler.pt']
        has_rng = bool(list(path.glob('rng_state*.pth')))
        has_weights = bool(list(path.glob('*.safetensors'))) or bool(list(path.glob('*.bin')))
        if all((path/f).is_file() for f in required) and has_rng and has_weights:
            candidates.append(path)
    if not candidates:
        progress.log('No complete training checkpoint exists yet; restarting at step 0 with the same configuration')
    else:
        checkpoint = str(max(candidates,key=lambda x:int(x.name.rsplit('-',1)[-1])))
        progress.log(f'Resuming from {checkpoint}; work after that checkpoint may be repeated')

import torch
from huggingface_hub import snapshot_download, HfApi
from aligntune.core.backend_factory import create_sft_trainer
from aligntune.core.sft.training_args_builder import build_sft_config
from transformers import TrainerCallback, set_seed
from trl import SFTTrainer
if not torch.cuda.is_available():
    p.error('Use a Linux NVIDIA GPU environment')
set_seed(a.seed)
# Restrict this workflow to one visible GPU; do not use device_map=auto for training.
bf16 = torch.cuda.is_bf16_supported() and not a.load_in_4bit
with progress.stage('model-and-data-setup'):
    model_path = snapshot_download(a.model, revision=a.revision,
        ignore_patterns=['*.bin','*.h5','*.msgpack','*.onnx','*.gguf'])
    quantization = {}
    if a.load_in_4bit:
        quantization = {'load_in_4bit':True,'bnb_4bit_quant_type':'nf4',
                        'bnb_4bit_use_double_quant':True,'bnb_4bit_compute_dtype':'float16'}
    wrapper = create_sft_trainer(model_name=model_path, dataset_name=a.data,
        config_name='sft_messages', split='train', backend='trl', output_dir=a.out,
        num_epochs=1, max_steps=a.max_steps, batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-4, max_seq_length=1024, use_peft=True, lora_r=16, lora_alpha=32,
        lora_target_modules=['q_proj','v_proj'], lora_dropout=.05, quantization=quantization,
        precision='bf16' if bf16 else 'fp16', bf16=bf16, fp16=not bf16, device_map={'':0},
        gradient_checkpointing=True, seed=a.seed, data_seed=a.seed, eval_strategy='no',
        load_best_model_at_end=False, report_to='none', loggers=[], save_steps=a.save_steps,
        logging_steps=1, compute_rouge=False, compute_bleu=False)
    wrapper.setup_model()
    if a.load_in_4bit and not getattr(wrapper.model, 'is_loaded_in_4bit', False):
        raise RuntimeError('4-bit loading was requested but the model is not marked as 4-bit; refusing an unintended memory-heavy run')
    trainable = sum(v.numel() for v in wrapper.model.parameters() if v.requires_grad)
    total = sum(v.numel() for v in wrapper.model.parameters())
    if trainable == 0:
        raise RuntimeError('No trainable adapter parameters found')
    progress.log(f'Trainable parameters: {trainable:,} / {total:,}; loaded_4bit={getattr(wrapper.model,"is_loaded_in_4bit",False)}')
    wrapper.model.config.use_cache = False
    wrapper.setup_data()
    dataset = wrapper.train_dataset
    dataset = dataset.select_columns([name for name in ['messages','chat_template_kwargs','prompt','completion'] if name in dataset.column_names])
    config = build_sft_config(config=wrapper.config, train_dataset=dataset,
        eval_dataset=None, output_dir=a.out, save_total_limit=2)

class CheckpointLog(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        progress.log(f'TRAIN step={state.global_step}/{a.max_steps} metrics={logs}')
    def on_save(self,args,state,control,**kwargs):
        folder = out / f'checkpoint-{state.global_step}'
        progress.log(f'CHECKPOINT saved: {folder}')
        if a.backup_repo:
            try:
                api=HfApi()
                api.create_repo(a.backup_repo,repo_type='dataset',private=True,exist_ok=True)
                info=api.repo_info(a.backup_repo,repo_type='dataset')
                if not info.private:
                    raise RuntimeError('Recovery backups require a private dataset repo')
                api.upload_folder(repo_id=a.backup_repo,repo_type='dataset',folder_path=str(folder),
                    path_in_repo=f'{out.name}/{folder.name}',commit_message=f'FinSafe recovery step {state.global_step}')
                api.upload_file(repo_id=a.backup_repo,repo_type='dataset',path_or_fileobj=str(contract_path),
                    path_in_repo=f'{out.name}/finsafe_training.json')
                progress.log('DURABLE CHECKPOINT: uploaded full recovery state')
            except Exception as error:
                progress.log(f'BACKUP FAILED ({type(error).__name__}); local checkpoint remains, but is not protected from session loss')

with progress.stage('training'):
    # Upstream wrapper.train() does not forward resume_from_checkpoint. Use its
    # model/data/config and explicitly invoke the underlying TRL trainer instead.
    trainer=SFTTrainer(model=wrapper.model,args=config,train_dataset=dataset,
        processing_class=wrapper.tokenizer,peft_config=None,callbacks=[CheckpointLog()])
    # AlignTune/TRL may recast LoRA weights to BF16 during setup. AMP's FP16
    # scaler requires full-precision trainable weights; keep frozen NF4 intact.
    dtype_counts = {}
    for parameter in trainer.model.parameters():
        if parameter.requires_grad:
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
            name = str(parameter.dtype)
            dtype_counts[name] = dtype_counts.get(name, 0) + parameter.numel()
    progress.log(f'Trainable parameter dtypes before optimizer creation: {dtype_counts}')
    result=trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(a.out)
    wrapper.tokenizer.save_pretrained(a.out)
    atomic_json(out/'completed.json',{'global_step':result.global_step,'metrics':result.metrics})
progress.log('DONE; final adapter and metadata saved. Back up before ending the session.')
