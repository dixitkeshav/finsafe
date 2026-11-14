"""Fit one predeclared SafeTune refusal direction using calibration prompts only."""
import argparse,json,importlib.metadata
from pathlib import Path
from finsafe.data import read_jsonl
from finsafe.progress import Progress,atomic_json
from finsafe.resume import sha256_file
from finsafe.steering import adapter_sha
p=argparse.ArgumentParser()
p.add_argument('--model',default='CohereLabs/tiny-aya-global')
p.add_argument('--revision',required=True)
p.add_argument('--adapter')
p.add_argument('--data',required=True)
p.add_argument('--out',required=True)
p.add_argument('--layer',type=int,default=18)
a=p.parse_args()
out=Path(a.out)
contract={'model':a.model,'revision':a.revision,'adapter_sha256':adapter_sha(a.adapter),
 'calibration_sha256':sha256_file(a.data),'layer':a.layer,'safetune':importlib.metadata.version('safetune')}
if (out/'config.json').exists():
 cfg=json.loads((out/'config.json').read_text())
 if any(cfg.get(k)!=v for k,v in contract.items()):raise ValueError('Calibration contract changed')
 if cfg['direction_sha256']!=sha256_file(out/'direction.safetensors'):raise ValueError('Direction corrupted')
 print('Calibration already complete',flush=True);raise SystemExit(0)
rows=read_jsonl(a.data)
if any(r['split']!='calibration' for r in rows):raise ValueError('Calibration split only')
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig,extract_refusal_direction
from safetensors.torch import save_file
import torch
progress=Progress(out/'logs','calibration')
set_seed(42)
with progress.stage('load-model'):
 tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
 model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,dtype=torch.float16,device_map={'':0},
  quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
 if a.adapter:
  from peft import PeftModel
  model=PeftModel.from_pretrained(model,a.adapter)
 model.eval()
with progress.stage('fit-predeclared-layer'):
 harmful=[r['prompt'] for r in rows if r['kind']=='harmful_direct']
 benign=[r['prompt'] for r in rows if r['kind']=='benign_sensitive']
 if min(len(harmful),len(benign))<4:raise ValueError('Insufficient calibration contrast')
 direction,layer,_=extract_refusal_direction(model,tok,harmful,benign,
  RefusalDirectionConfig(target_layers=[a.layer],pick_layer=a.layer,select_directions=False))
 if not torch.isfinite(direction).all() or float(direction.norm())<1e-8:raise ValueError('Invalid direction')
 out.mkdir(parents=True,exist_ok=True)
 save_file({'direction':direction.float().cpu()},str(out/'direction.safetensors'))
 atomic_json(out/'config.json',{**contract,'direction_sha256':sha256_file(out/'direction.safetensors'),
  'harmful_n':len(harmful),'benign_n':len(benign),'layer':layer,'selection':'fixed in advance; no validation/test selection'})
 progress.log('Frozen inference steering artifact saved; this is not a standalone model')
