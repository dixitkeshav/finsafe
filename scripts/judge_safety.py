"""Pinned aligned-model judge, condition-blind prompts, per-response recovery."""
import argparse,json,hashlib,random
from pathlib import Path
from finsafe.data import read_jsonl,write_jsonl
from finsafe.resume import sha256_file
from finsafe.progress import Progress,atomic_json
from finsafe.judge import RUBRIC,parse_label,quality_gate
p=argparse.ArgumentParser()
p.add_argument('--manifest',required=True,help='JSON list: data, predictions, out per condition')
p.add_argument('--out',required=True,help='Judge state directory')
p.add_argument('--model',default='CohereLabs/tiny-aya-global')
p.add_argument('--revision',required=True)
a=p.parse_args();out=Path(a.out);progress=Progress(out/'logs','judge')
reference=read_jsonl('data/judge-validation/reference_v1.jsonl')
manifest=json.loads(Path(a.manifest).read_text())
contract={'model':a.model,'revision':a.revision,'rubric_sha256':hashlib.sha256(RUBRIC.encode()).hexdigest(),
 'reference_sha256':sha256_file('data/judge-validation/reference_v1.jsonl'),'manifest_sha256':sha256_file(a.manifest),
 'predictions':{r['predictions']:sha256_file(r['predictions']) for r in manifest},'runner_sha256':sha256_file(__file__)}
marker=out/'contract.json'
if marker.exists() and json.loads(marker.read_text())!=contract:raise ValueError('Judge contract changed')
atomic_json(marker,contract)
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
set_seed(42)
with progress.stage('load-pinned-aligned-judge'):
 tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
 model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,dtype=torch.float16,device_map={'':0},
  quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
 model.eval()
cache=read_jsonl(out/'decisions.jsonl') if (out/'decisions.jsonl').exists() else []
lookup={r['key']:r for r in cache}
def classify(prompt,response):
 key=hashlib.sha256(json.dumps([prompt,response],ensure_ascii=False).encode()).hexdigest()
 if key in lookup:return lookup[key]
 payload=json.dumps({'user_prompt':prompt,'model_response':response},ensure_ascii=False)
 last=None
 for attempt in range(2):
  messages=[{'role':'system','content':RUBRIC},{'role':'user','content':payload+('\nReturn exactly valid JSON, with a label from the four listed labels.' if attempt else '')}]
  inputs=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
  inputs={k:v.to(model.device) for k,v in inputs.items()}
  with torch.inference_mode():
   result=model.generate(**inputs,do_sample=False,max_new_tokens=160,pad_token_id=tok.pad_token_id or tok.eos_token_id)
  text=tok.decode(result[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
  try:
   decision={**parse_label(text),'key':key,'raw_judgment':text}
   cache.append(decision);lookup[key]=decision;write_jsonl(out/'decisions.jsonl',cache)
   progress.log(f'Judge checkpoint {len(cache)} classifications')
   return decision
  except (ValueError,json.JSONDecodeError) as error:last=error
 raise ValueError('Judge failed to return a valid classification; no safety rates will be published') from last
with progress.stage('reference-quality-gate'):
 refs=[classify(r['prompt'],r['output']) for r in reference]
 gate=quality_gate(reference,refs);atomic_json(out/'quality.json',gate)
 progress.log(json.dumps(gate))
 if not gate['passed']:
  raise RuntimeError('Judge quality gate failed; assistant review required before safety metrics')
with progress.stage('condition-blind-response-labeling'):
 tasks=[]
 for item in manifest:
  data=read_jsonl(item['data']);pred=read_jsonl(item['predictions'])
  if len(pred)!=len(data) or [r['id'] for r in pred]!=[r['id'] for r in data]:raise ValueError('Incomplete/mismatched predictions')
  tasks.extend((item,row,answer) for row,answer in zip(data,pred))
 random.Random(23017).shuffle(tasks)
 decisions={}
 for item,row,answer in tasks:
  result=classify(row['prompt'],answer['output'])
  decisions[(item['out'],row['id'])]={'id':row['id'],'label':result['label'],'notes':result['rationale'],
   'output_sha256':hashlib.sha256(answer['output'].encode()).hexdigest(),
   'reviewer':f'Automated TinyAya aligned judge {a.revision}; AI reference gate; no human review'}
 for item in manifest:
  data=read_jsonl(item['data'])
  write_jsonl(item['out'],[decisions[(item['out'],r['id'])] for r in data])
 atomic_json(out/'completed.json',{'status':'complete','n':len(tasks),'quality':gate,'limitations':gate['limitation']})
