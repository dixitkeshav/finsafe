"""Freeze the project-authored pilot, validate splits and export reviewable prompts."""
import argparse, hashlib, json
from pathlib import Path
from finsafe.data import read_jsonl, validate, write_jsonl, canonical
from finsafe.progress import atomic_json
p=argparse.ArgumentParser();p.add_argument('--out',default='data/processed/safety-pilot-v1');a=p.parse_args()
source=Path('data/benchmarks/finance_safety_pilot_v1.jsonl')
rows=validate(read_jsonl(source));out=Path(a.out)
manifest={'version':'finance-safety-pilot-v1','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'counts':{s:sum(r['split']==s for r in rows) for s in ['calibration','validation','test']},
 'groups':{s:len({r['group_id'] for r in rows if r['split']==s}) for s in ['calibration','validation','test']},
 'attack_budget':'one fixed fictional-roleplay variant per harmful behavior; per-prompt ASR',
 'limitations':['AI-assisted project-authored pilot; independent prompt review pending','English only; one attack family','Small behavior count; no broad robustness claim','Calibration/validation/test behavior objectives are disjoint but topic distributions differ'],
 'checks':['schema','unique IDs','exact normalized prompts','group-disjoint splits','one direct + one attack + one benign per behavior']}
for group in {r['group_id'] for r in rows}:
 kinds=[r['kind'] for r in rows if r['group_id']==group]
 if sorted(kinds)!=sorted(['harmful_direct','harmful_attack','benign_sensitive']):raise ValueError(group)
if len({canonical(r['prompt']) for r in rows})!=len(rows):raise ValueError('Duplicate prompts')
if out.exists() and (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('Existing frozen pilot changed; use a new version')
out.mkdir(parents=True,exist_ok=True)
for split in manifest['counts']:write_jsonl(out/f'{split}.jsonl',[r for r in rows if r['split']==split])
atomic_json(out/'manifest.json',manifest);print(json.dumps(manifest,indent=2))
