"""Export prompt/response review rows without hiding missing human annotations."""
import argparse,hashlib,json,random
from pathlib import Path
from finsafe.data import read_jsonl,validate,write_jsonl
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--predictions',required=True);p.add_argument('--out',required=True);a=p.parse_args()
rows=validate(read_jsonl(a.data));pred=read_jsonl(a.predictions);lookup={r['id']:r for r in pred}
if len(lookup)!=len(pred) or set(lookup)!={r['id'] for r in rows}:raise ValueError('Prediction IDs mismatch')
source_hash=hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
if any(r.get('dataset_sha256')!=source_hash for r in pred):raise ValueError('Prediction source mismatch')
packet=[{'id':r['id'],'prompt':r['prompt'],'kind':r['kind'],'output':lookup[r['id']]['output'],
 'output_sha256':hashlib.sha256(lookup[r['id']]['output'].encode()).hexdigest(),
 'output_cap_hit':lookup[r['id']].get('output_tokens',0)>=lookup[r['id']].get('max_new_tokens',256),
 'label':'UNREVIEWED','reviewer':'','notes':''} for r in rows]
random.Random(42).shuffle(packet);write_jsonl(a.out,packet)
print(f'{len(packet)} review rows exported; safety rates remain unmeasured until reviewed labels exist.')
