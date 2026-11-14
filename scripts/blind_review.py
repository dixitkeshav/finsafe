"""Create shuffled condition-hidden response packets and restore validated labels."""
import argparse, hashlib, json, random
from pathlib import Path
from finsafe.data import read_jsonl, validate, write_jsonl
from finsafe.progress import atomic_json
from finsafe.scoring import LABELS

def digest(text):return hashlib.sha256(text.encode()).hexdigest()

def create(data,prediction_paths,out,seed=42):
    rows=validate(read_jsonl(data));lookup={r['id']:r for r in rows}
    if any(r['kind']=='finance' for r in rows):raise ValueError('Use safety-only evaluation data')
    expected_hash=hashlib.sha256(Path(data).read_bytes()).hexdigest()
    destination=Path(out)
    if destination.exists():raise ValueError('Use a fresh review directory; do not overwrite annotation work')
    packet=[];mapping={};conditions=set()
    for path in prediction_paths:
        predictions=read_jsonl(path)
        if len(predictions)!=len(rows) or {p['id'] for p in predictions}!=set(lookup):raise ValueError('Prediction IDs mismatch')
        names={p.get('condition') for p in predictions}
        if len(names)!=1 or not next(iter(names)):raise ValueError('One named condition required per file')
        condition=next(iter(names))
        if condition in conditions:raise ValueError('Duplicate condition')
        conditions.add(condition)
        for p in predictions:
            if p.get('dataset_sha256')!=expected_hash or not isinstance(p.get('output'),str):raise ValueError('Prediction source/output mismatch')
            response_hash=digest(p['output'])
            key=digest(json.dumps([condition,p['id'],response_hash],ensure_ascii=False))[:24]
            if key in mapping:raise ValueError('Review ID collision')
            mapping[key]={'condition':condition,'id':p['id'],'output_sha256':response_hash,
                          'prompt_sha256':digest(lookup[p['id']]['prompt'])}
            packet.append({'id':key,'prompt':lookup[p['id']]['prompt'],'kind':lookup[p['id']]['kind'],
                'output':p['output'],'output_sha256':response_hash,
                'output_cap_hit':p.get('max_new_tokens') is not None and p.get('output_tokens',0)>=p['max_new_tokens'],
                'label':'UNREVIEWED','reviewer':'','notes':''})
    if len(conditions)<2:raise ValueError('At least two model conditions are required')
    random.Random(seed).shuffle(packet)
    destination.mkdir(parents=True)
    write_jsonl(destination/'review.jsonl',packet)
    atomic_json(destination/'coordinator-only-mapping.json',{'dataset_sha256':expected_hash,'seed':seed,'items':mapping,
        'packet_sha256':hashlib.sha256((destination/'review.jsonl').read_bytes()).hexdigest()})
    print(f'{len(packet)} shuffled review rows created. Give reviewers only review.jsonl; keep the mapping separate.')

def restore(packet,mapping_path,out):
    rows=read_jsonl(packet);mapping=json.loads(Path(mapping_path).read_text())['items']
    if len(rows)!=len(mapping) or {r['id'] for r in rows}!=set(mapping):raise ValueError('Review IDs missing, duplicated or extra')
    labels={}
    for row in rows:
        item=mapping[row['id']]
        if digest(row['output'])!=item['output_sha256'] or row.get('output_sha256')!=item['output_sha256']:raise ValueError('Reviewed response was changed')
        if digest(row['prompt'])!=item['prompt_sha256']:raise ValueError('Reviewed prompt was changed')
        if row.get('label') not in LABELS or not row.get('reviewer','').strip():raise ValueError('All responses need reviewed labels and reviewer identity')
        labels.setdefault(item['condition'],[]).append({'id':item['id'],'label':row['label'],'reviewer':row['reviewer'],
             'notes':row.get('notes',''),'output_sha256':item['output_sha256']})
    destination=Path(out)
    if destination.exists():raise ValueError('Use a fresh annotation directory')
    destination.mkdir(parents=True)
    index={}
    for condition,annotations in labels.items():
        # Hash filenames so condition names cannot become arbitrary paths.
        filename=digest(condition)[:16]+'.jsonl'
        write_jsonl(destination/filename,annotations);index[condition]=filename
    atomic_json(destination/'index.json',index)
    print('Validated per-condition annotation files exported. Use index.json to select each file for scoring.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('create');p.add_argument('--data',required=True);p.add_argument('--predictions',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42)
    p=sub.add_parser('restore');p.add_argument('--packet',required=True);p.add_argument('--mapping',required=True);p.add_argument('--out',required=True)
    a=parser.parse_args()
    if a.command=='create':create(a.data,a.predictions,a.out,a.seed)
    else:restore(a.packet,a.mapping,a.out)
