"""Recompute published metrics anonymously on CPU; no training/inference claimed."""
import argparse,json,shutil,contextlib
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from finsafe.progress import Progress,atomic_json
from finsafe.scoring import summarize
from finsafe.cli import audit_replay,audit_safety_replay
p=argparse.ArgumentParser();p.add_argument('--out',default='runs/reproduced-public-v1');a=p.parse_args()
out=Path(a.out);progress=Progress(out/'logs','public-reproduction');api=HfApi(token=False)
repos={name:f'Finsafe-team/finsafe-{name}' for name in ['runs','training','benchmark']}
revisions={name:api.repo_info(repo,repo_type='dataset').sha for name,repo in repos.items()}
atomic_json(out/'source-revisions.json',revisions)
def get(name,file,destination):
 source=hf_hub_download(repos[name],file,repo_type='dataset',revision=revisions[name],token=False)
 destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
 return destination
with progress.stage('download-public-data-and-recompute'):
 finance=get('training','source-prepared/validation.jsonl',out/'data/finance.jsonl')
 safety=get('benchmark','safety-splits/validation.jsonl',out/'data/safety.jsonl')
 expected=json.loads(Path(hf_hub_download(repos['runs'],'results.json',repo_type='dataset',revision=revisions['runs'],token=False)).read_text())
 checks=[]
 for condition in ['aligned','sft100','steer5','steer20']:
  reference=next(r for r in expected['conditions'] if r['condition']==condition)
  for kind,data in [('finance',finance),('safety',safety)]:
   folder=out/condition/kind
   pred=get('runs',f'{condition}/{kind}/predictions.jsonl',folder/'predictions.jsonl')
   labels=get('runs',f'{condition}/{kind}/annotations.jsonl',folder/'annotations.jsonl') if kind=='safety' else None
   report=summarize(data,pred,labels,folder/'report.json')
   for metric,result in report['metrics'].items():
    if result['value']!=reference[kind]['metrics'][metric]['value'] or result['n']!=reference[kind]['metrics'][metric]['n']:raise ValueError(f'Reproduction mismatch: {condition}/{kind}/{metric}')
   with (folder/'auditkit-replay.log').open('w') as log,contextlib.redirect_stdout(log):
    if kind=='safety':audit_safety_replay(data,pred,labels,folder/'auditkit.json')
    else:audit_replay(data,pred,folder/'auditkit.json')
   checks.append(f'{condition}/{kind}')
   progress.log('MATCHED published metrics: '+checks[-1])
 atomic_json(out/'completed.json',{'status':'all published metric values and denominators matched','checks':checks,'source_revisions':revisions,'scope':'CPU replay of saved outputs and AI annotations; no new inference/training or independent label validation'})
progress.log('DONE: all eight condition/task reports reproduced anonymously')
