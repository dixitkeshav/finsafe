"""Validate frozen assistant-reviewed annotations and rerun real AuditKIT on CPU."""
import argparse,json,shutil,math
from pathlib import Path
from finsafe.progress import Progress,atomic_json
from finsafe.resume import sha256_file
from finsafe.scoring import summarize
from finsafe.cli import audit_safety_replay
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];run=Path(a.run).resolve();package=root/'data/reviewed-validation-v1'
manifest=json.loads((package/'manifest.json').read_text());data=root/'data/processed/safety-pilot-v1/validation.jsonl'
if sha256_file(data)!=manifest['dataset_sha256']:raise ValueError('Review dataset mismatch')
progress=Progress(run/'review-finalization/logs','review-finalization')
with progress.stage('validate-and-score-all-reviewed-conditions'):
 for name,expected in manifest['conditions'].items():
  folder=run/name/'safety';pred=folder/'predictions.jsonl';labels=package/(name+'.jsonl')
  if sha256_file(pred)!=expected['predictions_sha256'] or sha256_file(labels)!=expected['annotations_sha256']:raise ValueError('Reviewed response/annotation hashes changed')
  shutil.copy2(labels,folder/'annotations.jsonl')
  report=summarize(data,pred,folder/'annotations.jsonl',folder/'report.json')
  for key in ['direct_harmful_compliance','attack_success_rate','over_refusal']:
   m=report['metrics'][key];n=m['n'];v=m['value'];z=1.959963984540054;den=1+z*z/n
   center=(v+z*z/(2*n))/den;radius=z*math.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
   m['events']=round(v*n);m['ci95_wilson_independent_groups']=[max(0,center-radius),min(1,center+radius)]
  report['review']={'type':'Codex assistant review','independent_human':False,'n':36,'condition_hidden':True,'rubric':'frozen rubric-v1','automated_judge_rejected':True,'uncertainty_note':'Bootstrap can collapse at boundaries. Wilson intervals treat twelve distinct groups per kind as independent; small pilot and one AI reviewer do not establish general robustness.'}
  atomic_json(folder/'report.json',report)
  audit_safety_replay(data,pred,folder/'annotations.jsonl',folder/'auditkit.json')
 atomic_json(run/'review-finalization/manifest.json',manifest)
 protocol=json.loads((run/'protocol.json').read_text());protocol['final_label_source']='Codex assistant review of all 144 condition-hidden responses; no independent human review; original automated judge rejected'
 atomic_json(run/'protocol.json',protocol)
 progress.log('All reviewed safety reports and AuditKIT replays complete; original failed judge gate retained')
