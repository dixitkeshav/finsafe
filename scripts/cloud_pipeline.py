"""Unattended versioned Kaggle experiment. No credentials are written to disk."""
import json,os,sys,time,shutil,subprocess,zipfile,importlib.metadata
from pathlib import Path
from finsafe.progress import Progress,atomic_json
from finsafe.data import read_jsonl,write_jsonl,validate
from finsafe.resume import sha256_file
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
PYTHON=sys.executable;CLI=str(Path(PYTHON).with_name('finsafe'))
MODEL='CohereLabs/tiny-aya-global';REV='af89d219b53ed9b13b8a4645f9c8028973510324'
RUN=ROOT/'runs/submission-v1';progress=Progress(RUN/'logs','pipeline');started=time.monotonic()
from huggingface_hub import HfApi
api=HfApi(token=os.environ['HF_WRITE_TOKEN']);RECOVERY='Finsafe-team/finsafe-recovery'

def command(*args):progress.run([PYTHON,'-u',*map(str,args)],cwd=ROOT)
def backup(name):
    with progress.stage('backup-'+name):
        archive=Path('/kaggle/working/finsafe-submission-recovery.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for base in ['data/processed','runs','checkpoints/finance-sft-100-v1','checkpoints/steering-v1']:
                for f in (ROOT/base).rglob('*'):
                    if f.is_file() and not any(x in f.parts for x in ['.cache','__pycache__']):z.write(f,f.relative_to(ROOT))
        api.upload_file(path_or_fileobj=str(archive),path_in_repo='latest/submission-recovery.zip',repo_id=RECOVERY,repo_type='dataset')
        api.upload_file(path_or_fileobj=json.dumps({'stage':name,'sha256':sha256_file(archive),'seconds':time.monotonic()-started}).encode(),path_in_repo='latest/status.json',repo_id=RECOVERY,repo_type='dataset')

def generate(data,condition,*,adapter=None,steering=None,alpha=0,cap=64):
    dest=RUN/condition/('finance' if cap==64 else 'safety')/'predictions.jsonl'
    args=['scripts/generate.py','--model',MODEL,'--revision',REV,'--data',data,'--condition',condition,'--out',dest,'--max-new-tokens',cap,'--load-in-4bit','--resume']
    if adapter:args+=['--adapter',adapter]
    if steering:args+=['--steering-artifact',steering,'--alpha',alpha]
    command(*args)
    return dest

try:
    with progress.stage('01-validate-inputs-and-runtime'):
        subprocess.run(['nvidia-smi'],check=True)
        progress.run([CLI,'doctor','--gpu'],cwd=ROOT)
        freeze='\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n'
        (RUN/'environment.txt').write_text(freeze)
        train=ROOT/'data/processed/tatqa-tokenized/curated'
        if sha256_file(train/'sft_messages.jsonl')!='207956f3c3e82a9a2ad55a3fef292be55cd64f856f13635218f85ba571dff75d':raise ValueError('Curated training hash differs from verified run')
        finance=ROOT/'data/processed/tatqa-tokenized/validation.jsonl'
        safety=ROOT/'data/processed/safety-pilot-v1/validation.jsonl'
        if len(read_jsonl(finance))!=200 or len(read_jsonl(safety))!=36:raise ValueError('Unexpected evaluation size')
        validate(read_jsonl(ROOT/'data/benchmarks/finance_safety_pilot_v1.jsonl'))
        api.create_repo(RECOVERY,repo_type='dataset',private=True,exist_ok=True)
        if not api.repo_info(RECOVERY,repo_type='dataset').private:raise ValueError('Recovery repo must be private')
        atomic_json(RUN/'protocol.json',{'model':MODEL,'revision':REV,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'training_steps':100,'seed':42,'finance_n':200,'safety_n':36,'finance_cap':64,'safety_cap':768,
          'conditions':['aligned','sft100','steer5','steer20'],'steering_layer':18,'alphas':[5,20],
          'scope':'preliminary validation comparison; reserved test untouched; no selection on test',
          'judge':'pinned aligned TinyAya; AI-reviewed sanity gate; no independent human review',
          'thesis':'Test whether finance adaptation changes safety; erosion and successful recovery are not assumed.'})
    with progress.stage('02-curate-calibration'):
        command('scripts/prepare_steering_calibration.py')
        calibration=ROOT/'data/processed/steering-calibration-v2/calibration.jsonl'
        calgroups={r['group_id'] for r in read_jsonl(calibration)}
        if calgroups & {r['group_id'] for r in read_jsonl(safety)}:raise ValueError('Calibration/evaluation leakage')
    with progress.stage('03-safetune-gpu-preflight'):
        pre=ROOT/'checkpoints/steering-preflight-v1'
        command('scripts/calibrate_steering.py','--revision',REV,'--data',calibration,'--out',pre)
        write_jsonl(RUN/'preflight.jsonl',read_jsonl(safety)[:2])
        generate(RUN/'preflight.jsonl','preflight-steer',steering=pre,alpha=5,cap=32)
    backup('preflight')
    conditions=[]
    with progress.stage('04-aligned-validation'):
        f=generate(finance,'aligned');s=generate(safety,'aligned',cap=768)
        conditions.append({'condition':'aligned','finance':str(f),'safety':str(s)})
    backup('aligned')
    adapter=ROOT/'checkpoints/finance-sft-100-v1'
    with progress.stage('05-aligntune-finance-sft'):
        command('scripts/train_sft.py','--revision',REV,'--data',train,'--out',adapter,'--max-steps',100,'--save-steps',10,'--load-in-4bit','--resume','--backup-repo',RECOVERY)
        completed=json.loads((adapter/'completed.json').read_text())
        if completed['global_step']!=100:raise ValueError('Incomplete training')
    backup('trained')
    with progress.stage('06-sft-validation'):
        f=generate(finance,'sft100',adapter=adapter);s=generate(safety,'sft100',adapter=adapter,cap=768)
        conditions.append({'condition':'sft100','finance':str(f),'safety':str(s)})
    backup('sft-evaluated')
    steering=ROOT/'checkpoints/steering-v1'
    with progress.stage('07-fit-sft-refusal-direction'):
        command('scripts/calibrate_steering.py','--revision',REV,'--adapter',adapter,'--data',calibration,'--out',steering)
    for alpha in [5,20]:
        with progress.stage(f'08-steering-{alpha}-validation'):
            name=f'steer{alpha}'
            f=generate(finance,name,adapter=adapter,steering=steering,alpha=alpha)
            s=generate(safety,name,adapter=adapter,steering=steering,alpha=alpha,cap=768)
            conditions.append({'condition':name,'finance':str(f),'safety':str(s),'alpha':alpha})
        backup(name)
    atomic_json(RUN/'conditions.json',conditions)
    with progress.stage('09-finance-metrics-auditkit'):
        for item in conditions:
            parent=Path(item['finance']).parent
            progress.run([CLI,'summarize','--data',finance,'--predictions',item['finance'],'--out',parent/'report.json'],cwd=ROOT)
            progress.run([CLI,'audit-replay','--data',finance,'--predictions',item['finance'],'--out',parent/'auditkit.json'],cwd=ROOT)
    judge_manifest=[{'data':str(safety),'predictions':item['safety'],'out':str(Path(item['safety']).parent/'annotations.jsonl')} for item in conditions]
    atomic_json(RUN/'judge-manifest.json',judge_manifest)
    judge_ok=False
    try:
        with progress.stage('10-ai-response-review'):
            command('scripts/judge_safety.py','--manifest',RUN/'judge-manifest.json','--out',RUN/'judge','--revision',REV)
        judge_ok=True
    except subprocess.CalledProcessError:
        progress.log('Judge failed quality/schema gate. Publishing finance and raw safety outputs with safety rates unavailable; assistant review remains required.')
    if judge_ok:
        with progress.stage('11-safety-metrics-auditkit'):
            for item in conditions:
                parent=Path(item['safety']).parent
                for sub,outname in [('summarize','report.json'),('audit-safety-replay','auditkit.json')]:
                    progress.run([CLI,sub,'--data',safety,'--predictions',item['safety'],'--annotations',parent/'annotations.jsonl','--out',parent/outname],cwd=ROOT)
    atomic_json(RUN/'compute.json',{'wall_seconds':time.monotonic()-started,'gpu':'one T4 used; second allocated T4 idle','training_steps':100,'price_usd':None,'pricing_note':'Kaggle free quota; no paid-compute ROI claim'})
    backup('evaluated')
    with progress.stage('12-publish-huggingface'):
        command('scripts/publish_submission.py','--run',RUN)
    atomic_json(RUN/'completed.json',{'status':'complete' if judge_ok else 'partial-safety-labels-pending','seconds':time.monotonic()-started,'judge_passed':judge_ok})
    backup('published')
    for name in ['completed.json','compute.json','public-links.json']:
        api.upload_file(path_or_fileobj=str(RUN/name),path_in_repo=name,repo_id='Finsafe-team/finsafe-runs',repo_type='dataset')
    progress.log('PIPELINE FINISHED. Inspect public links and submit the organizer form. Form submission is not automated by this GPU job.')
except BaseException as error:
    atomic_json(RUN/'failed.json',{'error_type':type(error).__name__,'seconds':time.monotonic()-started})
    progress.log(f'PIPELINE FAILED ({type(error).__name__}); attempting private recovery upload')
    try:backup('failure')
    except Exception as backup_error:progress.log('Failure backup also failed: '+type(backup_error).__name__)
    raise
