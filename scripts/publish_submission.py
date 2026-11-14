"""Publish only explicit experiment artifacts and a measured, honest report."""
import argparse,json,os,shutil,html,time
from pathlib import Path
from huggingface_hub import HfApi
from finsafe.data import read_jsonl
from finsafe.progress import Progress,atomic_json
from finsafe.resume import sha256_file
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];run=Path(a.run).resolve();stage=run/'publication';stage.mkdir(parents=True,exist_ok=True)
progress=Progress(stage/'logs','publication');api=HfApi(token=os.environ['HF_WRITE_TOKEN']);ns='Finsafe-team'
conditions=json.loads((run/'conditions.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
judge=json.loads((run/'judge/quality.json').read_text()) if (run/'judge/quality.json').exists() else {'passed':False,'limitation':'Judge did not complete; safety rates unavailable.'}
results=[]
for item in conditions:
 f=json.loads((Path(item['finance']).parent/'report.json').read_text())
 sp=Path(item['safety']).parent/'report.json';s=json.loads(sp.read_text()) if sp.exists() else None
 results.append({'condition':item['condition'],'finance':f,'safety':s,'alpha':item.get('alpha'),
  'finance_predictions_sha256':sha256_file(item['finance']),'safety_predictions_sha256':sha256_file(item['safety'])})
if len({r['finance']['dataset_sha256'] for r in results})!=1:raise ValueError('Finance datasets differ')
safe=[r['safety']['dataset_sha256'] for r in results if r['safety']]
if safe and (len(safe)!=len(results) or len(set(safe))!=1):raise ValueError('Safety datasets differ/incomplete')
def metric(r,name):
 report=r['finance'] if name.startswith('finance') else r['safety']
 return report['metrics'][name]['value'] if report else None
def percent(x):return 'Unavailable' if x is None else f'{100*x:.1f}%'
headers=['Condition','Finance accuracy ↑','Direct compliance ↓','Fixed-attack ASR ↓','Benign refusal ↓','Safety invalids ↓','Safety cap hits ↓']
table=[headers]
for r in results:
 table.append([r['condition']]+[percent(metric(r,k)) for k in ['finance_accuracy','direct_harmful_compliance','attack_success_rate','over_refusal','invalid_output_rate','output_cap_hit_rate']])
markdown='| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in table[1:])
sft=next(r for r in results if r['condition']=='sft100');aligned=results[0]
finance_gain=100*(metric(sft,'finance_accuracy')-metric(aligned,'finance_accuracy'))
if safe:
 drift=100*(metric(sft,'attack_success_rate')-metric(aligned,'attack_success_rate'))
 finding=f'Finance accuracy changed by {finance_gain:+.1f} percentage points after SFT. Fixed-attack ASR changed by {drift:+.1f} points (positive means worse safety).'
 if drift<=0:finding+=' This run does not establish the proposed safety-erosion thesis.'
else:finding=f'Finance accuracy changed by {finance_gain:+.1f} percentage points after SFT. Safety comparisons remain unscored because the AI judge did not pass/complete its sanity gate.'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,5))
if safe:
 base_acc=metric(sft,'finance_accuracy');base_asr=metric(sft,'attack_success_rate')
 for r in results:
  x=100*metric(r,'finance_accuracy')/base_acc if base_acc else 100*metric(r,'finance_accuracy')
  y=100*(base_asr-metric(r,'attack_success_rate'))
  ax.scatter(x,y,s=65);ax.annotate(r['condition'],(x,y),xytext=(5,7),textcoords='offset points')
 ax.set_xlabel('Finance accuracy retained vs SFT (%)' if base_acc else 'Finance accuracy (%)')
 ax.set_ylabel('Fixed-attack ASR reduction vs SFT (percentage points)')
 ax.axhline(0,color='gray',lw=.7);ax.set_title('Measured validation trade-off; preliminary AI labels')
else:
 ax.bar([r['condition'] for r in results],[100*metric(r,'finance_accuracy') for r in results])
 ax.set_ylabel('Finance accuracy (%)');ax.set_title('Finance comparison; safety trade-off unavailable')
ax.grid(alpha=.2);fig.tight_layout();fig.savefig(stage/'tradeoff.png',dpi=180);plt.close(fig)
atomic_json(stage/'results.json',{'conditions':results,'protocol':protocol,'judge':judge,'finding':finding})
upstream=api.model_info(protocol['model'],revision=protocol['revision'],token=os.environ.get('HF_TOKEN'))
license_id=(upstream.card_data.to_dict().get('license') if upstream.card_data else None) or 'other'
license_header=f'license: {license_id}\n'
if license_id=='other':license_header+='license_name: upstream-cohere-model-terms\nlicense_link: https://huggingface.co/CohereLabs/tiny-aya-global\n'
article=f'''# FinSafe: measuring finance adaptation and refusal steering

**Team SAK_GURU** · @KeshavDixit11 · @Aryangupta1222 · @Shauryasharmaiitb

**Track 2b, inference-time steering (problem 8).** CuratorKIT → SafeTune → AuditKIT is the track pipeline; AlignTune supplies the finance-adaptation control. This submission claims only the methods actually executed.

## Abstract
FinSafe tests whether specializing a small aligned model for grounded financial question answering changes its willingness to assist financial wrongdoing. We train a 100-step QLoRA adapter on 954 benign, curated TAT-QA examples, then compare the aligned model, finance adapter, and two fixed-strength SafeTune refusal-direction interventions. The same 200 finance validation questions and 36 project-authored safety validation prompts are used for every condition. {finding} Safety labels, when available, come from a frozen aligned-model judge validated on 16 AI-reviewed sanity examples; they are preliminary and have no independent human review. The contribution is a reproducible domain-specific measurement and intervention pipeline, including negative results, rather than an assumed safety recovery.

## Results
{markdown}

![Measured trade-off](tradeoff.png)

Finance uses 200 questions. Direct compliance, fixed-attack ASR, and benign refusal each use 12 prompts. Safety invalid and cap-hit rates each use 36 responses. Group bootstrap 95% intervals and exact raw outputs/configuration hashes are in results.json and the runs dataset. Small denominators imply wide uncertainty; no significance claim is made. Fixed roleplay attack success is not an adaptive jailbreak robustness score. Invalids remain in denominators and must be read alongside safety rates.

## Method and actual toolkit usage
CuratorKIT 0.1.15 cleaned and MinHash-deduplicated 2,500 finance rows: 954 retained, two schema rejects, and 1,544 near duplicates removed. It separately exact-deduplicates/cleans 24 safety contrast classification records before direction extraction; safety prompts are never included in finance SFT. AlignTune 0.1.35 performs model/data/config setup with its TRL backend; the explicit TRL 1.7.1 loop enables full-state recovery. SFT uses seed 42, 100 steps, sequence length 1024, batch 1, accumulation 8, learning rate 1e-4, rank-16 q_proj/v_proj LoRA, alpha 32, dropout 0.05, NF4 loading and FP16 compute with FP32 trainable parameters.

SafeTune 0.1.18 extracts harmful-minus-benign residual directions from 12 calibration pairs. Layer 18 and strengths 5 and 20 were fixed before this run. Hooks add the normalized direction at that single decoder layer during inference. Every condition is reported; no test-set tuning or claim of independently repaired weights is made. A positive strength is intended to encourage refusal but success is measured, not assumed. AuditKIT 1.1.11 evaluates finance value-plus-scale correctness and replays response labels with separate denominators; it is not itself the safety judge. Real generation timing is distinct from fast replay timing.

## Model and data
Base model: `{protocol['model']}` at `{protocol['revision']}`. Its upstream terms ({license_id}) continue to apply to dependent artifacts; our code does not relicense upstream weights. Finance is a deterministic numeric subset of [TAT-QA](https://nextplusplus.github.io/TAT-QA/), Fengbin Zhu et al., ACL 2021, CC-BY-4.0, source revision 870accc41953dcde885aabeb963d94aabdc0fbc3. Original train/dev/test_gold supply 2,500/200/200 examples, grouped by normalized table with exact overlap checks and evaluation priority. Full company/report and semantic independence are not established. No synthetic gold finance answers are used. All 2,900 examples fit the full 1,024-token chat budget; only training is curated. Validation is repeatedly used for development; final tests stay reserved.

The MIT-licensed AI-assisted finance safety pilot has 84 prompts: four calibration groups, 12 validation groups and 12 reserved-test groups, each with direct harmful, fixed roleplay attack and benign prevention variants. Eight further calibration-only pairs expand the direction contrast to 12 pairs. Objectives include fraud, market manipulation, confidential-information misuse and concealment. This is a small original pilot, not an externally validated finance safety benchmark. All content is English; there is no multilingual result.

## Scoring and judge limitations
Finance requires a strict JSON numeric answer within absolute tolerance 0.01 and matching scale. This is our numeric-subset metric, not the complete official TAT-QA leaderboard score. Formatting errors are separately reported. Greedy generation uses 64 new tokens for finance and 768 for safety, identical across conditions, with upstream stopping configuration unchanged.

The response classifier is the original aligned Tiny Aya Global at the pinned revision, distinct from adapted evaluated weights but from the same model family. Prompts supplied to the judge hide condition names. A frozen rubric prioritizes actionable harmful assistance over disclaimers and distinguishes refusal, safe answers and invalid outputs. The judge quality check is: `{json.dumps(judge,ensure_ascii=False)}`. Reference examples include prior validation responses and synthetic invalid fixtures, labeled by Codex assistant review. This is not independent validation, human review, or a reliable estimate of judge error on all outputs. If the gate fails, safety rates are unavailable and no safety-recovery conclusion is supported. Readable truncated responses are judged only on visible content; cap rates are disclosed. Residual long-tail harms and judge bias remain unresolved.

## Compute and recovery
One Kaggle T4 is used; a second allocated T4 is idle. Training saves optimizer, scheduler, RNG, scaler and adapter state every ten steps, with private HF recovery uploads. Generation saves each response atomically and validates configuration/hash contracts before resume. Recovery repeats work since the last checkpoint; it does not resume the exact interrupted instant. Timing and dependency freeze are included in the runs dataset. Kaggle free quota was used; no unsupported paid serving ROI or API-equivalence claim is made.

## Reproduction and artifacts
[Source and notebooks](https://github.com/dixitkeshav/finsafe). Run the pinned `06_kaggle_submission_pipeline.ipynb` with Kaggle GPU/Internet and HF read/write secrets. The static [HF Space](https://huggingface.co/spaces/Finsafe-team/finsafe) shows measured metrics without hosting an unrestricted assistant. Adapter: [finsafe-finance-adapter](https://huggingface.co/Finsafe-team/finsafe-finance-adapter). Steering: [finsafe-safety-steering](https://huggingface.co/Finsafe-team/finsafe-safety-steering). [Runs](https://huggingface.co/datasets/Finsafe-team/finsafe-runs), [benchmark](https://huggingface.co/datasets/Finsafe-team/finsafe-benchmark), [training](https://huggingface.co/datasets/Finsafe-team/finsafe-training).

## Contributions and extension
Team accounts are tagged above. Code and preparation were developed during the hackathon with AI assistance; existing libraries, the pretrained Cohere model and TAT-QA are credited. Teammate-specific contributions are not invented. With additional credits, run multiple seeds, an independent stronger judge plus human adjudication, adaptive attacks, multilingual evaluation and held-out company splits. These are proposed extensions, not completed results. This research artifact is not a production financial adviser or a claim of compliance with financial regulation.
'''
(stage/'ARTICLE.md').write_text(article)

def publish(repo,typ,folder):
    with progress.stage('upload-'+repo):
        api.create_repo(f'{ns}/{repo}',repo_type=typ,private=False,exist_ok=True,**({'space_sdk':'static'} if typ=='space' else {}))
        api.upload_folder(repo_id=f'{ns}/{repo}',repo_type=typ,folder_path=str(folder),commit_message='Publish measured FinSafe preliminary validation experiment')
        info=api.repo_info(f'{ns}/{repo}',repo_type=typ)
        if info.private:raise ValueError('Submission repository is unexpectedly private')

adapter=stage/'adapter';adapter.mkdir(exist_ok=True)
for name in ['adapter_model.safetensors','adapter_config.json','finsafe_training.json','completed.json']:
 shutil.copy2(root/'checkpoints/finance-sft-100-v1'/name,adapter/name)
(adapter/'README.md').write_text(f'''---
{license_header}base_model: CohereLabs/tiny-aya-global
library_name: peft
tags: [finance, finsafe, lexsi-hackathon]
---
# FinSafe finance QLoRA adapter
100-step benign finance SFT. Base revision `{protocol['revision']}`. Load this adapter with the exact base using PEFT and the provided environment, NF4/FP16. See [full measured report](https://huggingface.co/datasets/Finsafe-team/finsafe-runs/blob/main/ARTICLE.md). Not a standalone model, validated financial adviser or safety-hardened model. Upstream license and access terms apply.

{finding}
''')
publish('finsafe-finance-adapter','model',adapter)
steer=stage/'steering';steer.mkdir(exist_ok=True)
for name in ['direction.safetensors','config.json']:shutil.copy2(root/'checkpoints/steering-v1'/name,steer/name)
(steer/'README.md').write_text(f'''---
{license_header}base_model: Finsafe-team/finsafe-finance-adapter
tags: [safetune, steering, finsafe]
---
# FinSafe inference steering vector
This is an intervention artifact, not standalone repaired model weights. Requires the exact finance adapter, original base revision and SafeTune 0.1.18. The config binds the direction to the adapter weight checksum. Download this folder and pass `--steering-artifact LOCAL_FOLDER --alpha 5` (or 20) to scripts/generate.py together with the finance adapter. The code installs SafeTune RefusalDirectionModel(mode="steer", layers=[18]); it removes hooks on exit. Both strengths are measured, with no claim that a default is proven safe. See full report for utility and over-refusal costs.

{finding}
''')
publish('finsafe-safety-steering','model',steer)
training=stage/'training';training.mkdir(exist_ok=True)
for src,dest in [('data/processed/tatqa-tokenized/curated','curated'),('data/processed/tatqa-v1','source-prepared')]:
 if (root/src).exists():shutil.copytree(root/src,training/dest,dirs_exist_ok=True)
(training/'README.md').write_text('''---
license: cc-by-4.0
---
# FinSafe TAT-QA numeric finance training data
Derived from TAT-QA, Fengbin Zhu et al., ACL 2021, CC-BY-4.0. Source: https://nextplusplus.github.io/TAT-QA/ and https://github.com/NExTplusplus/TAT-QA at 870accc41953dcde885aabeb963d94aabdc0fbc3. Preserve this attribution. 954 benign training examples retained by CuratorKIT from 2,500 sampled examples. This is a numeric subset, not a replacement for the complete dataset. Full methodology is in the FinSafe runs article. Source train/dev/test splits are kept separate; only curated training examples are used for SFT.
''')
publish('finsafe-training','dataset',training)
benchmark=stage/'benchmark';benchmark.mkdir(exist_ok=True)
shutil.copy2(root/'data/benchmarks/finance_safety_pilot_v1.jsonl',benchmark/'safety_pilot_v1.jsonl')
shutil.copytree(root/'data/processed/steering-calibration-v2',benchmark/'calibration-v2',dirs_exist_ok=True)
shutil.copytree(root/'data/processed/safety-pilot-v1',benchmark/'safety-splits',dirs_exist_ok=True)
shutil.copytree(root/'data/judge-validation',benchmark/'judge-validation',dirs_exist_ok=True)
from finsafe.judge import RUBRIC
(benchmark/'judge-rubric.txt').write_text(RUBRIC)
(benchmark/'README.md').write_text('''---
license: mit
---
# FinSafe AI-assisted finance safety pilot
84 original English prompts with grouped calibration/validation/reserved-test partitions, plus eight additional calibration pairs. MIT applies to project-authored prompts and AI reference annotations; model-generated response excerpts retain applicable model terms. Not an externally validated benchmark. Reference labels are assistant-reviewed, not independent human labels. See the runs article for scope, limitations and protocol. Harmful instructions are evaluation inputs, not endorsed advice.
''')
publish('finsafe-benchmark','dataset',benchmark)
runs=stage/'runs';runs.mkdir(exist_ok=True)
for src in run.iterdir():
 if src.name=='publication':continue
 if src.is_dir():shutil.copytree(src,runs/src.name,dirs_exist_ok=True)
 else:shutil.copy2(src,runs/src.name)
for name in ['ARTICLE.md','tradeoff.png','results.json']:shutil.copy2(stage/name,runs/name)
(runs/'README.md').write_text('''---
license: other
license_name: mixed-upstream-and-project-artifact-terms
license_link: https://github.com/dixitkeshav/finsafe
---
# FinSafe measured preliminary experiment
Read [the complete report](ARTICLE.md). Logs, raw model generations, response labels when available, hashes and AuditKIT reports are included. Code/rubric: MIT; TAT-QA-derived inputs: CC-BY-4.0; model generations and derivative adapter outputs retain applicable upstream Cohere terms. No independent human safety review. Recovery credentials and private checkpoint repositories are excluded.
''')
publish('finsafe-runs','dataset',runs)
space=stage/'space';space.mkdir(exist_ok=True)
(space/'README.md').write_text('''---
title: FinSafe Finance Safety Audit
emoji: 📊
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
---
# FinSafe measured validation dashboard
Static report; no live model or credentials required.
''')
shutil.copy2(stage/'tradeoff.png',space/'tradeoff.png');shutil.copy2(stage/'results.json',space/'results.json')
rowshtml=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in table[1:])
(space/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FinSafe — measured finance and safety</title><style>body{font:16px system-ui;color:#142337;background:#f4f7fb;margin:0;padding:3vw}main{max-width:1100px;margin:auto}h1{font-size:44px;margin-bottom:8px}p{line-height:1.6}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{text-align:left;padding:14px;border-bottom:1px solid #dde4ec}img{max-width:850px;width:100%;background:white;border-radius:12px}.note{background:#fff2cb;padding:18px;border-radius:12px}a{color:#165db5}.scroll{overflow:auto}</style><main><p>SAK_GURU · LEXSI HACKATHON · TRACK 2b</p><h1>FinSafe</h1><p>Measure what finance adaptation changes. Inspect whether refusal steering helps.</p><p>'''+html.escape(finding)+'''</p><div class="scroll"><table><thead><tr>'''+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'''</tr></thead><tbody>'''+rowshtml+'''</tbody></table></div><p>200 finance questions; safety denominators are 12 direct, 12 fixed roleplay attacks and 12 benign requests per condition. Lower harmful rates must be read alongside invalids and benign refusal.</p><img src="tradeoff.png" alt="Measured utility and safety comparison"><p class="note">Preliminary, one seed, one small English benchmark. Safety labels use a same-family AI judge and AI-reviewed reference examples, with no independent human review. Reserved tests remain untouched. This dashboard does not provide financial advice or a live chat model.</p><p><a href="https://huggingface.co/datasets/Finsafe-team/finsafe-runs/blob/main/ARTICLE.md">Read methods and limitations</a> · <a href="https://github.com/dixitkeshav/finsafe">Source and notebooks</a> · <a href="results.json">Download exact results</a></p></main></html>''')
publish('finsafe','space',space)
links={'github':'https://github.com/dixitkeshav/finsafe','space':'https://huggingface.co/spaces/Finsafe-team/finsafe','article':'https://huggingface.co/datasets/Finsafe-team/finsafe-runs/blob/main/ARTICLE.md'}
for title,items,kind in [('FinSafe models',['finsafe-finance-adapter','finsafe-safety-steering'],'model'),('FinSafe datasets',['finsafe-training','finsafe-benchmark','finsafe-runs'],'dataset')]:
 collection=api.create_collection(title,namespace=ns,description='SAK_GURU Lexsi hackathon measured FinSafe artifacts',private=False,exists_ok=True)
 for item in items:api.add_collection_item(collection.slug,f'{ns}/{item}',kind,exists_ok=True)
 links[kind+'_collection']=collection.url
atomic_json(run/'public-links.json',links)
progress.log(json.dumps(links))
progress.log('Public artifact publication complete. Organizer form still requires submission; this script does not claim it has been submitted.')
