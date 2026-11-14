import unittest
from pathlib import Path
from finsafe.data import read_jsonl,validate
from finsafe.scoring import finance_format_valid,finance_correct
class SafetyPilotTests(unittest.TestCase):
 def test_behavior_controls_never_cross_splits(self):
  rows=validate(read_jsonl(Path(__file__).parents[1]/'data/benchmarks/finance_safety_pilot_v1.jsonl'))
  self.assertEqual(len(rows),84)
  for group in {r['group_id'] for r in rows}:
   subset=[r for r in rows if r['group_id']==group]
   self.assertEqual(len({r['split'] for r in subset}),1)
   self.assertEqual({r['kind'] for r in subset},{'harmful_direct','harmful_attack','benign_sensitive'})
 def test_malformed_and_wrong_numeric_outputs_are_distinguished(self):
  row={'answer_type':'numeric_scale','target':{'answer':5,'scale':'million'}}
  self.assertFalse(finance_format_valid('```json\n{"answer":5,"scale":"million"}\n```',row))
  self.assertFalse(finance_format_valid('{"answer":NaN,"scale":"million"}',row))
  self.assertTrue(finance_format_valid('{"answer":9,"scale":"million"}',row))
  self.assertFalse(finance_correct('{"answer":9,"scale":"million"}',row))
 def test_safety_audit_rejects_unreviewed_and_uses_kind_denominators(self):
  import tempfile,hashlib,json
  from finsafe.cli import audit_safety_replay
  from finsafe.data import write_jsonl
  source=Path(__file__).parents[1]/'data/benchmarks/finance_safety_pilot_v1.jsonl'
  rows=[r for r in read_jsonl(source) if r['split']=='validation'][:3]
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);data=folder/'data.jsonl';pred=folder/'pred.jsonl';ann=folder/'ann.jsonl';out=folder/'audit.json'
   write_jsonl(data,rows);sha=hashlib.sha256(data.read_bytes()).hexdigest()
   write_jsonl(pred,[{'id':r['id'],'output':'fixture','dataset_sha256':sha,'condition':'test'} for r in rows])
   labels=[{'id':r['id'],'label':label,'reviewer':'unit-test','output_sha256':hashlib.sha256(b'fixture').hexdigest()} for r,label in zip(rows,['refusal','harmful_assistance','safe_answer'])]
   write_jsonl(ann,labels);audit_safety_replay(data,pred,ann,out)
   report=json.loads(out.read_text())
   self.assertEqual(report['reviewed_metrics']['attack_success_rate']['value'],1)
   self.assertEqual(report['reviewed_metrics']['direct_harmful_compliance']['value'],0)
   self.assertEqual(report['auditkit_by_kind']['attack_success_rate']['n'],1)
   labels[0]['label']='UNREVIEWED';write_jsonl(ann,labels)
   with self.assertRaises(ValueError):audit_safety_replay(data,pred,ann,out)
