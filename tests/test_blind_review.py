import importlib.util,unittest,tempfile,json,hashlib
from pathlib import Path
from finsafe.data import read_jsonl,write_jsonl
spec=importlib.util.spec_from_file_location('blind_review',Path(__file__).parents[1]/'scripts/blind_review.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class BlindReviewTests(unittest.TestCase):
 def test_round_trip_hides_conditions_and_rejects_changed_outputs(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);data=root/'data.jsonl'
   rows=[r for r in read_jsonl(Path(__file__).parents[1]/'data/benchmarks/finance_safety_pilot_v1.jsonl') if r['split']=='validation'][:3]
   write_jsonl(data,rows);sha=hashlib.sha256(data.read_bytes()).hexdigest();files=[]
   for condition in ['baseline','tuned']:
    p=root/(condition+'.jsonl');write_jsonl(p,[{'id':r['id'],'condition':condition,'dataset_sha256':sha,'output':'fixture-'+condition} for r in rows]);files.append(p)
   module.create(data,files,root/'packet');packet=root/'packet/review.jsonl';mapping=root/'packet/coordinator-only-mapping.json'
   reviews=read_jsonl(packet)
   self.assertTrue(all('condition' not in r and r['id'] not in {x['id'] for x in rows} for r in reviews))
   for r in reviews:r.update(label='safe_answer',reviewer='fixture-reviewer')
   write_jsonl(packet,reviews);module.restore(packet,mapping,root/'labels')
   index=json.loads((root/'labels/index.json').read_text());self.assertEqual(set(index),{'baseline','tuned'})
   self.assertEqual(len(read_jsonl(root/'labels'/index['baseline'])),3)
   reviews[0]['output']='changed';write_jsonl(packet,reviews)
   with self.assertRaisesRegex(ValueError,'changed'):module.restore(packet,mapping,root/'tampered')
