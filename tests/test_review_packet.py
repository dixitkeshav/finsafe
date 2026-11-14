import unittest,tempfile,subprocess,sys,json,hashlib
from pathlib import Path
from finsafe.data import read_jsonl,write_jsonl
class ReviewPacketTests(unittest.TestCase):
 def test_rerun_preserves_labels_and_rejects_changed_responses(self):
  project=Path(__file__).parents[1]
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);data=root/'data.jsonl';pred=root/'pred.jsonl';review=root/'review.jsonl'
   rows=[r for r in read_jsonl(project/'data/benchmarks/finance_safety_pilot_v1.jsonl') if r['split']=='validation'][:3]
   write_jsonl(data,rows);sha=hashlib.sha256(data.read_bytes()).hexdigest()
   predictions=[{'id':r['id'],'dataset_sha256':sha,'output':'fixture','output_tokens':1,'max_new_tokens':10} for r in rows]
   write_jsonl(pred,predictions)
   command=[sys.executable,str(project/'scripts/review_packet.py'),'--data',str(data),'--predictions',str(pred),'--out',str(review)]
   subprocess.run(command,check=True,capture_output=True)
   labels=read_jsonl(review);labels[0].update(label='safe_answer',reviewer='test-reviewer',notes='Preserve this reviewed work')
   write_jsonl(review,labels);original=review.read_bytes()
   subprocess.run(command,check=True,capture_output=True);self.assertEqual(review.read_bytes(),original)
   predictions[0]['output']='changed';write_jsonl(pred,predictions)
   result=subprocess.run(command,capture_output=True);self.assertNotEqual(result.returncode,0)
   self.assertEqual(review.read_bytes(),original)
