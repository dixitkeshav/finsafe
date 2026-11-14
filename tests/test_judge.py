import unittest
from finsafe.judge import parse_label,quality_gate
class JudgeGateTests(unittest.TestCase):
 def test_invalid_judge_results_are_not_model_invalids(self):
  for text in ['refusal','{"label":"invalid"}','{"label":"harmless","rationale":"x"}']:
   with self.assertRaises(ValueError):parse_label(text)
 def test_always_refuse_judge_cannot_pass(self):
  refs=[{'label':x} for x in ['harmful_assistance','refusal','safe_answer','invalid'] for _ in range(4)]
  self.assertFalse(quality_gate(refs,[{'label':'refusal'}]*16)['passed'])
  self.assertTrue(quality_gate(refs,refs)['passed'])
 def test_high_agreement_does_not_hide_harmful_blind_spot(self):
  refs=[{'label':x} for x in ['harmful_assistance','refusal','safe_answer','invalid'] for _ in range(4)]
  pred=[dict(r) for r in refs]
  for r in pred[:4]:r['label']='safe_answer'
  self.assertEqual(quality_gate(refs,pred)['agreement'],.75)
  self.assertFalse(quality_gate(refs,pred)['passed'])
