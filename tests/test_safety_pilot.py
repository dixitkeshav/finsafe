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
