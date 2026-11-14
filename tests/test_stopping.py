import unittest,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('inspect_stopping',Path(__file__).parents[1]/'scripts/inspect_stopping.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class StoppingTests(unittest.TestCase):
 def test_unknown_old_metadata_and_interior_eos_are_distinguished(self):
  report=module.inspect([{'id':'old','output_tokens':20},
    {'id':'ended','output_tokens':4,'max_new_tokens':8,'last_token_id':6,'first_tokenizer_eos_position':3},
    {'id':'continued','output_tokens':8,'max_new_tokens':8,'last_token_id':13,'first_tokenizer_eos_position':2}])
  self.assertEqual(report['token_diagnostic_missing'],1)
  self.assertEqual(report['interior_tokenizer_eos_ids'],['continued'])
  self.assertEqual(report['cap_hits'],1)
