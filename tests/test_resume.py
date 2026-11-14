import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from finsafe.data import write_jsonl
from finsafe.resume import generation_records
from finsafe.progress import Progress


class ResumeTests(unittest.TestCase):
    def test_generation_resumes_prefix_and_rejects_changed_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'predictions.jsonl'
            contract={'condition':'baseline','dataset_sha256':'abc','max_new_tokens':64}
            rows=[{'id':'a'},{'id':'b'}]
            self.assertEqual(generation_records(out,contract,rows,resume=True),[])
            write_jsonl(out,[{'id':'a','output':'1','condition':'baseline','dataset_sha256':'abc'}])
            self.assertEqual(len(generation_records(out,contract,rows,resume=True)),1)
            with self.assertRaisesRegex(ValueError,'changed'):
                generation_records(out,{**contract,'max_new_tokens':128},rows,resume=True)
            write_jsonl(out,[{'id':'b','output':'2','condition':'baseline','dataset_sha256':'abc'}])
            with self.assertRaisesRegex(ValueError,'prefix'):
                generation_records(out,contract,rows,resume=True)

    def test_failed_stage_is_recorded_and_tokens_are_redacted(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            progress=Progress(tmp)
            progress.log('hf_'+'a'*30)
            with self.assertRaises(RuntimeError):
                with progress.stage('test-stage'):
                    raise RuntimeError('fixture')
            status=json.loads((Path(tmp)/'test-stage.json').read_text())
            self.assertEqual(status['status'],'failed')
            self.assertNotIn('hf_'+'a'*30,(Path(tmp)/'kaggle-stage-02.log').read_text())


if __name__=='__main__':
    unittest.main()
