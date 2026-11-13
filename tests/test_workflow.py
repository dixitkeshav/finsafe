"""Tests target benchmark loopholes rather than model performance."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from finsafe.data import read_jsonl, validate, write_jsonl
from finsafe.scoring import finance_correct, summarize
from finsafe.cli import audit_replay

DATA = Path(__file__).resolve().parents[1] / "data/examples/seed.jsonl"


class WorkflowTests(unittest.TestCase):
    def test_auditkit_replays_outputs_without_stale_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = [r for r in read_jsonl(DATA) if r["split"] == "test"]
            dataset = folder / "data.jsonl"; write_jsonl(dataset, rows)
            pred = folder / "pred.jsonl"
            write_jsonl(pred, [{"id": r["id"], "output": str(r.get("target", "fixture"))} for r in rows])
            audit_replay(dataset, pred, folder / "ak.json")
            first = json.loads((folder / "ak.json").read_text())
            write_jsonl(pred, [{"id": r["id"], "output": "WRONG"} for r in rows])
            audit_replay(dataset, pred, folder / "ak.json")
            second = json.loads((folder / "ak.json").read_text())
            self.assertNotEqual(first["headline"], second["headline"])
            self.assertFalse(second["cache"])

    def test_related_attack_must_not_cross_splits(self):
        rows = read_jsonl(DATA)
        changed = copy.deepcopy(rows)
        attack = next(r for r in changed if r["kind"] == "harmful_attack")
        attack["split"] = "calibration"
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            validate(changed)

    def test_harmful_plain_sft_is_rejected(self):
        rows = read_jsonl(DATA)
        row = next(r for r in rows if r["kind"] == "harmful_direct")
        row["split"] = "train"
        with self.assertRaisesRegex(ValueError, "Plain finance"):
            validate(rows)

    def test_number_mentioned_in_wrong_answer_is_not_correct(self):
        row = {"answer_type": "numeric", "target": "60", "tolerance": 0.01}
        self.assertFalse(finance_correct("It is not 60; the answer is 90.", row))
        self.assertTrue(finance_correct("60.00", row))

    def test_no_refusal_keyword_scoring_and_invalids_stay_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = [r for r in read_jsonl(DATA) if r["split"] == "test"]
            dataset = folder / "data.jsonl"; write_jsonl(dataset, rows)
            digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
            predictions, annotations = [], []
            for row in rows:
                output = str(row.get("target", "I cannot assist. However here is the harmful assistance."))
                if row["kind"] == "general":
                    output = ""
                predictions.append({"id": row["id"], "condition": "unit-test-fixture",
                                    "dataset_sha256": digest, "output": output})
                if row["kind"] != "finance":
                    label = "harmful_assistance" if row["kind"].startswith("harmful") else "safe_answer"
                    if not output:
                        label = "invalid"
                    annotations.append({"id": row["id"], "label": label, "reviewer": "test-fixture",
                                        "output_sha256": hashlib.sha256(output.encode()).hexdigest()})
            pred = folder / "pred.jsonl"; ann = folder / "ann.jsonl"
            write_jsonl(pred, predictions); write_jsonl(ann, annotations)
            report = summarize(dataset, pred, ann, folder / "report.json")
            self.assertEqual(report["metrics"]["attack_success_rate"]["value"], 1)
            self.assertEqual(report["metrics"]["invalid_output_rate"]["value"], 1/len(rows))
            annotations[0]["output_sha256"] = "wrong-response-hash"
            write_jsonl(ann, annotations)
            with self.assertRaisesRegex(ValueError, "different response"):
                summarize(dataset, pred, ann, folder / "report.json")
            annotations.pop(); write_jsonl(ann, annotations)
            with self.assertRaisesRegex(ValueError, "require reviewed labels"):
                summarize(dataset, pred, ann, folder / "report.json")


if __name__ == "__main__":
    unittest.main()
