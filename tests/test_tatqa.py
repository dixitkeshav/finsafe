import json
import unittest
from finsafe.tatqa import convert_context
from finsafe.scoring import finance_correct


class TatqaTests(unittest.TestCase):
    def test_value_and_scale_both_required(self):
        row = {"answer_type": "numeric_scale", "target": {"answer": 4.6, "scale": "million"}}
        self.assertTrue(finance_correct('{"answer":4.60,"scale":"million"}', row))
        self.assertFalse(finance_correct('{"answer":4.6,"scale":"percent"}', row))
        self.assertFalse(finance_correct('4.6', row))
        self.assertFalse(finance_correct('{"answer":true,"scale":"million"}', row))
        self.assertFalse(finance_correct('{"answer":NaN,"scale":"million"}', row))

    def test_conversion_preserves_scale_without_gold_derivation(self):
        c = {"table": {"uid": "t1", "table": [["Year", "Sales"], ["2020", "4.6"]]},
             "paragraphs": [{"order": 1, "text": "Sales are in millions."}],
             "questions": [{"uid": "q1", "answer_type": "arithmetic", "answer": 4.6,
                            "question": "What is sales?", "scale": "million", "derivation": "SECRET GOLD DERIVATION"}]}
        first = convert_context(c, "train")[0]
        self.assertEqual(first["target"], {"answer": 4.6, "scale": "million"})
        self.assertNotIn("SECRET GOLD", first["prompt"])
        c["table"]["uid"] = "different-source-id"
        self.assertEqual(first["group_id"], convert_context(c, "test")[0]["group_id"])


if __name__ == "__main__":
    unittest.main()
