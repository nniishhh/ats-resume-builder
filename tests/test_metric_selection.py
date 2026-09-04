import unittest

from main_code.resume_bullet_workflow import _evidence_numeric_claims
from main_code.workflow_prompts import build_bullet_generation_system_prompt


class MetricSelectionTests(unittest.TestCase):
    def test_prompt_prioritizes_main_metric_and_limits_sub_metrics(self):
        prompt = build_bullet_generation_system_prompt(200, 240)

        self.assertIn('"main_metric" is the default measurable result', prompt)
        self.assertIn("Add at most ONE sub-metric", prompt)
        self.assertIn("never place multiple sub-metrics in the same bullet", prompt)

    def test_grounding_reads_main_and_sub_metric_fields(self):
        evidence = [
            {
                "main_metric": "63.7 seconds of added runtime",
                "sub_metrics": ["Roughly 70% longer than normal"],
            }
        ]

        self.assertEqual(_evidence_numeric_claims(evidence), {"63.7", "70%"})


if __name__ == "__main__":
    unittest.main()
