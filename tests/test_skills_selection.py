import json
import unittest
from unittest.mock import patch

from main_code.resume_bullet_workflow import select_skills_for_jd
from main_code.workflow_prompts import build_skills_selection_prompts


def _inventory(category_count: int = 5, skills_per_category: int = 8):
    return {
        "categories": [
            {
                "id": f"category_{category_index}",
                "label": f"Category {category_index}",
                "skills": [
                    {
                        "name": f"Skill {category_index}-{skill_index}",
                        "evidence": "selected project",
                    }
                    for skill_index in range(skills_per_category)
                ],
            }
            for category_index in range(category_count)
        ]
    }


class SkillsSelectionPromptTests(unittest.TestCase):
    def test_prompt_explains_purpose_limits_and_selected_evidence(self):
        evidence = {
            "experience_bullets": {"swat": ["Built a routing workflow."]},
            "academic_projects": [],
        }

        system_prompt, user_prompt = build_skills_selection_prompts(
            jd_signals={"must_have": ["Python"]},
            inventory=_inventory(),
            selected_resume_evidence=evidence,
            max_categories=4,
            max_per_category=6,
            max_total_skills=24,
        )

        self.assertIn("compact recruiter-and-ATS index", system_prompt)
        self.assertIn("at most 4 categories", system_prompt)
        self.assertIn("at most 6 skills per category", system_prompt)
        self.assertIn("at most 24 skills total", system_prompt)
        self.assertIn("Aim for 16-20 total skills", system_prompt)
        self.assertIn("selected experience bullet", system_prompt)
        self.assertEqual(
            json.loads(user_prompt)["selected_resume_evidence"], evidence
        )

    @patch("main_code.resume_bullet_workflow.call_llm")
    def test_selector_enforces_category_per_category_and_total_caps(self, call_llm):
        inventory = _inventory()
        call_llm.return_value = json.dumps(
            {
                "categories": [
                    {
                        "id": category["id"],
                        "label": category["label"],
                        "skills": [skill["name"] for skill in category["skills"]],
                    }
                    for category in inventory["categories"]
                ]
            }
        )

        categories, dropped = select_skills_for_jd(
            jd_signals={"must_have": ["Skill 0-0"]},
            model="test-model",
            inventory=inventory,
            selected_bullets={"swat": ["Used Skill 0-0."]},
            max_categories=4,
            max_per_category=6,
            max_total_skills=17,
        )

        self.assertEqual(dropped, [])
        self.assertLessEqual(len(categories), 4)
        self.assertTrue(all(len(category["skills"]) <= 6 for category in categories))
        self.assertEqual(sum(len(category["skills"]) for category in categories), 17)

    @patch("main_code.resume_bullet_workflow.call_llm", side_effect=RuntimeError)
    def test_fallback_obeys_the_same_limits(self, _call_llm):
        categories, dropped = select_skills_for_jd(
            jd_signals={},
            model="test-model",
            inventory=_inventory(),
            max_categories=3,
            max_per_category=4,
            max_total_skills=10,
        )

        self.assertEqual(dropped, [])
        self.assertEqual(len(categories), 3)
        self.assertEqual(sum(len(category["skills"]) for category in categories), 10)


if __name__ == "__main__":
    unittest.main()
