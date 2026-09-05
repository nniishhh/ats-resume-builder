"""What makes a drafted cover letter safe to submit.

These are not "does it produce nice prose" tests — that is a judgement call for
the human reading it. They pin the two rules that decide whether a draft may
reach an employer at all: it may not contain a placeholder, and it may not state
a figure the evidence does not support. Both fail CLOSED — the draft is refused,
not trimmed — because a cover letter is attached to a real application and a
trimmed lie is still a lie.
"""

import unittest

from main_code.build_cover_letter import render_cover_letter_tex
from main_code.cover_letter_workflow import (
    MAX_COVER_LETTER_WORDS,
    _allowed_claims,
    parse_cover_letter,
    validate_cover_letter,
)

EVIDENCE = {
    "mta": [
        {
            "company": "MTA New York City Transit",
            "problem": "Flagging records and train arrivals lived in separate systems.",
            "actions": [
                "Built a pipeline joining 6M train-arrival records to 490K flagging work orders"
            ],
            "main_metric": "63.7 seconds of added runtime",
            "sub_metrics": ["Roughly 70% longer than normal"],
        }
    ]
}

JD = "You will analyse transit performance data across a 6,000-vehicle fleet."

# A letter long enough to clear the word floor, so a length complaint never
# masks the rule actually under test.
FILLER = (
    "I have spent the last two years turning messy operational data into decisions "
    "that operating departments will actually act on, which is the part of this work "
    "I care about most. The problems that interest me are the ones where the data "
    "exists but nobody has joined it up yet, and where the answer changes how a "
    "schedule gets written rather than sitting in a deck. I read the posting as "
    "describing exactly that kind of work, and it is the reason I am applying rather "
    "than looking for a general analytics role. I would be glad to talk through how "
    "the approach transfers to the systems your team already runs day to day. "
)


def allowed():
    return _allowed_claims(EVIDENCE, JD)


class PlaceholderRejectionTests(unittest.TestCase):
    """A placeholder must fail the draft, never be quietly cleaned up.

    This is the whole reason the validator refuses instead of repairing: a letter
    reading "Dear [Hiring Manager]" that got trimmed is a letter that gets sent.
    """

    def test_square_bracket_placeholder_is_rejected(self):
        letter = [f"I am applying to [Company] for this role. {FILLER}", FILLER]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertTrue(any("placeholder" in i.lower() for i in issues), issues)

    def test_double_brace_placeholder_is_rejected(self):
        letter = [f"My work at Nvidia on {{{{project}}}} applies here. {FILLER}", FILLER]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertTrue(any("placeholder" in i.lower() for i in issues), issues)

    def test_angle_bracket_and_todo_are_rejected(self):
        for bad in ("<YOUR NAME>", "TBD", "TODO"):
            with self.subTest(bad=bad):
                letter = [f"Nvidia is hiring and {bad} fits. {FILLER}", FILLER]
                issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
                self.assertTrue(any("placeholder" in i.lower() for i in issues), (bad, issues))

    def test_a_clean_letter_passes(self):
        letter = [
            "I am applying for the Data Scientist role at Nvidia. " + FILLER,
            "At MTA New York City Transit I joined 6M train-arrival records to 490K "
            "flagging work orders and showed 63.7 seconds of added runtime. " + FILLER,
        ]
        self.assertEqual(validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed()), [])


class NumericGroundingTests(unittest.TestCase):
    """Figures must come from the evidence or the JD — the bullets' rule, in prose."""

    def test_invented_figure_is_rejected(self):
        letter = [
            "At Nvidia I would repeat what I did at the MTA, where I cut delays by 45%. "
            + FILLER,
            FILLER,
        ]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertTrue(any("45%" in i for i in issues), issues)

    def test_evidence_figures_are_allowed(self):
        letter = [
            "At Nvidia I would apply the same approach. " + FILLER,
            "I joined 6M records to 490K work orders and measured 63.7 seconds of "
            "added runtime, about 70% longer than normal. " + FILLER,
        ]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertEqual(issues, [], issues)

    def test_a_figure_quoted_from_the_job_description_is_allowed(self):
        # Quoting the posting's own numbers back at it is legitimate, and the JD
        # is part of the allowed set for exactly that reason.
        letter = [
            "Your 6,000-vehicle fleet is the scale I have worked at. Nvidia. " + FILLER,
            FILLER,
        ]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertEqual(issues, [], issues)

    def test_a_year_is_not_treated_as_a_claim(self):
        # Years are exempt: "since 2024" is not a metric about results, and
        # failing it would reject almost every honest letter.
        letter = [
            "I have worked on transit data since 2024, and Nvidia is the next step. "
            + FILLER,
            FILLER,
        ]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertEqual(issues, [], issues)


class MagnitudeInflationTests(unittest.TestCase):
    """Evidence about minutes must not license a letter about millions.

    The shared _numeric_claims regex reads the "m" of "minutes" as a magnitude
    suffix, so "3.1 minutes" and "3.1 million" collapse to the same token. Bare
    numbers alone therefore cannot catch this; spelled-out magnitudes are checked
    as their own claim.
    """

    def test_a_magnitude_absent_from_the_evidence_is_rejected(self):
        letter = [
            "Nvidia. At the MTA I analysed 6 million train-arrival records. " + FILLER,
            FILLER,
        ]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertTrue(any("magnitude" in i.lower() for i in issues), issues)

    def test_minutes_in_the_evidence_do_not_license_millions_in_the_letter(self):
        evidence = {
            "mta": [{"company": "MTA", "main_metric": "3.1 minutes of added travel time"}]
        }
        letter = [
            "Nvidia. My work covered 3.1 million riders a day. " + FILLER,
            FILLER,
        ]
        issues = validate_cover_letter(
            letter, "Nvidia", "Data Scientist", _allowed_claims(evidence, "")
        )
        self.assertTrue(any("magnitude" in i.lower() for i in issues), issues)

    def test_a_magnitude_present_in_the_evidence_is_allowed(self):
        evidence = {"mta": [{"company": "MTA", "problem": "A network of 3.1 million riders."}]}
        letter = [
            "Nvidia. My work covered 3.1 million riders a day. " + FILLER,
            FILLER,
        ]
        issues = validate_cover_letter(
            letter, "Nvidia", "Data Scientist", _allowed_claims(evidence, "")
        )
        self.assertEqual(issues, [], issues)


class InventedAffinityTests(unittest.TestCase):
    """The prose-only failure mode: a motivation nobody supplied.

    Has no bullet equivalent — a bullet cannot claim to have admired someone
    since childhood — and the model reaches for it unprompted.
    """

    def test_long_admired_is_rejected(self):
        letter = [f"I have long admired Nvidia's work. {FILLER}", FILLER]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertTrue(any("personal history" in i for i in issues), issues)

    def test_since_childhood_is_rejected(self):
        letter = [f"Since childhood I wanted to work at Nvidia. {FILLER}", FILLER]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertTrue(any("personal history" in i for i in issues), issues)


class EmployerNamedTests(unittest.TestCase):
    def test_a_letter_that_never_names_the_company_is_rejected(self):
        letter = [FILLER, FILLER]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertTrue(any("never names the employer" in i for i in issues), issues)


class LengthTests(unittest.TestCase):
    def test_too_long_is_rejected(self):
        letter = ["Nvidia. " + FILLER * 8, FILLER]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertTrue(any("too long" in i for i in issues), issues)

    def test_too_short_is_rejected(self):
        letter = ["I would like to work at Nvidia.", "Thanks."]
        issues = validate_cover_letter(letter, "Nvidia", "Data Scientist", allowed())
        self.assertTrue(any("too short" in i for i in issues), issues)

    def test_the_word_cap_is_what_keeps_the_letter_to_one_page(self):
        # build_cover_letter deliberately has no "drop a sentence" ladder, so the
        # word cap is the only thing standing between a draft and a two-page
        # letter. If this constant grows, the page check has to be revisited.
        self.assertLessEqual(MAX_COVER_LETTER_WORDS, 400)


class ParsingTests(unittest.TestCase):
    """The model adds a salutation and sign-off however firmly it is told not to."""

    def test_salutation_and_signoff_are_stripped(self):
        raw = (
            "Dear Hiring Team,\n\n"
            "First paragraph about the role.\n\n"
            "Second paragraph about the evidence.\n\n"
            "Sincerely,\nOranich Jamkachornkiat"
        )
        self.assertEqual(
            parse_cover_letter(raw),
            ["First paragraph about the role.", "Second paragraph about the evidence."],
        )

    def test_markdown_fence_is_stripped(self):
        raw = "```\nOne paragraph.\n\nTwo paragraph.\n```"
        self.assertEqual(parse_cover_letter(raw), ["One paragraph.", "Two paragraph."])

    def test_hard_wrapped_lines_become_one_paragraph(self):
        raw = "This sentence was\nhard wrapped by the model.\n\nSecond block here."
        self.assertEqual(
            parse_cover_letter(raw),
            ["This sentence was hard wrapped by the model.", "Second block here."],
        )


class TemplateInjectionTests(unittest.TestCase):
    """A leaked marker must be loud, not invisible."""

    TEMPLATE = (
        "<<<DATE>>>\n\nHiring Team \\\\\n<<<COMPANY>>>\n\n"
        "\\textbf{Re: <<<ROLE>>>}\n\nDear Hiring Team,\n\n<<<BODY>>>\n"
    )

    def test_all_markers_are_filled(self):
        tex = render_cover_letter_tex(
            self.TEMPLATE, ["Body one.", "Body two."], "Nvidia", "Data Scientist"
        )
        for marker in ("<<<DATE>>>", "<<<COMPANY>>>", "<<<ROLE>>>", "<<<BODY>>>"):
            self.assertNotIn(marker, tex)
        self.assertIn("Nvidia", tex)
        self.assertIn("Data Scientist", tex)

    def test_a_marker_arriving_via_the_body_raises_rather_than_compiling(self):
        # Markers are filled in a fixed order — DATE, COMPANY, ROLE, then BODY — so
        # a marker that arrives *inside* the body is never substituted. Without the
        # leak check this compiles happily into a letter reading "<<<COMPANY>>>".
        with self.assertRaises(RuntimeError):
            render_cover_letter_tex(
                self.TEMPLATE,
                ["A paragraph mentioning <<<COMPANY>>> verbatim."],
                "Nvidia",
                "Data Scientist",
            )

    def test_a_template_with_an_extra_copy_of_a_marker_still_fills_it(self):
        # Every occurrence is replaced, not just the first — a template that names
        # the company twice must not leak the second one.
        tex = render_cover_letter_tex(
            self.TEMPLATE + "\n<<<COMPANY>>>\n",
            ["Body one.", "Body two."],
            "Nvidia",
            "Data Scientist",
        )
        self.assertNotIn("<<<COMPANY>>>", tex)
        self.assertEqual(tex.count("Nvidia"), 2)

    def test_latex_special_characters_are_escaped(self):
        tex = render_cover_letter_tex(
            self.TEMPLATE, ["Cut cost 30% & saved $5."], "AT&T", "R&D Scientist"
        )
        self.assertIn("30\\%", tex)
        self.assertIn("AT\\&T", tex)
        self.assertIn("R\\&D", tex)
        self.assertIn("\\$5", tex)

    def test_no_paragraphs_refuses(self):
        with self.assertRaises(ValueError):
            render_cover_letter_tex(self.TEMPLATE, [], "Nvidia", "Data Scientist")


if __name__ == "__main__":
    unittest.main()
