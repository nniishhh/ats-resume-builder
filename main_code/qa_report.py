"""Build report for a generated resume.

Everything here is deterministic — no LLM calls. The point is to surface, before the PDF is
sent anywhere, the classes of defect that otherwise only get caught by a human reading the
rendered page carefully:

  - two-page output
  - bullets that wrap badly and leave an orphan line
  - repeated starting verbs
  - the same distinctive figure repeated in back-to-back bullets within a company
  - claims (numbers, tools) that do not appear in the underlying evidence files
  - skills that are not in the master inventory
  - JD must-haves the resume never mentions

Rendered as plain text so it works identically in the CLI and in Streamlit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

try:  # works both as `main_code.qa_report` and as a bare local import
    from main_code import layout
except ImportError:  # pragma: no cover - direct script use
    import layout

OK = "OK"
WARN = "!!"


@dataclass
class QAReport:
    pages: int = 0
    trim_actions: List[str] = field(default_factory=list)
    plan_note: str = ""
    bullet_count: int = 0
    repeated_verbs: Dict[str, int] = field(default_factory=dict)
    orphans: List[str] = field(default_factory=list)
    redundant_facts: List[str] = field(default_factory=list)
    grounding_issues: List[str] = field(default_factory=list)
    skills_kept: int = 0
    skills_dropped: List[str] = field(default_factory=list)
    jd_must_have: List[str] = field(default_factory=list)
    jd_matched: List[str] = field(default_factory=list)

    @property
    def jd_missing(self) -> List[str]:
        return [k for k in self.jd_must_have if k not in self.jd_matched]

    @property
    def clean(self) -> bool:
        return (
            self.pages == 1
            and not self.repeated_verbs
            and not self.orphans
            and not self.redundant_facts
            and not self.grounding_issues
            and not self.skills_dropped
        )

    def render(self) -> str:
        rows: List[str] = []

        page_note = f"{self.pages} page{'s' if self.pages != 1 else ''}"
        if self.trim_actions:
            page_note += f"   ({len(self.trim_actions)} trim pass{'es' if len(self.trim_actions) != 1 else ''}: {'; '.join(self.trim_actions)})"
        rows.append(_row("Layout", self.pages == 1, page_note))

        if self.plan_note:
            rows.append(_row("Structure", True, self.plan_note))

        verb_note = f"{self.bullet_count} bullets, "
        if self.repeated_verbs:
            dupes = ", ".join(f"{v} x{c}" for v, c in sorted(self.repeated_verbs.items()))
            verb_note += f"repeated starting verbs: {dupes}"
        else:
            verb_note += "all starting verbs unique"
        rows.append(_row("Verbs", not self.repeated_verbs, verb_note))

        if self.orphans:
            rows.append(_row("Line fit", False, self.orphans[0]))
            for extra in self.orphans[1:]:
                rows.append(_row("", False, extra))
        else:
            rows.append(_row("Line fit", True, "no orphan lines"))

        if self.redundant_facts:
            rows.append(_row("Redundancy", False, self.redundant_facts[0]))
            for extra in self.redundant_facts[1:]:
                rows.append(_row("", False, extra))
        else:
            rows.append(_row("Redundancy", True, "no repeated figures in adjacent bullets"))

        if self.grounding_issues:
            rows.append(_row("Grounding", False, self.grounding_issues[0]))
            for extra in self.grounding_issues[1:]:
                rows.append(_row("", False, extra))
        else:
            rows.append(_row("Grounding", True, "all claims traced to evidence"))

        skills_note = f"{self.skills_kept} items, all traced to inventory"
        if self.skills_dropped:
            skills_note = (
                f"{self.skills_kept} kept, "
                f"{len(self.skills_dropped)} dropped (not in inventory): "
                + ", ".join(self.skills_dropped)
            )
        rows.append(_row("Skills", not self.skills_dropped, skills_note))

        if self.jd_must_have:
            note = f"{len(self.jd_matched)}/{len(self.jd_must_have)} must-haves matched"
            if self.jd_missing:
                note += "\n" + " " * 14 + "missing: " + ", ".join(self.jd_missing)
            rows.append(_row("JD coverage", not self.jd_missing, note))

        header = "BUILD REPORT" + ("" if self.clean else "  (issues found)")
        return "\n".join([header, "-" * 68, *rows])


def _row(label: str, ok: bool, note: str) -> str:
    mark = OK if ok else WARN
    return f"{label:<12}{mark}  {note}"


def check_layout(pdf_path: Path, report: QAReport) -> QAReport:
    """Fill in page count, verb repetition, and orphan lines from the compiled PDF."""
    report.pages = layout.page_count(pdf_path)
    bullets = layout.measure_bullets(pdf_path)
    width = layout.column_width(bullets)

    report.bullet_count = len(bullets)
    report.repeated_verbs = layout.repeated_verbs(bullets)
    report.orphans = [
        f"bullet #{b.index + 1} last line {b.fill_ratio(width):.0%} full: {b.lines[-1]!r}"
        for b in layout.find_orphans(bullets, col_width=width)
    ]
    return report


_NUMBER = re.compile(r"\d[\d,.]*\s*(?:%|K|M|B|x)?", re.IGNORECASE)


def check_redundancy(
    bullets_by_company: Dict[str, Sequence[str]], report: QAReport
) -> QAReport:
    """Flag a distinctive figure (e.g. "10K", "6M") repeated in back-to-back bullets.

    Each fact is grounded and each bullet reads fine in isolation, but a reader scanning
    top to bottom sees the same headline number twice in a row — a case-length check on a
    single bullet can't catch this, only comparing neighbors can.
    """
    for company, bullets in bullets_by_company.items():
        bullet_numbers = [_normalised_numbers(b) for b in bullets]
        for i in range(len(bullet_numbers) - 1):
            shared = bullet_numbers[i] & bullet_numbers[i + 1]
            if shared:
                report.redundant_facts.append(
                    f"{company} #{i + 1}/#{i + 2}: both mention {', '.join(sorted(shared))}"
                )
    return report


def check_grounding(
    bullets_by_company: Dict[str, Sequence[str]],
    evidence_by_company: Dict[str, Sequence[dict]],
    report: QAReport,
) -> QAReport:
    """Flag numeric claims that do not appear anywhere in that company's evidence.

    Deliberately narrow: numbers are unambiguous and are the highest-consequence thing to get
    wrong. Free-text tool names produce too many false positives to gate on, so those are left
    to the skills inventory whitelist instead.
    """
    for company, bullets in bullets_by_company.items():
        evidence_blob = _evidence_text(evidence_by_company.get(company, []))
        evidence_numbers = _normalised_numbers(evidence_blob)
        for i, bullet in enumerate(bullets, start=1):
            for token in _normalised_numbers(bullet):
                if token not in evidence_numbers:
                    report.grounding_issues.append(
                        f"{company} #{i}: '{token}' not found in evidence"
                    )
    return report


def _evidence_text(projects: Sequence[dict]) -> str:
    parts: List[str] = []
    for project in projects:
        for key in ("problem", "actions", "main_metric", "sub_metrics", "results", "tools", "keywords",
                    "example_bullets", "harvested_bullets"):
            value = project.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple)):
                parts.extend(str(v) for v in value)
    return " ".join(parts)


def _normalised_numbers(text: str) -> set[str]:
    out = set()
    for raw in _NUMBER.findall(text):
        token = raw.replace(",", "").replace(" ", "").rstrip(".").upper()
        if token and not token.replace("%", "").replace(".", "").isspace():
            out.add(token)
    return out


# Words that carry no matching signal — a keyword and the resume can phrase the
# connective tissue differently and still mean the same thing.
_FILLER = {
    "a", "an", "and", "the", "of", "in", "on", "for", "with", "to", "or",
    "experience", "strong", "using", "related", "field", "degree",
}


def _content_tokens(text: str) -> set[str]:
    """Significant words, with punctuation that varies between phrasings removed."""
    cleaned = re.sub(r"[^a-z0-9+#]+", " ", text.lower())
    return {t for t in cleaned.split() if t and t not in _FILLER}


def check_jd_coverage(
    must_have: Sequence[str], rendered_text: str, report: QAReport
) -> QAReport:
    """Which JD must-have keywords actually appear in the finished resume.

    Matched on content words rather than as a literal substring. A plain `in` test
    reported "MS in Operations Research" as missing from a resume reading "MS,
    Operations Research" — the degree was right there, only the punctuation differed,
    and the one number meant to measure fit was wrong because of it.

    All content words must be present, so this stays conservative: it forgives
    punctuation and filler, not genuinely absent skills.
    """
    haystack = _content_tokens(rendered_text)
    report.jd_must_have = list(must_have)
    report.jd_matched = [
        k for k in must_have
        if (toks := _content_tokens(k)) and toks <= haystack
    ]
    return report
