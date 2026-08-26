"""Rendered-layout measurement for compiled resumes.

Everything here is deterministic and reads the compiled PDF as ground truth. The LLM cannot
see rendered width, so character-count rules are only ever a proxy; these helpers measure what
actually happened on the page.

Two problems this solves:

1. **Page overflow.** Nothing in the pipeline used to check page count, so a two-page resume
   shipped silently.
2. **Orphan lines.** A bullet that wraps and leaves one or two words alone on its last line looks
   unfinished. Character-band rules cannot detect this because it depends on where the text
   actually breaks.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

# pdftotext -layout indents wrapped continuation lines. Two spaces is what the current
# template produces; keep the check loose so spacing tweaks do not silently break parsing.
_CONTINUATION_INDENT = "  "

# Fallback column width in characters, calibrated against the current template
# (US Letter, 0.5in side margins, Times New Roman 10pt). Only used when a document
# has too few bullet lines to measure from.
DEFAULT_COL_WIDTH = 126

# A bullet's last line this sparse reads as an orphan ("by 120%." alone on a line).
#
# Calibrated by measuring real resumes rather than guessed: at a ~132-char column, 0.15 is
# about 20 characters, or three short words. A first attempt at 0.5 flagged half the bullets
# on pages that look perfectly fine, so it is deliberately low — this check exists to catch
# obviously broken wraps, not to police line balance.
DEFAULT_MIN_FILL = 0.15

# Never treat a last line longer than this as an orphan, whatever the ratio says.
MIN_ACCEPTABLE_LAST_LINE = 20


@dataclass
class BulletMetrics:
    """One rendered bullet, as it actually appears in the PDF."""

    index: int
    text: str
    lines: List[str]

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def last_line_len(self) -> int:
        return len(self.lines[-1].strip()) if self.lines else 0

    @property
    def start_verb(self) -> str:
        match = re.match(r"\s*([A-Za-z][a-z]+)", self.text)
        return match.group(1) if match else ""

    def fill_ratio(self, col_width: int) -> float:
        if not self.lines or col_width <= 0:
            return 1.0
        return self.last_line_len / col_width


def _require(binary: str) -> None:
    if shutil.which(binary) is None:
        raise RuntimeError(
            f"'{binary}' not found. Install poppler (macOS: brew install poppler)."
        )


def page_count(pdf_path: Path) -> int:
    """Number of pages in the compiled PDF."""
    _require("pdfinfo")
    out = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not read page count from pdfinfo output for {pdf_path}")
    return int(match.group(1))


def extract_layout_text(pdf_path: Path) -> str:
    """Text of the PDF with line breaks preserved as rendered."""
    _require("pdftotext")
    return subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def measure_bullets(pdf_path: Path) -> List[BulletMetrics]:
    """Parse the rendered PDF into per-bullet line measurements.

    A bullet starts at a line whose first non-space character is the bullet glyph. Wrapped
    continuation lines are indented and carry no glyph; anything else (section headings,
    company lines, skills rows) closes the current bullet.
    """
    bullets: List[BulletMetrics] = []
    current: Optional[List[str]] = None

    for raw in extract_layout_text(pdf_path).splitlines():
        stripped = raw.strip()
        if stripped.startswith("•"):
            if current is not None:
                bullets.append(_finalise(len(bullets), current))
            current = [stripped.lstrip("•").strip()]
        elif current is not None and stripped and raw.startswith(_CONTINUATION_INDENT):
            current.append(stripped)
        elif current is not None:
            bullets.append(_finalise(len(bullets), current))
            current = None

    if current is not None:
        bullets.append(_finalise(len(bullets), current))
    return bullets


def _finalise(index: int, lines: List[str]) -> BulletMetrics:
    return BulletMetrics(index=index, text=" ".join(lines), lines=list(lines))


def column_width(bullets: List[BulletMetrics]) -> int:
    """Self-calibrating column width, in characters.

    Uses the widest rendered line actually observed, so it stays correct if margins or font
    size change. Falls back to the calibrated default when there is not enough to measure.
    """
    widths = [len(line) for b in bullets for line in b.lines[:-1]]  # full lines only
    return max(widths) if widths else DEFAULT_COL_WIDTH


def find_orphans(
    bullets: List[BulletMetrics],
    min_fill: float = DEFAULT_MIN_FILL,
    col_width: Optional[int] = None,
) -> List[BulletMetrics]:
    """Bullets whose final rendered line is too sparse to look intentional."""
    width = col_width if col_width is not None else column_width(bullets)
    return [
        b
        for b in bullets
        if b.line_count > 1
        and b.fill_ratio(width) < min_fill
        and b.last_line_len < MIN_ACCEPTABLE_LAST_LINE
    ]


def repeated_verbs(bullets: List[BulletMetrics]) -> dict[str, int]:
    """Starting verbs used more than once, verb -> count."""
    counts: dict[str, int] = {}
    for b in bullets:
        verb = b.start_verb
        if verb:
            counts[verb] = counts.get(verb, 0) + 1
    return {v: c for v, c in counts.items() if c > 1}


def char_budget_for_lines(target_lines: int, col_width: int = DEFAULT_COL_WIDTH) -> tuple[int, int]:
    """Character range that renders as roughly `target_lines` well-filled lines.

    Lower bound keeps the last line at least half full; upper bound stays inside the target.
    This replaces the old fixed 200-240 band, which forced every bullet to the same shape and
    made clean one-liners impossible to express.
    """
    if target_lines < 1:
        raise ValueError("target_lines must be >= 1")
    lower = int(col_width * (target_lines - 1 + DEFAULT_MIN_FILL))
    upper = int(col_width * target_lines) - 2  # leave room for the bullet glyph and indent
    return max(lower, 40), upper


# ---------------------------------------------------------------------------
# Deterministic verb de-duplication
# ---------------------------------------------------------------------------

# Past-tense, transitive resume openers. Substituting one for another at position 0 is
# grammatically safe because they all take a direct object the same way ("Built a pipeline"
# -> "Engineered a pipeline"). Grouped by rough meaning so a swap does not change the claim:
# an analysis verb never becomes a construction verb.
VERB_ALTERNATIVES: dict[str, List[str]] = {
    # building / creating
    "built": ["Engineered", "Developed", "Created", "Constructed", "Implemented",
              "Assembled", "Produced", "Established", "Delivered", "Shipped"],
    "created": ["Built", "Developed", "Established", "Introduced", "Produced",
                "Assembled", "Authored"],
    "developed": ["Built", "Engineered", "Created", "Implemented", "Produced",
                  "Established", "Authored"],
    "engineered": ["Built", "Developed", "Constructed", "Implemented", "Assembled",
                   "Designed", "Produced"],
    "implemented": ["Built", "Developed", "Deployed", "Delivered", "Introduced",
                    "Rolled out", "Established"],
    "designed": ["Architected", "Structured", "Formulated", "Devised", "Planned"],
    "architected": ["Designed", "Structured", "Devised", "Planned"],
    "utilized": ["Used", "Applied", "Leveraged", "Employed"],
    # analysis
    "analyzed": ["Examined", "Assessed", "Investigated", "Evaluated", "Studied",
                 "Diagnosed", "Profiled"],
    "evaluated": ["Assessed", "Benchmarked", "Analyzed", "Examined", "Compared",
                  "Reviewed"],
    "assessed": ["Evaluated", "Analyzed", "Examined"],
    "benchmarked": ["Evaluated", "Compared", "Assessed"],
    "quantified": ["Measured", "Isolated", "Estimated"],
    "measured": ["Quantified", "Assessed"],
    # improvement / automation
    "automated": ["Streamlined", "Systematized", "Productionized", "Scripted",
                  "Mechanized"],
    "optimized": ["Improved", "Tuned", "Refined", "Strengthened", "Enhanced"],
    "improved": ["Optimized", "Strengthened", "Refined", "Enhanced", "Raised"],
    "streamlined": ["Automated", "Simplified", "Consolidated"],
    "productionized": ["Automated", "Operationalized", "Deployed"],
    "reduced": ["Cut", "Lowered", "Decreased"],
    # modelling
    "modeled": ["Formulated", "Specified"],
    "formulated": ["Modeled", "Specified", "Defined"],
    "trained": ["Fit", "Tuned", "Calibrated"],
    # delivery / communication
    "presented": ["Delivered", "Briefed", "Communicated"],
    "delivered": ["Presented", "Shipped", "Provided"],
    "linked": ["Joined", "Matched", "Connected"],
    "transformed": ["Converted", "Reshaped", "Turned"],
    "instrumented": ["Wired", "Equipped", "Fitted"],
}

_ITEM_VERB = re.compile(r"(\\item\s+)([A-Z][a-z]+)(\s)")


def dedupe_starting_verbs_in_tex(tex: str) -> tuple[str, List[str]]:
    """Ensure every bullet in the document starts with a distinct verb.

    Runs on the assembled .tex rather than on generated bullets, so it also covers static
    content the generator never sees -- academic-project bullets come verbatim from
    proj_academic_2-2.json and are otherwise never verb-checked.

    Deterministic: no model call. Only the first word is touched and only swapped for a
    same-meaning alternative, so no claim in the bullet can change.

    Two passes matter here. A single incremental pass will happily rewrite "Built" to
    "Developed" without knowing that a later bullet already starts with "Developed",
    trading one collision for another. So every verb in the document is collected first,
    and replacements are only drawn from outside that whole set.
    """
    matches = list(_ITEM_VERB.finditer(tex))
    if not matches:
        return tex, []

    all_verbs = [m.group(2) for m in matches]
    taken = {v.lower() for v in all_verbs}
    changes: List[str] = []
    seen: set[str] = set()
    replacements: dict[int, str] = {}

    for i, verb in enumerate(all_verbs):
        key = verb.lower()
        if key not in seen:
            seen.add(key)
            continue
        for candidate in VERB_ALTERNATIVES.get(key, []):
            if candidate.lower() not in taken:
                replacements[i] = candidate
                taken.add(candidate.lower())
                seen.add(candidate.lower())
                changes.append(f"{verb} -> {candidate}")
                break
        # No free alternative: leave it rather than force an awkward verb.

    if not replacements:
        return tex, []

    out: List[str] = []
    cursor = 0
    for i, match in enumerate(matches):
        if i not in replacements:
            continue
        out.append(tex[cursor : match.start()])
        out.append(f"{match.group(1)}{replacements[i]}{match.group(3)}")
        cursor = match.end()
    out.append(tex[cursor:])
    return "".join(out), changes


# ---------------------------------------------------------------------------
# Deterministic orphan repair
# ---------------------------------------------------------------------------

# Meaning-preserving contractions, ordered cheapest-first. Each removes characters without
# dropping a fact -- the same edits made by hand when a bullet spills two words onto a third
# line ("for" -> "on", dropping "by" before a percentage).
SAFE_SHORTENINGS: List[tuple[str, str]] = [
    (r"\bin order to\b", "to"),
    (r"\bas well as\b", "and"),
    (r"\bwhich (?:also )?(?:enabled|allowed|helped)\b", "to"),
    (r"\bwas able to\b", "could"),
    (r"\bapproximately\b", "about"),
    (r"\bin addition to\b", "besides"),
    (r"\bwith the goal of\b", "to"),
    (r"\bresulting in\b", "yielding"),
    (r"\bthereby\b", ""),
    (r"\bsuccessfully\b", ""),
    (r"\bsignificantly\b", ""),
    (r"\bsubstantially\b", ""),
    (r"\beffectively\b", ""),
    (r"\bfurther\b", ""),
    (r"\bvarious\b", ""),
    (r"\bcomprehensive\b", ""),
    (r"\bincreasing by\b", "increasing"),
    (r"\breducing by\b", "reducing"),
    (r"\bcutting by\b", "cutting"),
    (r"\bimproving by\b", "improving"),
    (r"\bacross a total of\b", "across"),
    (r"\ba total of\b", ""),
    (r"\bin terms of\b", "in"),
    (r"\border to\b", "to"),
    (r"\bhelped to\b", "helped"),
    (r"\bserved to\b", ""),
    (r"\bthat were\b", ""),
    (r"\bthat was\b", ""),
    (r"\bwhich were\b", ""),
    (r"\bwhich was\b", ""),
]

_ITEM_LINE = re.compile(r"^(\s*\\item\s+)(.*)$", re.MULTILINE)


def _normalise_for_match(text: str) -> str:
    """Strip LaTeX markup and collapse whitespace, so rendered text can be matched to source."""
    out = re.sub(r"\\[a-zA-Z]+\s*", "", text)
    out = out.replace("\\", "").replace("{", "").replace("}", "").replace("$", "")
    out = out.replace("~", " ").replace("&", "&")
    return re.sub(r"\s+", " ", out).strip().lower()


def shorten_bullet(text: str, target_reduction: int) -> tuple[str, bool]:
    """Apply safe contractions until the text loses at least `target_reduction` characters."""
    original_len = len(text)
    out = text
    for pattern, replacement in SAFE_SHORTENINGS:
        if original_len - len(out) >= target_reduction:
            break
        candidate = re.sub(pattern, replacement, out, count=1, flags=re.IGNORECASE)
        candidate = re.sub(r"\s{2,}", " ", candidate).replace(" ,", ",").replace(" .", ".")
        if candidate != out:
            out = candidate
    return out.strip(), (original_len - len(out)) >= target_reduction


def repair_orphans_in_tex(
    tex: str,
    pdf_path: Path,
    fallback_shortener: Optional[Callable[[str, int], str]] = None,
) -> tuple[str, List[str]]:
    """Rewrite bullets whose last rendered line is an orphan, so they fit one line fewer.

    Two stages. Deterministic contractions from SAFE_SHORTENINGS run first and handle any
    bullet carrying filler. A bullet that is already tight has nothing safe to remove, so
    `fallback_shortener(text, max_chars)` is called if supplied -- see
    resume_bullet_workflow.shorten_bullet_llm, which verifies the rewrite before accepting
    it. Without a fallback the bullet is left alone rather than mangled, and the QA report
    still flags it.
    """
    bullets = measure_bullets(pdf_path)
    width = column_width(bullets)
    orphans = find_orphans(bullets, col_width=width)
    if not orphans:
        return tex, []

    wanted = {_normalise_for_match(b.text)[:40]: b for b in orphans}
    changes: List[str] = []

    def _maybe_shorten(match: re.Match[str]) -> str:
        prefix, body = match.group(1), match.group(2)
        key = _normalise_for_match(body)[:40]
        target = wanted.get(key)
        if target is None:
            return match.group(0)
        # Drop the orphan line plus a little slack so it comfortably clears the boundary.
        needed = target.last_line_len + 8
        shortened, achieved = shorten_bullet(body, needed)
        if not achieved and fallback_shortener is not None:
            budget = max(len(body) - needed, 60)
            rewritten = fallback_shortener(body, budget)
            if rewritten and rewritten != body and len(rewritten) <= len(shortened):
                shortened = rewritten
                achieved = len(body) - len(rewritten) >= needed
        if shortened == body:
            return match.group(0)
        changes.append(
            f"shortened bullet #{target.index + 1} by {len(body) - len(shortened)} chars"
            + ("" if achieved else " (partial)")
        )
        return f"{prefix}{shortened}"

    return _ITEM_LINE.sub(_maybe_shorten, tex), changes
