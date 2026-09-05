"""Cover letter generation, sharing the resume pipeline's evidence and grounding.

Everything upstream of drafting is reused from resume_bullet_workflow: the JD
signals, the evidence loader, the LLM call and its backoff. What is new here is
the drafting step and — the part that matters — a validator built for prose.

Why prose needs its own validator: the bullet rules do not transfer. A bullet is
checked on length (200-240 chars), unique starting verbs and numeric grounding.
Of those only grounding applies to a paragraph, and prose makes fabrication
*easier* than bullets do. A bullet that invents a metric looks conspicuous; a
sentence that invents an enthusiasm ("I have long admired your work on X") reads
perfectly natural and is just as false. So the validator here refuses rather
than repairs, and the highest-consequence rule is the placeholder rule: a draft
containing "[Company]" that gets trimmed instead of rejected is a draft that
gets submitted with "[Company]" in it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from main_code.resume_bullet_workflow import (
    DEFAULT_MODEL,
    _numeric_claims,
    _evidence_numeric_claims,
    call_llm,
    get_task_llm_settings,
    write_prompt_log,
)
from main_code.workflow_prompts import (
    build_cover_letter_prompts,
    build_cover_letter_repair_prompts,
)

LLM_TASK_COVER_LETTER = "cover_letter"
LLM_TASK_COVER_LETTER_REPAIR = "cover_letter_repair"

MAX_COVER_LETTER_PARAGRAPHS = 4
MIN_COVER_LETTER_PARAGRAPHS = 2
MIN_COVER_LETTER_WORDS = 180
MAX_COVER_LETTER_WORDS = 380
MAX_COVER_LETTER_ATTEMPTS = 2

# Placeholders, in the shapes models actually emit. Deliberately broad: the cost
# of a false positive is one repair round, the cost of a false negative is a
# submitted application reading "Dear [Hiring Manager]".
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\[[^\]\n]{0,60}\]"),           # [Company], [insert metric]
    re.compile(r"\{\{[^}\n]{0,60}\}\}"),        # {{role}}
    re.compile(r"\{[A-Za-z_][^}\n]{0,60}\}"),   # {company_name}
    re.compile(r"<[A-Za-z_][^>\n]{0,60}>"),     # <YOUR NAME>
    re.compile(r"\b(?:TBD|TODO|FIXME|XXX+|LOREM IPSUM)\b", re.IGNORECASE),
    re.compile(r"\b(?:insert|add|your)\s+(?:company|name|role|title|metric)\b", re.IGNORECASE),
)

# Years are not claims about the candidate's results, and a letter legitimately
# names the current year or a degree date. Excluded from grounding so the check
# stays about invented metrics.
_YEAR = re.compile(r"^(?:19|20)\d\d$")

# Openers that assert a motivation nobody supplied. These are the prose analogue
# of an invented metric and the model reaches for them unprompted.
_INVENTED_AFFINITY = re.compile(
    r"\b(?:"
    r"long(?:\s+time)?\s+admire[rd]?|have\s+admired|always\s+admired|"
    r"since\s+(?:childhood|i\s+was\s+(?:a\s+)?(?:child|young|kid))|"
    r"lifelong\s+(?:fan|dream|passion)|dream\s+(?:job|company)|"
    r"avid\s+(?:user|reader|follower)|"
    r"ever\s+since\s+i\s+(?:first\s+)?(?:used|discovered|encountered)"
    r")\b",
    re.IGNORECASE,
)


def parse_cover_letter(raw: str) -> List[str]:
    """Split a raw reply into body paragraphs.

    Strips the salutation, date and sign-off the template supplies, because the
    model adds them anyway however firmly it is told not to, and a letter with
    two "Dear Hiring Manager" lines is a letter nobody sends.
    """
    text = raw.strip()

    # Drop a markdown fence if the model wrapped its answer in one.
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]

    salutation = re.compile(r"^(?:dear\b|to\s+whom\b|hello\b|hi\b|greetings\b)", re.IGNORECASE)
    signoff = re.compile(
        r"^(?:sincerely|regards|best(?:\s+regards)?|kind\s+regards|thank\s+you|yours\b|"
        r"respectfully|warmly)\b[,.]?\s*$",
        re.IGNORECASE,
    )
    datelike = re.compile(
        r"^(?:\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})$"
    )

    cleaned: List[str] = []
    for block in blocks:
        first_line = block.splitlines()[0].strip()
        if salutation.match(first_line) and len(block) < 120:
            continue
        if datelike.match(first_line):
            continue
        if signoff.match(first_line):
            # A sign-off block ends the letter; anything after it is a name.
            break
        cleaned.append(block)

    # A trailing paragraph that is only a sign-off plus a name (no blank line
    # between them) still needs its tail removed.
    if cleaned:
        lines = cleaned[-1].splitlines()
        while lines and (signoff.match(lines[-1].strip()) or len(lines[-1].strip().split()) <= 4):
            if len(lines) == 1:
                break
            if signoff.match(lines[-1].strip()) or signoff.match(lines[-2].strip() if len(lines) > 1 else ""):
                lines.pop()
                continue
            break
        cleaned[-1] = "\n".join(lines).strip()
        if not cleaned[-1]:
            cleaned.pop()

    # Collapse hard-wrapped lines inside a paragraph into one flowing line.
    return [re.sub(r"\s*\n\s*", " ", block).strip() for block in cleaned if block.strip()]


# Spelled-out magnitudes, checked separately from bare numbers. The shared
# _numeric_claims regex treats a trailing "m"/"k"/"b" as a magnitude suffix even
# when it is the first letter of an ordinary word, so "3.1 minutes" and
# "3.1 million" both reduce to the token "3.1M" — meaning evidence about minutes
# would license a letter about millions. Harmless in a bullet, where the unit sits
# right beside the figure and a human reads one line; not harmless in a paragraph.
_MAGNITUDE_PHRASE = re.compile(
    r"(\d[\d,.]*)\s*(million|billion|thousand|trillion)\b", re.IGNORECASE
)


def _magnitude_phrases(text: str) -> set[str]:
    """{'3.1 million'} — number and spelled-out magnitude as one claim."""
    return {
        f"{m.group(1).replace(',', '').rstrip('.')} {m.group(2).lower()}"
        for m in _MAGNITUDE_PHRASE.finditer(text)
    }


def _corpus(
    evidence: Dict[str, List[Dict[str, Any]]],
    jd_text: str,
    extra: Sequence[str] = (),
) -> str:
    parts: List[str] = [jd_text, *extra]
    for projects in evidence.values():
        for project in projects:
            for value in project.values():
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, (list, tuple)):
                    parts.extend(str(v) for v in value)
    return " ".join(parts)


def _allowed_claims(
    evidence: Dict[str, List[Dict[str, Any]]],
    jd_text: str,
    extra: Sequence[str] = (),
) -> Dict[str, set[str]]:
    """Every figure the letter is permitted to state.

    The evidence, plus the JD itself — quoting the posting's own figures back at
    it ("your 40-million-rider network") is legitimate and common — plus anything
    the caller vouches for, such as a GPA from the profile.
    """
    numbers: set[str] = set()
    for projects in evidence.values():
        numbers |= _evidence_numeric_claims(projects)
    numbers |= _numeric_claims(jd_text)
    for value in extra:
        numbers |= _numeric_claims(value)
    return {
        "numbers": numbers,
        "magnitudes": _magnitude_phrases(_corpus(evidence, jd_text, extra)),
    }


def validate_cover_letter(
    paragraphs: List[str],
    company_name: str,
    position_name: str,
    allowed: Dict[str, set[str]],
) -> List[str]:
    """Every reason this draft must not be used, as plain sentences.

    Returns an empty list for a usable draft. Callers treat a non-empty list as
    a refusal, not a warning — see the module docstring.
    """
    issues: List[str] = []
    body = "\n\n".join(paragraphs)

    if not paragraphs:
        return ["The reply contained no letter body at all."]

    if len(paragraphs) < MIN_COVER_LETTER_PARAGRAPHS:
        issues.append(
            f"A cover letter needs at least {MIN_COVER_LETTER_PARAGRAPHS} paragraphs "
            f"(got {len(paragraphs)})."
        )
    if len(paragraphs) > MAX_COVER_LETTER_PARAGRAPHS:
        issues.append(
            f"Use at most {MAX_COVER_LETTER_PARAGRAPHS} paragraphs (got {len(paragraphs)})."
        )

    words = len(body.split())
    if words < MIN_COVER_LETTER_WORDS:
        issues.append(f"The letter is too short ({words} words, minimum {MIN_COVER_LETTER_WORDS}).")
    if words > MAX_COVER_LETTER_WORDS:
        issues.append(f"The letter is too long ({words} words, maximum {MAX_COVER_LETTER_WORDS}).")

    # The rule that matters most: a placeholder that survives review is a
    # placeholder that gets submitted, so it fails the draft outright.
    for pattern in _PLACEHOLDER_PATTERNS:
        found = pattern.findall(body)
        if found:
            unique = sorted({f.strip() for f in found})[:5]
            issues.append(
                "The letter still contains unfilled placeholders: "
                + ", ".join(repr(f) for f in unique)
                + ". Rewrite those sentences so they need no placeholder."
            )
            break

    if company_name and company_name.lower() not in body.lower():
        issues.append(
            f"The letter never names the employer ({company_name!r}). A letter that does "
            "not name the company reads as a template."
        )

    # Grounding, the bullets' rule applied to prose. Years are exempt.
    stated_magnitudes = _magnitude_phrases(body)
    inflated = sorted(stated_magnitudes - allowed.get("magnitudes", set()))
    if inflated:
        issues.append(
            "The letter states magnitudes that appear nowhere in the evidence or the "
            "job description: " + ", ".join(inflated) + "."
        )

    stated = {n for n in _numeric_claims(body) if not _YEAR.match(n)}
    invented = sorted(stated - allowed.get("numbers", set()))
    if invented:
        issues.append(
            "The letter states figures that appear in neither the evidence nor the job "
            "description: " + ", ".join(invented) + ". Remove them or use a figure that "
            "does appear in the evidence."
        )

    affinity = _INVENTED_AFFINITY.search(body)
    if affinity:
        issues.append(
            f"The letter asserts a personal history nobody supplied ({affinity.group(0)!r}). "
            "Ground interest in what the job description says the work is."
        )

    return issues


def generate_cover_letter(
    company_name: str,
    position_name: str,
    jd_text: str,
    jd_signals: Dict[str, Any],
    evidence: Dict[str, List[Dict[str, Any]]],
    model: str = DEFAULT_MODEL,
    education: List[str] | None = None,
    extra_allowed_numbers: Sequence[str] = (),
    log_prompts: bool = False,
) -> Tuple[List[str], List[str]]:
    """Draft a cover letter, with one repair round.

    Returns (paragraphs, issues). An empty issues list means the draft passed
    every check. A non-empty one means it did not, and the paragraphs are
    returned anyway so the caller can show the human what was rejected and why —
    but they must not be presented as a usable letter.
    """
    task_model, task_temperature = get_task_llm_settings(
        LLM_TASK_COVER_LETTER, fallback_model=model
    )
    allowed = _allowed_claims(evidence, jd_text, extra_allowed_numbers)

    system_prompt, user_prompt = build_cover_letter_prompts(
        company_name=company_name,
        position_name=position_name,
        jd_text=jd_text,
        jd_signals=jd_signals,
        evidence=evidence,
        education=education,
        max_paragraphs=MAX_COVER_LETTER_PARAGRAPHS,
        min_words=MIN_COVER_LETTER_WORDS,
        max_words=MAX_COVER_LETTER_WORDS,
    )
    if log_prompts:
        write_prompt_log(system_prompt, {"payload": user_prompt}, "cover_letter")

    raw = call_llm(
        model=task_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=task_temperature,
        max_tokens=2048,
    )
    paragraphs = parse_cover_letter(raw)
    issues = validate_cover_letter(paragraphs, company_name, position_name, allowed)
    if not issues:
        return paragraphs, []

    # One repair round, exactly as the bullet path gets. A second would mostly
    # produce a differently-broken letter for another minute of model time.
    repair_model, repair_temperature = get_task_llm_settings(
        LLM_TASK_COVER_LETTER_REPAIR, fallback_model=model
    )
    repair_system, repair_user = build_cover_letter_repair_prompts(
        draft="\n\n".join(paragraphs),
        issues=issues,
        company_name=company_name,
        position_name=position_name,
        evidence=evidence,
    )
    if log_prompts:
        write_prompt_log(repair_system, {"payload": repair_user}, "cover_letter_repair")

    try:
        repaired_raw = call_llm(
            model=repair_model,
            messages=[
                {"role": "system", "content": repair_system},
                {"role": "user", "content": repair_user},
            ],
            temperature=repair_temperature,
            max_tokens=2048,
        )
    except Exception:
        # The first draft plus its issues is more useful than an exception here;
        # the caller has to refuse either way.
        return paragraphs, issues

    repaired = parse_cover_letter(repaired_raw)
    repaired_issues = validate_cover_letter(repaired, company_name, position_name, allowed)
    if not repaired_issues:
        return repaired, []

    # Both failed. Return whichever failed less, and every reason.
    if len(repaired_issues) < len(issues):
        return repaired, repaired_issues
    return paragraphs, issues


def evidence_for_letter(data_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """The same work_*.json evidence the resume bullets are drawn from."""
    from main_code.build_resume import _evidence_by_company

    return _evidence_by_company(data_dir)
