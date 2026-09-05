"""Render and compile a cover letter, reusing the resume's LaTeX toolchain.

Shares compile_to_pdf, cleanup_aux_files, escape_latex and slugify with
build_resume, and layout.page_count for the one-page check. What it does NOT
share is the fit ladder: the resume's ladder drops the least relevant bullet when
the page overflows, which is a sensible thing to do to a list and a terrible
thing to do to a paragraph — silently deleting a sentence changes what the letter
claims. Length is controlled at generation time instead (a word cap the validator
enforces), and a letter that still overflows tightens spacing and then reports
rather than cutting text.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from main_code import layout
from main_code.build_resume import (
    cleanup_aux_files,
    compile_to_pdf,
    escape_latex,
    slugify,
)
from main_code.cover_letter_workflow import (
    evidence_for_letter,
    generate_cover_letter,
)
from main_code.resume_bullet_workflow import DEFAULT_MODEL, extract_jd_signals

TEMPLATE_NAME = "cover_letter.tex"

_MARKERS = ("<<<DATE>>>", "<<<COMPANY>>>", "<<<ROLE>>>", "<<<BODY>>>")


def _today() -> str:
    """Today's date, from LOCAL parts.

    Deliberately not a UTC conversion: an evening in New York is already
    tomorrow in UTC, and a letter dated tomorrow is a letter that looks
    machine-generated to the one person guaranteed to read it.
    """
    now = datetime.now()
    return f"{now:%B} {now.day}, {now.year}"


def render_cover_letter_tex(
    template: str,
    paragraphs: Sequence[str],
    company_name: str,
    position_name: str,
    date_line: str | None = None,
) -> str:
    """Fill the template. Raises if any marker survives.

    A leaked marker is the failure mode worth being loud about — the letter still
    compiles, still looks finished, and still says "<<<COMPANY>>>" halfway down.
    """
    if not paragraphs:
        raise ValueError("Refusing to render a cover letter with no body paragraphs.")

    body = "\n\n".join(escape_latex(p.strip()) for p in paragraphs if p.strip())

    tex = template
    tex = tex.replace("<<<DATE>>>", escape_latex(date_line or _today()))
    tex = tex.replace("<<<COMPANY>>>", escape_latex(company_name.strip()))
    tex = tex.replace("<<<ROLE>>>", escape_latex(position_name.strip()))
    tex = tex.replace("<<<BODY>>>", body)

    leaked = [m for m in _MARKERS if m in tex]
    if leaked:
        raise RuntimeError(
            "Cover letter template markers were not filled: " + ", ".join(leaked)
        )
    return tex


def _tighten_letter_spacing(tex: str) -> str:
    """One spacing step, for a letter that ran onto a second page.

    Only whitespace is touched — never the prose. If this is not enough the
    caller is told the page count rather than having a sentence removed.
    """
    return (
        tex.replace(r"\setlength{\parskip}{9pt}", r"\setlength{\parskip}{6pt}")
        .replace("top=0.7in, bottom=0.7in", "top=0.55in, bottom=0.55in")
        .replace(r"\vspace{10pt}", r"\vspace{6pt}")
    )


def compile_cover_letter(
    paragraphs: Sequence[str],
    company_name: str,
    position_name: str,
    data_dir: Path,
    output_dir: Path,
    date_line: str | None = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Render, compile, and keep it to one page without editing the prose.

    Returns (pdf_path, report).
    """
    template_path = data_dir / TEMPLATE_NAME
    if not template_path.exists():
        raise FileNotFoundError(f"{template_path} not found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    tex = render_cover_letter_tex(
        template_path.read_text(encoding="utf-8"),
        paragraphs,
        company_name,
        position_name,
        date_line,
    )

    stem = f"cover_letter_{slugify(company_name)}_{slugify(position_name)}" or "cover_letter"
    tex_path = output_dir / f"{stem}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    pdf_path = compile_to_pdf(tex_path)

    pages = layout.page_count(pdf_path)
    tightened = False
    if pages > 1:
        tex_path.write_text(_tighten_letter_spacing(tex), encoding="utf-8")
        pdf_path = compile_to_pdf(tex_path)
        pages = layout.page_count(pdf_path)
        tightened = True

    cleanup_aux_files(tex_path)

    report: Dict[str, Any] = {
        "pages": pages,
        "spacing_tightened": tightened,
        "word_count": sum(len(p.split()) for p in paragraphs),
        "paragraph_count": len(paragraphs),
        "tex_path": str(tex_path),
    }
    if pages > 1:
        # Reported, not fixed. Cutting a sentence to win a line is the resume's
        # trade-off, not a letter's.
        report["warning"] = (
            f"The letter is {pages} pages even after tightening spacing. "
            "Shorten the draft rather than letting it run over."
        )
    return pdf_path, report


def build_cover_letter(
    company_name: str,
    position_name: str,
    jd_text: str,
    data_dir: Path,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    education: List[str] | None = None,
    log_prompts: bool = False,
) -> Tuple[Path | None, List[str], Dict[str, Any]]:
    """End to end: JD -> signals -> draft -> validate -> PDF.

    Returns (pdf_path, paragraphs, report). pdf_path is None when the draft failed
    validation — nothing is compiled from a draft we would not send, so there is
    no rejected PDF lying around to be attached by mistake.
    """
    jd_signals = extract_jd_signals(jd_text, model)
    evidence = evidence_for_letter(data_dir)

    paragraphs, issues = generate_cover_letter(
        company_name=company_name,
        position_name=position_name,
        jd_text=jd_text,
        jd_signals=jd_signals,
        evidence=evidence,
        model=model,
        education=education,
        log_prompts=log_prompts,
    )

    report: Dict[str, Any] = {"jd_signals": jd_signals, "issues": issues}
    if issues:
        report["rejected"] = True
        return None, paragraphs, report

    pdf_path, compile_report = compile_cover_letter(
        paragraphs=paragraphs,
        company_name=company_name,
        position_name=position_name,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    report.update(compile_report)
    report["rejected"] = False
    return pdf_path, paragraphs, report
