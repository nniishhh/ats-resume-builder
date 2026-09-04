"""Build a tailored resume: generate ATS-optimized bullets via the workflow,
inject them into main.tex, and compile to PDF.

Output naming: oranich_resume_{company}_{position}.tex / .pdf
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# When run as `python main_code/build_resume.py`, ensure imports resolve to
# the local project source (repo root), not an older installed package copy.
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

from main_code import layout, qa_report
from main_code.resume_bullet_workflow import (
    select_skills_for_jd,
    shorten_bullet_llm,
    DEFAULT_GENERATION_MODE,
    DEFAULT_MODEL,
    GENERATION_MODES,
    summarize_job_description,
    run_all_with_full_selection,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _parse_header_body(content: str) -> dict | None:
    """Parse a header+body JD format separated by '---'.

    Expected format::

        company_name: Acme
        position_name: SWE Intern
        ---
        Full job description text …

    Returns dict with parsed keys or None if the format doesn't match.
    """
    if "---" not in content:
        return None
    header, _, body = content.partition("---")
    meta: dict[str, str] = {}
    for line in header.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    if not meta:
        return None
    meta["job_description"] = body.strip()
    return meta


def parse_jd_metadata(
    jd_path: Path,
    model: str,
) -> Tuple[str, str, str]:
    """Return (company_name, position_name, raw_jd_text) from JD file.

    Supported formats:
      1. JSON  – ``{"company_name": …, "position_name": …, "job_description": …}``
      2. Header+body – key: value lines, then ``---``, then the JD body.
      3. Plain text – entire file is the JD.
    """
    content = jd_path.read_text(encoding="utf-8").strip()
    company_name = "unknown"
    position_name = ""
    jd_text = content

    # 1. Try JSON
    try:
        data = json.loads(content)
        company_name = data.get("company_name", "unknown")
        position_name = data.get("position_name", "")
        jd_text = data.get("job_description", content)
    except json.JSONDecodeError:
        # 2. Try header+body format
        parsed = _parse_header_body(content)
        if parsed:
            company_name = parsed.get("company_name", "unknown")
            position_name = parsed.get("position_name", "")
            jd_text = parsed.get("job_description", content)

    if not position_name:
        print("[info] position_name not in JD file; extracting via LLM …", file=sys.stderr)
        jd_summary = summarize_job_description(jd_text=jd_text, model=model)
        position_name = infer_position_name_from_jd(jd_summary)
        print(f"[info] Extracted position: {position_name}", file=sys.stderr)

    return company_name, position_name, jd_text


def infer_position_name_from_jd(jd_text: str) -> str:
    """Best-effort title extraction from JD text."""
    title_pattern = re.compile(
        r"(?i)\b(?:position|role|title)\s*[:\-]\s*([A-Za-z0-9/,&()\- ]+)"
    )
    match = title_pattern.search(jd_text)
    if match:
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" -:")
        if candidate:
            return candidate[:80]

    role_keywords = (
        "engineer",
        "scientist",
        "analyst",
        "manager",
        "intern",
        "consultant",
        "developer",
        "specialist",
    )
    for line in jd_text.splitlines():
        candidate = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        lower = candidate.lower()
        if not candidate or len(candidate.split()) > 14:
            continue
        if any(keyword in lower for keyword in role_keywords):
            return candidate[:80]

    return "unknown"


def slugify(text: str) -> str:
    """Convert text to a lowercase filename-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in plain-text bullet content."""
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("$", "\\$")
    text = text.replace("#", "\\#")
    text = text.replace("_", "\\_")
    return text


def _match_company_key(
    display_name: str, available_keys: List[str]
) -> Optional[str]:
    """Match a TeX display name to a workflow company key.

    Example: 'SWAT Mobility (AI-Driven …)' → 'swat_mobility'

    Prefers longer key matches to avoid ambiguity (e.g. 'swat_mobility'
    is preferred over 'swat' when both exist).
    """
    display_lower = display_name.lower()
    best_key: Optional[str] = None
    best_len = 0
    for key in available_keys:
        search = key.replace("_", " ")
        if search in display_lower and len(search) > best_len:
            best_key = key
            best_len = len(search)
    return best_key


def replace_experience_bullets(
    tex_content: str, bullets_by_company: Dict[str, List[str]]
) -> str:
    """Replace \\item lines inside the Experience section with new bullets."""
    lines = tex_content.split("\n")
    result: List[str] = []

    in_experience = False
    in_itemize = False
    replacing = False
    current_key: Optional[str] = None
    remaining_keys = set(bullets_by_company.keys())

    for line in lines:
        stripped = line.strip()

        # ── section tracking ────────────────────────────────────────
        if r"\section{Experience}" in line:
            in_experience = True
            result.append(line)
            continue

        if in_experience and stripped.startswith(r"\section{"):
            in_experience = False
            result.append(line)
            continue

        if not in_experience:
            result.append(line)
            continue

        # ── inside Experience section ───────────────────────────────

        # Detect company header
        bf_match = re.search(r"\\textbf\{([^}]+)\}", line)
        if bf_match:
            current_key = _match_company_key(
                bf_match.group(1), list(bullets_by_company.keys())
            )

        # Start of itemize block
        if r"\begin{itemize}" in stripped:
            in_itemize = True
            replacing = (
                current_key is not None and current_key in bullets_by_company
            )
            result.append(line)
            if replacing:
                for bullet in bullets_by_company[current_key]:
                    escaped = escape_latex(bullet)
                    result.append(f"    \\item {escaped}")
                remaining_keys.discard(current_key)
            continue

        # Inside itemize block
        if in_itemize:
            if r"\end{itemize}" in stripped:
                in_itemize = False
                replacing = False
                result.append(line)
                continue
            if replacing:
                continue  # skip old bullet lines
            result.append(line)
            continue

        result.append(line)

    if remaining_keys:
        print(
            f"[warning] No matching TeX section for: {remaining_keys}",
            file=sys.stderr,
        )

    return "\n".join(result)


def replace_columbia_coursework(tex_content: str, courses: List[str]) -> str:
    """Replace the Columbia coursework list in the Education section."""
    if not courses:
        return tex_content

    escaped_courses = [escape_latex(course) for course in courses if course.strip()]
    if not escaped_courses:
        return tex_content
    coursework_text = ", ".join(escaped_courses)

    columbia_anchor = r"\textbf{Columbia University}"
    start_idx = tex_content.find(columbia_anchor)
    if start_idx == -1:
        print(
            "[warning] Columbia section not found; coursework replacement skipped.",
            file=sys.stderr,
        )
        return tex_content

    next_section_idx = tex_content.find(r"\section{", start_idx + 1)
    end_idx = next_section_idx if next_section_idx != -1 else len(tex_content)
    section = tex_content[start_idx:end_idx]

    def _replace_coursework(match: re.Match[str]) -> str:
        return f"{match.group(1)}{coursework_text}"

    updated_section, changed = re.subn(
        r"(Coursework:\s*)([^\n]+)",
        _replace_coursework,
        section,
        count=1,
    )
    if changed == 0:
        print(
            "[warning] Coursework field not found in Columbia section; replacement skipped.",
            file=sys.stderr,
        )
        return tex_content

    return tex_content[:start_idx] + updated_section + tex_content[end_idx:]


def replace_academic_projects(
    tex_content: str,
    selected_projects: List[Dict[str, Any]],
) -> str:
    """Replace the Academic Projects section using selected project JSON records."""
    def _normalize_project_link(raw_link: Any) -> str:
        link = str(raw_link or "").strip()
        if not link or link.lower() in {"n/a", "na", "none", "null", "-"}:
            return ""
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", link):
            link = f"https://{link}"
        return link

    valid_projects: List[Dict[str, Any]] = []
    for project in selected_projects:
        topic = str(project.get("Topic", "")).strip()
        bullets = [
            str(item).strip() for item in project.get("Bullet", []) if str(item).strip()
        ]
        link = _normalize_project_link(project.get("Link", ""))
        if topic and bullets:
            valid_projects.append({"Topic": topic, "Bullet": bullets, "Link": link})
    if not valid_projects:
        return tex_content

    section_anchor = r"\section{Academic Projects}"
    start_idx = tex_content.find(section_anchor)
    if start_idx == -1:
        print(
            "[warning] Academic Projects section not found; replacement skipped.",
            file=sys.stderr,
        )
        return tex_content

    next_section_idx = tex_content.find(r"\section{", start_idx + len(section_anchor))
    end_idx = next_section_idx if next_section_idx != -1 else len(tex_content)

    lines: List[str] = [section_anchor]
    for idx, project in enumerate(valid_projects):
        if idx > 0:
            lines.extend(["", r"\vspace{6pt}", ""])
        topic = escape_latex(project["Topic"])
        link = project.get("Link", "")
        if link:
            lines.append(
                rf"\noindent \textbf{{{topic}}} \hfill \href{{{link}}}{{\textit{{Project Link}}}}"
            )
        else:
            lines.append(rf"\noindent \textbf{{{topic}}}")
        lines.append(r"\begin{itemize}")
        for bullet in project["Bullet"]:
            lines.append(f"    \\item {escape_latex(bullet)}")
        lines.append(r"\end{itemize}")

    new_section = "\n".join(lines) + "\n\n"
    return tex_content[:start_idx] + new_section + tex_content[end_idx:]


def drop_experience_entries(tex_content: str, drop_keys: List[str]) -> str:
    """Remove whole Experience entries for roles the structure plan dropped.

    replace_experience_bullets() only rewrites companies it is handed, so a dropped
    role silently kept the template's original placeholder bullets: the build report
    said "dropped shopee" while the PDF still carried Shopee with untailored text.
    Skipping generation is not the same as removing the entry, so remove it here.
    """
    if not drop_keys:
        return tex_content

    start = tex_content.find(r"\section{Experience}")
    if start == -1:
        return tex_content
    rest = tex_content.find(r"\section{", start + 1)
    end = rest if rest != -1 else len(tex_content)

    section = tex_content[start:end]
    # Each entry runs from its bold company heading to just before the next one.
    heads = [m.start() for m in re.finditer(r"\\noindent \\textbf\{", section)]
    if not heads:
        return tex_content

    keep: List[str] = []
    cursor = 0
    for i, head in enumerate(heads):
        stop = heads[i + 1] if i + 1 < len(heads) else len(section)
        entry = section[head:stop]
        name_match = re.search(r"\\textbf\{([^}]+)\}", entry)
        key = _match_company_key(name_match.group(1), drop_keys) if name_match else None
        if key in drop_keys:
            keep.append(section[cursor:head])
            cursor = stop
    keep.append(section[cursor:])

    return tex_content[:start] + "".join(keep) + tex_content[end:]


def _plan_note(plan: Dict[str, Any]) -> str:
    """One-line summary of how the page budget was allocated."""
    roles = plan.get("roles") or {}
    kept = ", ".join(f"{c} {n}" for c, n in roles.items() if n)
    dropped = [c for c, n in roles.items() if not n]
    note = f"{kept}; {plan.get('projects')} projects, {plan.get('coursework')} courses"
    if dropped:
        note += f"; dropped {', '.join(dropped)}"
    return note


def tighten_spacing(tex: str) -> str:
    """Adjust spacing and typesetting for generated resumes.

    The template (main.tex) is tuned for shorter original bullets.
    Generated bullets can be up to MAX_BULLET_CHARS (240) characters,
    so we: (a) compress vertical gaps, (b) suppress mid-word hyphenation
    that looks unprofessional on a resume.
    """
    # ── improve line-breaking ───────────────────────────────────────
    # microtype enables character protrusion for cleaner justified text.
    # \sloppy + \hyphenpenalty discourages mid-word breaks.
    tex = tex.replace(
        r"\usepackage{fontspec}",
        r"\usepackage{fontspec}" "\n"
        r"\usepackage{microtype}",
    )
    tex = tex.replace(
        r"\begin{document}",
        r"\begin{document}" "\n"
        r"\sloppy\hyphenpenalty=10000\exhyphenpenalty=10000",
    )

    # ── tighten margins and vertical spacing ────────────────────────
    # Reduce margins from 0.6in to 0.5in (standard for resumes)
    tex = tex.replace(
        "left=0.6in, right=0.6in, top=0.6in, bottom=0.6in",
        "left=0.5in, right=0.5in, top=0.5in, bottom=0.5in",
    )
    # Section title: reduce space before (12→8pt) and after (6→4pt)
    tex = tex.replace(
        r"\titlespacing{\section}{0pt}{12pt}{6pt}",
        r"\titlespacing{\section}{0pt}{8pt}{4pt}",
    )
    # Bullet list: tighten item gap
    tex = tex.replace(
        "itemsep=2pt, topsep=2pt, parsep=0pt",
        "itemsep=1pt, topsep=2pt, parsep=0pt",
    )
    # Shrink vertical gaps between entries
    tex = tex.replace(r"\vspace{6pt}", r"\vspace{4pt}")
    tex = tex.replace(r"\vspace{5pt}", r"\vspace{2pt}")
    return tex


def replace_skills(tex_content: str, categories: List[Dict[str, Any]]) -> str:
    """Rewrite the Skills section from tailored, whitelist-checked categories.

    Replaces every line between \\section{Skills} and \\end{document}. The template keeps an
    untailored fallback there so the document still compiles if skills selection is skipped.
    """
    if not categories:
        return tex_content

    rendered: List[str] = []
    for i, category in enumerate(categories):
        label = escape_latex(str(category.get("label", "Skills")))
        skills = ", ".join(escape_latex(str(s)) for s in category.get("skills", []))
        if not skills:
            continue
        prefix = r"\noindent " if i == 0 else ""
        suffix = r" \\" if i < len(categories) - 1 else ""
        rendered.append(f"{prefix}\\textbf{{{label}:}} {skills}.{suffix}")

    if not rendered:
        return tex_content

    pattern = re.compile(
        r"(\\section\{Skills\}\n)(.*?)(\n\\end\{document\})",
        re.DOTALL,
    )
    if not pattern.search(tex_content):
        print("[warning] Skills section anchor not found; leaving template skills.",
              file=sys.stderr)
        return tex_content
    return pattern.sub(lambda m: m.group(1) + "\n".join(rendered) + m.group(3), tex_content)


def fit_to_one_page(
    tex_path: Path,
    pdf_path: Path,
    max_passes: int = 10,
) -> Tuple[Path, List[str]]:
    """Shrink a multi-page resume back to one page, recording what it did.

    The template is tuned for shorter bullets than the generator produces, so overflow is
    routine. Previously nothing checked, and two-page PDFs shipped silently.

    Ladder, least destructive first — content is only touched after spacing is exhausted:
      1. tighten inter-entry vertical space
      2. tighten section spacing
      3. tighten margins
      4. drop the last bullet of the longest experience entry (repeats until it fits)

    Spacing is exhausted before any content is touched, and every action is recorded so the
    QA report can tell the user exactly what was removed.
    """
    actions: List[str] = []
    tex = tex_path.read_text(encoding="utf-8")

    ladder = [
        ("tightened entry spacing",
         [(r"\vspace{4pt}", r"\vspace{2pt}"), (r"\vspace{2pt}", r"\vspace{1pt}")]),
        ("tightened section spacing",
         [(r"\titlespacing{\section}{0pt}{8pt}{4pt}", r"\titlespacing{\section}{0pt}{5pt}{3pt}"),
          (r"\titlespacing{\section}{0pt}{5pt}{3pt}", r"\titlespacing{\section}{0pt}{4pt}{2pt}")]),
        ("tightened margins",
         [("top=0.5in, bottom=0.5in", "top=0.4in, bottom=0.4in")]),
    ]

    for _ in range(max_passes):
        if layout.page_count(pdf_path) == 1:
            break
        applied = False
        for label, replacements in ladder:
            if label in actions:
                continue
            changed = tex
            for old, new in replacements:
                changed = changed.replace(old, new)
            if changed != tex:
                tex = changed
                actions.append(label)
                applied = True
                break
        if not applied:
            dropped = _drop_last_bullet_of_longest_entry(tex)
            if dropped is None:
                break
            tex, note = dropped
            actions.append(note)
        tex_path.write_text(tex, encoding="utf-8")
        pdf_path = compile_to_pdf(tex_path)

    return pdf_path, actions


def polish_document(
    tex_path: Path,
    pdf_path: Path,
    model: str | None = None,
) -> Tuple[Path, List[str]]:
    """Close the two defects the QA report used to only describe.

    Runs after fit_to_one_page(), because both fixes depend on how the text actually
    rendered and both change length slightly:

      1. de-duplicate starting verbs across the whole document, including static
         academic-project bullets the generator never sees
      2. shorten bullets whose last line is an orphan, using meaning-preserving
         contractions only

    Both are deterministic. If a fix would push the document back to two pages, it is
    reverted rather than shipped.
    """
    actions: List[str] = []
    original_tex = tex_path.read_text(encoding="utf-8")
    tex = original_tex

    deduped, verb_changes = layout.dedupe_starting_verbs_in_tex(tex)
    if verb_changes:
        tex = deduped
        tex_path.write_text(tex, encoding="utf-8")
        pdf_path = compile_to_pdf(tex_path)
        actions.append("de-duplicated verbs (" + ", ".join(verb_changes) + ")")

    shortener = None
    if model:
        shortener = lambda text, budget: shorten_bullet_llm(  # noqa: E731
            bullet=text, max_chars=budget, model=model
        )
    # Orphan repair is empirical: shorten, recompile, re-measure. One pass can also create
    # a new orphan elsewhere by reflowing the page, so iterate until it is clean or stalls.
    for _ in range(3):
        repaired, orphan_changes = layout.repair_orphans_in_tex(
            tex, pdf_path, fallback_shortener=shortener
        )
        if not orphan_changes:
            break
        tex = repaired
        tex_path.write_text(tex, encoding="utf-8")
        pdf_path = compile_to_pdf(tex_path)
        actions.extend(orphan_changes)

    if actions and layout.page_count(pdf_path) > 1:
        tex_path.write_text(original_tex, encoding="utf-8")
        pdf_path = compile_to_pdf(tex_path)
        return pdf_path, ["polish reverted: fixes pushed the resume to two pages"]

    return pdf_path, actions


def _drop_last_bullet_of_longest_entry(tex: str) -> Optional[Tuple[str, str]]:
    """Remove the final \\item from whichever itemize block has the most bullets."""
    blocks = list(re.finditer(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", tex, re.DOTALL))
    if not blocks:
        return None
    target = max(blocks, key=lambda m: m.group(1).count(r"\item"))
    items = re.findall(r"^\s*\\item .*$", target.group(1), re.MULTILINE)
    if len(items) <= 1:
        return None
    body = target.group(1).replace(items[-1] + "\n", "").replace(items[-1], "")
    updated = tex[: target.start(1)] + body + tex[target.end(1) :]
    return updated, "dropped 1 bullet from the longest entry"


def _find_tex_compiler() -> Optional[List[str]]:
    """Return the command list for the first available TeX compiler, or None."""
    candidates = [
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error"],
        ["tectonic"],
    ]
    for cmd in candidates:
        if shutil.which(cmd[0]):
            return cmd
    return None


def compile_to_pdf(tex_path: Path) -> Path:
    """Compile a .tex file to PDF using the first available TeX compiler."""
    compiler = _find_tex_compiler()
    if compiler is None:
        raise RuntimeError(
            "No TeX compiler found (tried xelatex, tectonic).\n"
            "Install one of:\n"
            "  brew install --cask mactex        # full MacTeX\n"
            "  brew install --cask basictex       # minimal\n"
            "  brew install tectonic              # lightweight alternative"
        )

    cmd = compiler + [tex_path.name]
    print(f"[info] Compiling with: {compiler[0]}", file=sys.stderr)
    result = subprocess.run(
        cmd,
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log_lines = result.stdout.strip().splitlines()[-30:]
        print(f"[error] {compiler[0]} failed. Last log lines:", file=sys.stderr)
        for ln in log_lines:
            print(f"  {ln}", file=sys.stderr)
        raise RuntimeError(f"{compiler[0]} compilation failed.")

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"PDF was not created: {pdf_path}")
    return pdf_path


def cleanup_aux_files(tex_path: Path) -> None:
    """Remove auxiliary files created by xelatex."""
    for ext in (".aux", ".log", ".out"):
        aux = tex_path.with_suffix(ext)
        if aux.exists():
            aux.unlink()


# ── main pipeline ────────────────────────────────────────────────────────────


def _evidence_by_company(data_dir: Path) -> Dict[str, List[dict]]:
    """Load work_*.json keyed by the same company keys the bullets use.

    Used by the grounding check to verify every number in a generated bullet appears
    somewhere in that company's evidence.
    """
    evidence: Dict[str, List[dict]] = {}
    for path in sorted(data_dir.glob("work_*.json")):
        match = re.match(r"^work_(?P<company>.+)_\d+-\d+\.json$", path.name)
        if not match:
            continue
        try:
            evidence[match.group("company")] = json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            continue
    return evidence


def build_resume(
    jd_path: Path,
    template_path: Path,
    output_dir: Path,
    model: str,
    log_prompts: bool,
    generation_mode: str = DEFAULT_GENERATION_MODE,
    no_compile: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """End-to-end: generate bullets -> inject into TeX -> compile PDF.

    Returns (tex_path, pdf_path_or_None).
    """
    # 1. Parse JD metadata
    company_name, position_name, jd_text = parse_jd_metadata(
        jd_path=jd_path,
        model=model,
    )
    print(
        f"[info] Target: {company_name} — {position_name}",
        file=sys.stderr,
    )

    # 2. Generate bullets + JD-relevant coursework + top academic projects in parallel.
    # jd_signals (seniority/archetype) comes back from here too, since bullet generation
    # already needs it to calibrate jargon density — no need to extract it again below.
    (
        bullets,
        selected_courses,
        selected_topics,
        selected_academic_projects,
        jd_signals,
        plan,
    ) = run_all_with_full_selection(
        jd_path=jd_path,
        directory=jd_path.parent,
        model=model,
        log_prompts=log_prompts,
        generation_mode=generation_mode,
    )
    print(
        f"[info] Generated bullets for {len(bullets)} companies: "
        f"{list(bullets.keys())}",
        file=sys.stderr,
    )
    print(
        f"[info] Selected Columbia coursework: {selected_courses}",
        file=sys.stderr,
    )
    print(
        f"[info] Selected academic projects: {selected_topics}",
        file=sys.stderr,
    )

    # 2b. Tailored Skills from the JD signals already extracted above, whitelist-enforced in code
    skill_categories, dropped_skills = select_skills_for_jd(
        jd_signals=jd_signals, model=model
    )
    if dropped_skills:
        print(
            f"[warning] Dropped {len(dropped_skills)} skill(s) not in inventory: "
            f"{dropped_skills}",
            file=sys.stderr,
        )
    print(
        f"[info] Skills categories: {[c['label'] for c in skill_categories]}",
        file=sys.stderr,
    )

    # 3. Read template and replace bullets/coursework/academic projects/skills
    tex_content = template_path.read_text(encoding="utf-8")
    tex_content = drop_experience_entries(
        tex_content, [c for c, n in (plan.get("roles") or {}).items() if not n]
    )
    new_tex = replace_experience_bullets(tex_content, bullets)
    new_tex = replace_columbia_coursework(new_tex, selected_courses)
    new_tex = replace_academic_projects(new_tex, selected_academic_projects)
    new_tex = replace_skills(new_tex, skill_categories)

    # 3b. Tighten spacing so longer generated bullets still fit one page
    new_tex = tighten_spacing(new_tex)

    # 4. Write new .tex
    slug = f"oranich_resume_{slugify(company_name)}_{slugify(position_name)}"
    tex_out = output_dir / f"{slug}.tex"
    tex_out.write_text(new_tex, encoding="utf-8")
    print(f"[info] Wrote {tex_out}", file=sys.stderr)

    # 5. Compile, then fit to one page and report on what shipped
    report = qa_report.QAReport()
    report.skills_kept = sum(len(c["skills"]) for c in skill_categories)
    report.skills_dropped = dropped_skills

    if no_compile:
        return tex_out, None, report

    try:
        pdf_out = compile_to_pdf(tex_out)
        pdf_out, trim_actions = fit_to_one_page(tex_out, pdf_out)
        pdf_out, polish_actions = polish_document(tex_out, pdf_out, model=model)
        trim_actions = trim_actions + polish_actions
        cleanup_aux_files(tex_out)

        report.trim_actions = trim_actions
        report.plan_note = _plan_note(plan)
        qa_report.check_layout(pdf_out, report)
        qa_report.check_redundancy(bullets, report)
        qa_report.check_grounding(bullets, _evidence_by_company(jd_path.parent), report)
        qa_report.check_jd_coverage(
            jd_signals.get("must_have", []), layout.extract_layout_text(pdf_out), report
        )

        print(f"[info] Wrote {pdf_out}", file=sys.stderr)
        print("\n" + report.render(), file=sys.stderr)
        return tex_out, pdf_out, report
    except RuntimeError as exc:
        print(f"[warning] {exc}", file=sys.stderr)
        print(
            "[info] .tex file was written successfully. "
            "Compile it manually or upload to Overleaf.",
            file=sys.stderr,
        )
        return tex_out, None, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a tailored resume (generate bullets -> TeX -> PDF)."
    )
    parser.add_argument(
        "--jd",
        type=Path,
        default=Path("data/JD.txt"),
        help="Path to JD file (JSON with company_name, position_name, job_description).",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("data/main.tex"),
        help="Path to the LaTeX resume template.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for generated .tex/.pdf files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LITELLM_MODEL", DEFAULT_MODEL),
        help=f"LiteLLM model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--log-prompts",
        action="store_true",
        help="Write system/user prompts to output/prompt_logs/.",
    )
    parser.add_argument(
        "--generation-mode",
        type=str,
        choices=list(GENERATION_MODES),
        default=DEFAULT_GENERATION_MODE,
        help=(
            "How to generate bullets across companies: "
            "'single_prompt' (all companies in one prompt) or "
            "'sequential' (one company at a time with verb tracking)."
        ),
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip PDF compilation (only produce the .tex file).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tex_path, pdf_path, _report = build_resume(
            jd_path=args.jd,
            template_path=args.template,
            output_dir=output_dir,
            model=args.model,
            log_prompts=args.log_prompts,
            generation_mode=args.generation_mode,
            no_compile=args.no_compile,
        )
        print(f"\nResume ready:")
        print(f"   TeX: {tex_path}")
        if pdf_path:
            print(f"   PDF: {pdf_path}")
        else:
            print(f"   PDF: (not compiled — use xelatex or upload to Overleaf)")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
