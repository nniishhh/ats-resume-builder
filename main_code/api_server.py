"""Standalone (non-Streamlit) API + static frontend for the Resume Tailor.

Run locally:  uv run resume-web
Serves the app at http://localhost:8000 — a static HTML/JS frontend (web/)
talking to a small stateless JSON API. All heavy lifting is delegated to the
existing main_code.build_resume / main_code.resume_bullet_workflow modules,
so both UIs share one pipeline.
"""

import json
import os
import sys
import base64
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main_code import layout, qa_report
from main_code.build_resume import (
    _evidence_by_company,
    cleanup_aux_files,
    compile_to_pdf,
    fit_to_one_page,
    polish_document,
    replace_academic_projects,
    replace_columbia_coursework,
    replace_experience_bullets,
    _plan_note,
    drop_experience_entries,
    replace_skills,
    slugify,
    tighten_spacing,
)
from main_code.build_cover_letter import (
    compile_cover_letter,
    TEMPLATE_NAME as COVER_LETTER_TEMPLATE,
)
from main_code.cover_letter_workflow import (
    MAX_COVER_LETTER_PARAGRAPHS,
    MAX_COVER_LETTER_WORDS,
    MIN_COVER_LETTER_WORDS,
    evidence_for_letter,
    generate_cover_letter,
)
from main_code.resume_bullet_workflow import (
    DEFAULT_ACADEMIC_PROJECT_FILE,
    DEFAULT_COLUMBIA_COURSES,
    DEFAULT_GENERATION_MODE,
    DEFAULT_MODEL,
    DEFAULT_TOP_ACADEMIC_PROJECT_COUNT,
    DEFAULT_TOP_COURSE_COUNT,
    GENERATION_MODES,
    MAX_BULLET_CHARS,
    MIN_BULLET_CHARS,
    MODEL_CHOICES,
    extract_jd_signals,
    extract_numbered_bullets,
    extract_starting_verbs,
    generate_bullets,
    parse_filename,
    read_projects,
    run_all_with_full_selection,
    select_academic_projects_by_topics,
    select_skills_for_jd,
)

WORK_DIR = Path(os.getenv("RESUME_WORK_DIR", ".")).resolve()
DATA_DIR = Path(os.getenv("RESUME_DATA_DIR", WORK_DIR / "data")).resolve()
OUTPUT_DIR = Path(os.getenv("RESUME_OUTPUT_DIR", WORK_DIR / "output")).resolve()
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Resume Tailor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    company_name: str
    position_name: str
    jd_text: str
    model: str = DEFAULT_MODEL
    generation_mode: str = DEFAULT_GENERATION_MODE
    log_prompts: bool = False


class RegenerateRequest(BaseModel):
    company: str
    jd_text: str
    instruction: str = ""
    model: str = DEFAULT_MODEL
    other_bullets: Dict[str, List[str]] = {}
    seniority: str = ""


class CoverLetterRequest(BaseModel):
    company_name: str
    position_name: str
    jd_text: str
    model: str = DEFAULT_MODEL
    education: List[str] = []
    log_prompts: bool = False


class CompileCoverLetterRequest(BaseModel):
    company_name: str
    position_name: str
    paragraphs: List[str]
    # Re-validated server-side before compiling: the draft may have been edited
    # in the UI between the two calls, and an edit can reintroduce exactly what
    # the first pass rejected.
    jd_text: str = ""


class CompileRequest(BaseModel):
    company_name: str
    position_name: str
    jd_text: str
    model: str = DEFAULT_MODEL
    bullets: Dict[str, List[str]]
    selected_courses: List[str]
    selected_academic_topics: List[str]
    jd_signals: Dict[str, Any] = {}
    plan: Dict[str, Any] = {}


# ── Status / config ──────────────────────────────────────────────────────────


@app.get("/api/status")
def get_status():
    work_files = sorted(DATA_DIR.glob("work_*_*-*.json"))
    evidence = []
    for f in work_files:
        try:
            company, min_b, max_b = parse_filename(f)
            evidence.append({"company": company, "min": min_b, "max": max_b})
        except ValueError:
            continue
    template_path = DATA_DIR / "main.tex"
    return {
        "evidence_files": evidence,
        "template_exists": template_path.exists(),
        "cover_letter_template_exists": (DATA_DIR / COVER_LETTER_TEMPLATE).exists(),
        "model_choices": MODEL_CHOICES,
        "default_model": DEFAULT_MODEL,
        "generation_modes": list(GENERATION_MODES),
        "default_generation_mode": DEFAULT_GENERATION_MODE,
        "min_bullet_chars": MIN_BULLET_CHARS,
        "max_bullet_chars": MAX_BULLET_CHARS,
        "default_courses": DEFAULT_COLUMBIA_COURSES,
        "default_top_course_count": DEFAULT_TOP_COURSE_COUNT,
        "default_top_academic_project_count": DEFAULT_TOP_ACADEMIC_PROJECT_COUNT,
    }


@app.get("/api/jd-default")
def get_jd_default():
    existing_jd = DATA_DIR / "JD.txt"
    company, position, jd_text = "", "", ""
    if existing_jd.exists():
        content = existing_jd.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(content)
            company = data.get("company_name", "")
            position = data.get("position_name", "")
            jd_text = data.get("job_description", content)
        except json.JSONDecodeError:
            if "---" in content:
                header, _, body = content.partition("---")
                for line in header.strip().splitlines():
                    if ":" in line:
                        key, _, value = line.partition(":")
                        k = key.strip()
                        if k == "company_name":
                            company = value.strip()
                        elif k == "position_name":
                            position = value.strip()
                jd_text = body.strip()
            else:
                jd_text = content
    return {"company_name": company, "position_name": position, "jd_text": jd_text}


@app.get("/api/academic-projects")
def get_academic_projects():
    path = DATA_DIR / DEFAULT_ACADEMIC_PROJECT_FILE
    if not path.exists():
        return {"projects": []}
    return {"projects": read_projects(path)}


# ── Generate ─────────────────────────────────────────────────────────────────


@app.post("/api/generate")
def generate(req: GenerateRequest):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jd_path = OUTPUT_DIR / "_jd_temp_web.json"
    jd_path.write_text(
        json.dumps(
            {
                "company_name": req.company_name.strip(),
                "position_name": req.position_name.strip(),
                "job_description": req.jd_text.strip(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    try:
        (
            bullets,
            selected_courses,
            selected_topics,
            selected_academic_projects,
            jd_signals,
            plan,
        ) = run_all_with_full_selection(
            jd_path=jd_path,
            directory=DATA_DIR,
            model=req.model,
            log_prompts=req.log_prompts,
            generation_mode=req.generation_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if jd_path.exists():
            jd_path.unlink()

    return {
        "bullets": bullets,
        "selected_courses": selected_courses,
        "selected_academic_topics": selected_topics,
        "selected_academic_projects": selected_academic_projects,
        "jd_signals": jd_signals,
        "plan": plan,
    }


@app.post("/api/regenerate")
def regenerate(req: RegenerateRequest):
    matches = list(DATA_DIR.glob(f"work_{req.company}_*.json"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"No evidence file for '{req.company}'")

    guided_jd = req.jd_text
    if req.instruction.strip():
        guided_jd = (
            f"{req.jd_text}\n\n"
            "ADDITIONAL INSTRUCTION FROM THE CANDIDATE (highest priority, but it never\n"
            "overrides truthfulness or the evidence): "
            f"{req.instruction.strip()}"
        )

    other_verbs: List[str] = []
    for other, bullet_list in req.other_bullets.items():
        if other == req.company:
            continue
        other_verbs.extend(extract_starting_verbs(bullet_list))

    try:
        projects = read_projects(matches[0])
        output = generate_bullets(
            jd_text=guided_jd,
            project_file=matches[0],
            projects=projects,
            model=req.model,
            log_prompts=False,
            used_verbs=other_verbs,
            seniority=req.seniority,
        )
        bullets = extract_numbered_bullets(output)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not bullets:
        raise HTTPException(status_code=502, detail="No bullets returned")
    return {"bullets": bullets}


# ── Compile ──────────────────────────────────────────────────────────────────


@app.post("/api/compile")
def compile_resume(req: CompileRequest):
    template_path = DATA_DIR / "main.tex"
    if not template_path.exists():
        raise HTTPException(status_code=400, detail="data/main.tex not found.")

    try:
        tex_content = template_path.read_text(encoding="utf-8")
        tex_content = drop_experience_entries(
            tex_content, [c for c, n in (req.plan.get("roles") or {}).items() if not n]
        )
        new_tex = replace_experience_bullets(tex_content, req.bullets)

        academic_file = DATA_DIR / DEFAULT_ACADEMIC_PROJECT_FILE
        academic_projects = read_projects(academic_file) if academic_file.exists() else []
        selected_academic_projects = select_academic_projects_by_topics(
            project_list=academic_projects,
            selected_topics=req.selected_academic_topics,
        )

        new_tex = replace_columbia_coursework(new_tex, req.selected_courses)
        new_tex = replace_academic_projects(new_tex, selected_academic_projects)

        skills_dropped: List[str] = []
        if req.jd_signals:
            skill_categories, skills_dropped = select_skills_for_jd(
                jd_signals=req.jd_signals,
                model=req.model,
                selected_bullets=req.bullets,
                selected_projects=selected_academic_projects,
            )
            new_tex = replace_skills(new_tex, skill_categories)

        new_tex = tighten_spacing(new_tex)

        slug = f"oranich_resume_{slugify(req.company_name)}_{slugify(req.position_name)}"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tex_out = OUTPUT_DIR / f"{slug}.tex"
        tex_out.write_text(new_tex, encoding="utf-8")

        pdf_out = compile_to_pdf(tex_out)
        pdf_out, trim_actions = fit_to_one_page(tex_out, pdf_out)
        pdf_out, polish_actions = polish_document(tex_out, pdf_out, model=req.model)
        trim_actions = trim_actions + polish_actions
        cleanup_aux_files(tex_out)

        report = qa_report.QAReport()
        report.trim_actions = trim_actions
        report.skills_dropped = skills_dropped
        if req.plan:
            report.plan_note = _plan_note(req.plan)
        qa_report.check_layout(pdf_out, report)
        qa_report.check_redundancy(req.bullets, report)
        qa_report.check_grounding(req.bullets, _evidence_by_company(DATA_DIR), report)
        must_have = (req.jd_signals or {}).get("must_have", [])
        if must_have:
            qa_report.check_jd_coverage(must_have, layout.extract_layout_text(pdf_out), report)

        pdf_bytes = pdf_out.read_bytes()
        tex_text = tex_out.read_text(encoding="utf-8")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
        "pdf_name": pdf_out.name,
        "tex_text": tex_text,
        "tex_name": tex_out.name,
        "qa_report_text": report.render(),
        "qa_report_clean": report.clean,
    }


# ── Cover letter ─────────────────────────────────────────────────────────────


@app.post("/api/cover-letter")
def cover_letter(req: CoverLetterRequest):
    """Draft a cover letter and report every reason it might not be usable.

    Always returns 200 with an `issues` list rather than raising on a bad draft:
    the caller needs to SEE what was rejected and why. `usable` is the only field
    that decides whether this may be attached to an application.
    """
    if not req.jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text is required")

    try:
        jd_signals = extract_jd_signals(req.jd_text, req.model)
        evidence = evidence_for_letter(DATA_DIR)
        paragraphs, issues = generate_cover_letter(
            company_name=req.company_name.strip(),
            position_name=req.position_name.strip(),
            jd_text=req.jd_text.strip(),
            jd_signals=jd_signals,
            evidence=evidence,
            model=req.model,
            education=req.education or None,
            log_prompts=req.log_prompts,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "paragraphs": paragraphs,
        "issues": issues,
        "usable": not issues,
        "jd_signals": jd_signals,
        "word_count": sum(len(p.split()) for p in paragraphs),
        "limits": {
            "min_words": MIN_COVER_LETTER_WORDS,
            "max_words": MAX_COVER_LETTER_WORDS,
            "max_paragraphs": MAX_COVER_LETTER_PARAGRAPHS,
        },
    }


@app.post("/api/compile-cover-letter")
def compile_cover_letter_route(req: CompileCoverLetterRequest):
    """Compile approved paragraphs to a PDF.

    Kept separate from /api/compile, which is resume-shaped — it takes bullets,
    coursework and skills, none of which a letter has.
    """
    template_path = DATA_DIR / COVER_LETTER_TEMPLATE
    if not template_path.exists():
        raise HTTPException(
            status_code=400, detail=f"data/{COVER_LETTER_TEMPLATE} not found."
        )
    if not req.paragraphs:
        raise HTTPException(status_code=400, detail="No paragraphs to compile.")

    # Re-check the placeholder and grounding rules here. The paragraphs may have
    # been edited since /api/cover-letter approved them, and a hand-edit can put
    # back the one thing that must never be submitted.
    from main_code.cover_letter_workflow import _allowed_claims, validate_cover_letter

    issues = validate_cover_letter(
        req.paragraphs,
        req.company_name.strip(),
        req.position_name.strip(),
        _allowed_claims(evidence_for_letter(DATA_DIR), req.jd_text),
    )
    if issues:
        raise HTTPException(
            status_code=422,
            detail="This letter is not safe to submit: " + " ".join(issues),
        )

    try:
        pdf_path, report = compile_cover_letter(
            paragraphs=req.paragraphs,
            company_name=req.company_name.strip(),
            position_name=req.position_name.strip(),
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
        )
        pdf_bytes = pdf_path.read_bytes()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
        "pdf_name": pdf_path.name,
        "report": report,
    }


# ── Static frontend ──────────────────────────────────────────────────────────

if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")

    @app.get("/")
    def index():
        return FileResponse(
            str(WEB_DIR / "index.html"),
            headers={"Cache-Control": "no-store"},
        )

    @app.middleware("http")
    async def no_cache_assets(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-store"
        return response


def run() -> None:
    import uvicorn

    uvicorn.run(
        "main_code.api_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    run()
