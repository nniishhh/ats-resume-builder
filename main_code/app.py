"""Streamlit UI for the Resume Customization Workflow.

Run locally:  streamlit run main_code/app.py
Cloud:        Deploy via Docker to Google Cloud Run
"""

import json
import os
import sys
from pathlib import Path

# When run via `streamlit run main_code/app.py`, ensure imports resolve to the
# local project source (repo root), not an older installed package copy.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from main_code.build_resume import (
    compile_to_pdf,
    cleanup_aux_files,
    replace_academic_projects,
    replace_columbia_coursework,
    replace_experience_bullets,
    slugify,
    tighten_spacing,
)
from main_code.resume_bullet_workflow import (
    DEFAULT_ACADEMIC_PROJECT_FILE,
    DEFAULT_COLUMBIA_COURSES,
    DEFAULT_GENERATION_MODE,
    DEFAULT_MODEL,
    DEFAULT_TOP_ACADEMIC_PROJECT_COUNT,
    DEFAULT_TOP_COURSE_COUNT,
    GENERATION_MODES,
    MODEL_CHOICES,
    MIN_BULLET_CHARS,
    MAX_BULLET_CHARS,
    parse_filename,
    read_projects,
    run_all_with_full_selection,
    select_academic_projects_by_topics,
    select_top_academic_topics_for_jd,
    select_top_courses_for_jd,
    summarize_job_description,
)

WORK_DIR = Path(os.getenv("RESUME_WORK_DIR", ".")).resolve()
DATA_DIR = Path(os.getenv("RESUME_DATA_DIR", WORK_DIR / "data")).resolve()
OUTPUT_DIR = Path(os.getenv("RESUME_OUTPUT_DIR", WORK_DIR / "output")).resolve()

STEPS = ["Job Description", "Generate", "Review & Polish", "Export"]

APP_CSS = """
<style>
:root {
    --rt-primary: #1F4E5F;
    --rt-accent: #B8752E;
    --rt-border: #E2D9C7;
    --rt-text: #211F1C;
    --rt-muted: #6B655C;
}

.block-container { padding-top: 2.75rem; padding-bottom: 3rem; max-width: 1180px; }

/* ── Hero ─────────────────────────────────────────────────────────── */
.rt-eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--rt-accent);
    margin: 0 0 0.4rem 0;
}
.rt-tagline {
    color: var(--rt-muted);
    font-size: 1.04rem;
    max-width: 660px;
    line-height: 1.6;
    margin: 0.1rem 0 0 0;
}

/* ── Stepper ──────────────────────────────────────────────────────── */
.rt-stepper {
    display: flex; align-items: center; margin: 2rem 0 0.25rem 0;
    overflow-x: auto; -webkit-overflow-scrolling: touch;
    padding-bottom: 0.25rem;
}
.rt-step { display: flex; align-items: center; gap: 0.55rem; flex: 0 0 auto; }
.rt-step .dot {
    width: 27px; height: 27px; border-radius: 999px;
    display: flex; align-items: center; justify-content: center;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.72rem; font-weight: 600;
    border: 1.5px solid var(--rt-border);
    color: var(--rt-muted); background: #FFFFFF;
    transition: all 150ms ease;
}
.rt-step.done .dot { background: var(--rt-primary); border-color: var(--rt-primary); color: #fff; }
.rt-step.active .dot { border-color: var(--rt-accent); color: var(--rt-accent); box-shadow: 0 0 0 4px rgba(184,117,46,0.15); }
.rt-step .label { font-size: 0.82rem; font-weight: 500; color: var(--rt-muted); white-space: nowrap; }
.rt-step.active .label, .rt-step.done .label { color: var(--rt-text); }
.rt-step-line { flex: 1 1 auto; height: 1.5px; background: var(--rt-border); margin: 0 0.9rem; min-width: 16px; }
.rt-step-line.done { background: var(--rt-primary); }

/* ── Step headers inside cards ───────────────────────────────────── */
.rt-step-head { display: flex; align-items: baseline; gap: 0.65rem; margin-bottom: 0.1rem; }
.rt-step-head .num { font-family: "JetBrains Mono", monospace; color: var(--rt-accent); font-weight: 600; font-size: 0.85rem; }
.rt-step-head h2 { margin: 0 !important; }
.rt-step-sub { color: var(--rt-muted); font-size: 0.9rem; margin: 0.1rem 0 1.1rem 0; }

/* ── Cards ────────────────────────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: 0 1px 0 rgba(33,31,28,0.03), 0 12px 32px -20px rgba(33,31,28,0.35);
}

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    transition: transform 120ms ease, box-shadow 120ms ease, background-color 120ms ease;
    font-weight: 600;
}
.stButton > button:active, .stDownloadButton > button:active, .stFormSubmitButton > button:active {
    transform: translateY(1px);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    box-shadow: 0 1px 0 rgba(31,78,95,0.08), 0 12px 26px -10px rgba(31,78,95,0.5);
}

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .rt-eyebrow { color: var(--rt-primary); }
.rt-file-chip {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.4rem 0.7rem; margin-bottom: 0.4rem;
    background: #FFFFFF; border: 1px solid var(--rt-border); border-radius: 10px;
    font-size: 0.83rem;
}
.rt-file-chip .count {
    font-family: "JetBrains Mono", monospace; font-size: 0.72rem;
    color: var(--rt-muted);
}

/* ── Misc ─────────────────────────────────────────────────────────── */
.stTextArea textarea { font-size: 14px; line-height: 1.55; }
h4 { margin-top: 0.25rem !important; }
</style>
"""


# ── Layout helpers ──────────────────────────────────────────────────────────


def render_stepper(current_index: int) -> None:
    """Render a horizontal progress stepper across the four workflow stages."""
    parts = ['<div class="rt-stepper">']
    for i, label in enumerate(STEPS):
        state = "done" if i < current_index else ("active" if i == current_index else "")
        marker = "✓" if state == "done" else str(i + 1)
        parts.append(
            f'<div class="rt-step {state}">'
            f'<span class="dot">{marker}</span>'
            f'<span class="label">{label}</span>'
            f"</div>"
        )
        if i < len(STEPS) - 1:
            line_state = "done" if i < current_index else ""
            parts.append(f'<div class="rt-step-line {line_state}"></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def step_head(number: int, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="rt-step-head"><span class="num">{number:02d}</span>'
        f"<h2>{title}</h2></div>"
        f'<p class="rt-step-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


# ── Authentication ────────────────────────────────────────────────────────────


def check_password() -> bool:
    """Simple shared-password gate.

    Set APP_PASSWORD env var to enable. If unset, access is open (local dev).
    """
    password = os.getenv("APP_PASSWORD", "")
    if not password:
        return True

    if st.session_state.get("authenticated"):
        return True

    st.markdown("<div style='margin-top:12vh'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.container(border=True):
            st.markdown(
                '<p class="rt-eyebrow" style="text-align:center">Private workspace</p>'
                "<h2 style='text-align:center; margin:0 0 0.3rem 0'>Resume Tailor</h2>"
                "<p style='text-align:center; color:var(--rt-muted); margin-bottom:1.4rem'>"
                "Enter your password to continue</p>",
                unsafe_allow_html=True,
            )
            with st.form("login_form"):
                pwd = st.text_input(
                    "Password",
                    type="password",
                    label_visibility="collapsed",
                    placeholder="Password",
                )
                if st.form_submit_button(
                    "Sign in",
                    type="primary",
                    icon=":material/login:",
                    use_container_width=True,
                ):
                    if pwd == password:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Incorrect password.")
    return False


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar(
    work_files: list[Path],
) -> tuple[str, bool, str]:
    """Render sidebar settings."""
    with st.sidebar:
        st.markdown(
            '<p class="rt-eyebrow">Workspace</p>'
            "<h3 style='margin-top:0'>Generation Settings</h3>",
            unsafe_allow_html=True,
        )
        custom_label = "Custom (type manually)"
        configured_model = os.getenv("LITELLM_MODEL", DEFAULT_MODEL)
        dropdown_options = MODEL_CHOICES + [custom_label]
        default_index = (
            MODEL_CHOICES.index(configured_model)
            if configured_model in MODEL_CHOICES
            else len(MODEL_CHOICES)
        )
        selected = st.selectbox(
            "Model",
            options=dropdown_options,
            index=default_index,
            help=(
                "Model for all generation tasks. "
                "OpenAI (openai/...) needs OPENAI_API_KEY; "
                "Vertex (vertex_ai/...) needs VERTEXAI_PROJECT."
            ),
        )
        if selected == custom_label:
            model = st.text_input(
                "Custom model name",
                value="" if configured_model in MODEL_CHOICES else configured_model,
                placeholder="e.g. openai/gpt-4o-mini",
            ).strip() or DEFAULT_MODEL
        else:
            model = selected
        log_prompts = st.checkbox("Log prompts to disk", value=False)
        generation_mode = st.selectbox(
            "Generation Mode",
            options=list(GENERATION_MODES),
            index=list(GENERATION_MODES).index(DEFAULT_GENERATION_MODE),
            format_func=lambda mode: (
                "One Prompt (all companies JSON)"
                if mode == "single_prompt"
                else "Sequential (one company at a time)"
            ),
            help=(
                "single_prompt: all companies in one model call. "
                "sequential: generate one company at a time and avoid reusing starting verbs."
            ),
        )

        st.divider()
        st.markdown('<p class="rt-eyebrow">Evidence files</p>', unsafe_allow_html=True)
        if work_files:
            for f in work_files:
                try:
                    company, min_b, max_b = parse_filename(f)
                    name = company.replace("_", " ").title()
                    st.markdown(
                        f'<div class="rt-file-chip"><span>{name}</span>'
                        f'<span class="count">{min_b}–{max_b}</span></div>',
                        unsafe_allow_html=True,
                    )
                except ValueError:
                    st.text(f.name)
        else:
            st.warning("No work_*.json files found.")

        template = DATA_DIR / "main.tex"
        st.divider()
        st.markdown('<p class="rt-eyebrow">Template</p>', unsafe_allow_html=True)
        if template.exists():
            st.caption("data/main.tex ✓")
        else:
            st.error("data/main.tex not found.")

    return model, log_prompts, generation_mode


# ── JD Input ──────────────────────────────────────────────────────────────────


def render_jd_input() -> tuple[str, str, str]:
    """Render JD input. Returns (company_name, position_name, jd_text)."""

    # Pre-fill from existing JD.txt if present
    existing_jd = DATA_DIR / "JD.txt"
    default_company = ""
    default_position = ""
    default_jd = ""

    if existing_jd.exists():
        content = existing_jd.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(content)
            default_company = data.get("company_name", "")
            default_position = data.get("position_name", "")
            default_jd = data.get("job_description", content)
        except json.JSONDecodeError:
            # Try header+body format (key: value lines, then ---, then body)
            if "---" in content:
                header, _, body = content.partition("---")
                for line in header.strip().splitlines():
                    if ":" in line:
                        key, _, value = line.partition(":")
                        k = key.strip()
                        if k == "company_name":
                            default_company = value.strip()
                        elif k == "position_name":
                            default_position = value.strip()
                default_jd = body.strip()
            else:
                default_jd = content

    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input(
            "Company Name", value=default_company, placeholder="e.g. Netflix"
        )
    with col2:
        position_name = st.text_input(
            "Position Name", value=default_position, placeholder="e.g. AI Trainee"
        )

    jd_text = st.text_area(
        "Job Description",
        value=default_jd,
        height=250,
        placeholder="Paste the full job description here...",
    )

    uploaded = st.file_uploader(
        "Or upload a JD file (JSON or plain text)", type=["json", "txt"]
    )
    if uploaded:
        raw = uploaded.read().decode("utf-8").strip()
        try:
            data = json.loads(raw)
            company_name = data.get("company_name", company_name)
            position_name = data.get("position_name", position_name)
            jd_text = data.get("job_description", raw)
        except json.JSONDecodeError:
            if "---" in raw:
                header, _, body = raw.partition("---")
                for line in header.strip().splitlines():
                    if ":" in line:
                        key, _, value = line.partition(":")
                        k = key.strip()
                        if k == "company_name":
                            company_name = value.strip()
                        elif k == "position_name":
                            position_name = value.strip()
                jd_text = body.strip()
            else:
                jd_text = raw

    return company_name, position_name, jd_text


# ── Bullet Editor ─────────────────────────────────────────────────────────────


def render_bullet_editor(bullets: dict[str, list[str]]) -> dict[str, list[str]]:
    """Render editable bullet text areas with live character counts."""
    edited: dict[str, list[str]] = {}
    editor_nonce = int(st.session_state.get("editor_nonce", 0))

    for company, bullet_list in bullets.items():
        display = company.replace("_", " ").title()
        st.markdown(f"#### {display}")

        company_bullets: list[str] = []
        for i, bullet in enumerate(bullet_list):
            val = st.text_area(
                f"{display} — bullet {i + 1}",
                value=bullet,
                # Include a nonce so a new generation doesn't reuse old widget state.
                key=f"b_{editor_nonce}_{company}_{i}",
                height=80,
                label_visibility="collapsed",
            )
            n = len(val)
            if MIN_BULLET_CHARS <= n <= MAX_BULLET_CHARS:
                st.caption(f":green[{n} chars]")
            elif n < MIN_BULLET_CHARS:
                st.caption(f":orange[{n} chars — {MIN_BULLET_CHARS - n} under min]")
            else:
                st.caption(f":red[{n} chars — {n - MAX_BULLET_CHARS} over max]")
            company_bullets.append(val)

        edited[company] = company_bullets
    return edited


def render_combined_bullets(bullets: dict[str, list[str]]) -> None:
    """Render copy-friendly combined bullet blocks per company."""
    st.markdown('<p class="rt-eyebrow">Copy-ready</p>', unsafe_allow_html=True)
    st.caption("Combined bullets per company, ready to paste elsewhere.")
    for company, bullet_list in bullets.items():
        display = company.replace("_", " ").title()
        combined = "\n".join(
            f"- {bullet.strip()}" for bullet in bullet_list if bullet.strip()
        )
        combined_key = f"combined_{company}"
        # Keep this widget synced to latest edited/generated bullets each rerun.
        st.session_state[combined_key] = combined
        st.text_area(
            f"{display} — combined",
            key=combined_key,
            height=140,
        )


# ── Coursework & Projects Editor ──────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_academic_projects(path_str: str) -> list[dict]:
    """Load the academic project pool once (cached by file path)."""
    try:
        return read_projects(Path(path_str))
    except Exception:
        return []


def render_selection_editors(all_projects: list[dict]) -> None:
    """Editable pickers for Columbia coursework and academic projects.

    Reads the AI's auto-selection as the starting point, then lets the user
    add/remove entries. Selections are synced back to session_state so the
    build step picks them up.
    """
    nonce = int(st.session_state.get("editor_nonce", 0))

    # ── Coursework ──────────────────────────────────────────────────────
    current_courses = list(st.session_state.get("selected_courses", []) or [])
    course_options = list(dict.fromkeys(list(DEFAULT_COLUMBIA_COURSES) + current_courses))
    default_courses = [c for c in current_courses if c in course_options]
    st.markdown('<p class="rt-eyebrow">Columbia coursework</p>', unsafe_allow_html=True)
    st.caption(
        f"Listed under Education. AI picked ~{DEFAULT_TOP_COURSE_COUNT} for this role — "
        "add or drop any you like."
    )
    chosen_courses = st.multiselect(
        "Columbia coursework",
        options=course_options,
        default=default_courses,
        key=f"courses_ms_{nonce}",
        label_visibility="collapsed",
        placeholder="Choose coursework to feature",
    )
    st.session_state.selected_courses = chosen_courses

    # ── Academic projects ───────────────────────────────────────────────
    all_topics = [
        str(p.get("Topic", "")).strip()
        for p in all_projects
        if str(p.get("Topic", "")).strip()
    ]
    if all_topics:
        current_topics = list(st.session_state.get("selected_academic_topics", []) or [])
        topic_options = list(dict.fromkeys(all_topics + current_topics))
        default_topics = [t for t in current_topics if t in topic_options]
        st.markdown(
            '<p class="rt-eyebrow" style="margin-top:1rem">Academic projects</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Listed under Academic Projects. AI picked ~{DEFAULT_TOP_ACADEMIC_PROJECT_COUNT} "
            "most relevant — swap in any of your projects."
        )
        chosen_topics = st.multiselect(
            "Academic projects",
            options=topic_options,
            default=default_topics,
            key=f"topics_ms_{nonce}",
            label_visibility="collapsed",
            placeholder="Choose projects to feature",
        )
        st.session_state.selected_academic_topics = chosen_topics
        st.session_state.selected_academic_projects = select_academic_projects_by_topics(
            project_list=all_projects,
            selected_topics=chosen_topics,
        )


# ── Main App ──────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Resume Tailor",
        page_icon="🖋️",
        layout="wide",
    )

    st.markdown(APP_CSS, unsafe_allow_html=True)

    if not check_password():
        return

    st.markdown('<p class="rt-eyebrow">AI-powered · ATS-optimized</p>', unsafe_allow_html=True)
    st.title("Resume Tailor")
    st.markdown(
        '<p class="rt-tagline">Paste a job description and get resume bullets '
        "grounded in your real experience — reviewed, polished, and compiled "
        "to a professional PDF in minutes.</p>",
        unsafe_allow_html=True,
    )

    work_files = sorted(DATA_DIR.glob("work_*_*-*.json"))
    model, log_prompts, generation_mode = render_sidebar(work_files)
    template_path = DATA_DIR / "main.tex"
    academic_projects = load_academic_projects(str(DATA_DIR / DEFAULT_ACADEMIC_PROJECT_FILE))

    stage = 0
    if "bullets" in st.session_state:
        stage = 2
    if "pdf_bytes" in st.session_state:
        stage = 3
    render_stepper(stage)

    # ── Step 1: JD ────────────────────────────────────────────────────────
    with st.container(border=True):
        step_head(1, "Job Description", "Tell us who you're applying to.")
        company_name, position_name, jd_text = render_jd_input()

    ready = bool(
        jd_text.strip() and company_name.strip() and work_files and template_path.exists()
    )

    # ── Step 2: Generate ──────────────────────────────────────────────────
    with st.container(border=True):
        step_head(2, "Generate Bullets", "AI drafts ATS-optimized bullets from your evidence files.")

        if st.button(
            "Generate ATS-optimized bullets",
            type="primary",
            icon=":material/auto_awesome:",
            disabled=not ready,
            use_container_width=True,
        ):
            jd_data = {
                "company_name": company_name.strip(),
                "position_name": position_name.strip(),
                "job_description": jd_text.strip(),
            }
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            jd_path = OUTPUT_DIR / "_jd_temp.json"
            jd_path.write_text(
                json.dumps(jd_data, ensure_ascii=False), encoding="utf-8"
            )

            try:
                with st.spinner("Analyzing JD and generating bullets — this takes ~30 s ..."):
                    (
                        bullets,
                        selected_courses,
                        selected_topics,
                        selected_academic_projects,
                    ) = run_all_with_full_selection(
                        jd_path=jd_path,
                        directory=DATA_DIR,
                        model=model,
                        log_prompts=log_prompts,
                        generation_mode=generation_mode,
                    )
                st.session_state.bullets = bullets
                st.session_state.selected_courses = selected_courses
                st.session_state.selected_academic_topics = selected_topics
                st.session_state.selected_academic_projects = selected_academic_projects
                st.session_state.company_name = company_name.strip()
                st.session_state.position_name = position_name.strip()
                st.session_state.jd_text = jd_text.strip()
                st.session_state.editor_nonce = int(st.session_state.get("editor_nonce", 0)) + 1
                # Clear previous PDF so stale download button disappears
                st.session_state.pop("pdf_bytes", None)
                st.session_state.pop("tex_text", None)
                st.success(f"Generated bullets for {len(bullets)} companies.")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
            finally:
                if jd_path.exists():
                    jd_path.unlink()

    # ── Step 3: Edit & Build ──────────────────────────────────────────────
    if "bullets" not in st.session_state:
        return

    with st.container(border=True):
        step_head(3, "Review & Polish", "Fine-tune language, length, and impact before you export.")

        edited_bullets = render_bullet_editor(st.session_state.bullets)

        st.divider()
        render_selection_editors(academic_projects)

        st.divider()
        render_combined_bullets(edited_bullets)

    with st.container(border=True):
        step_head(4, "Export", "Compile your tailored resume to a polished PDF.")

        if st.button(
            "Compile resume PDF",
            type="primary",
            icon=":material/picture_as_pdf:",
            use_container_width=True,
        ):
            try:
                with st.spinner("Injecting bullets and compiling LaTeX ..."):
                    tex_content = template_path.read_text(encoding="utf-8")
                    new_tex = replace_experience_bullets(tex_content, edited_bullets)
                    # Selections come from the Step 3 pickers (synced to
                    # session_state). Only fall back to the LLM if they were
                    # never populated at all — an intentional empty pick is kept.
                    selected_courses = st.session_state.get("selected_courses")
                    if "selected_courses" not in st.session_state:
                        jd_summary = summarize_job_description(
                            jd_text=jd_text.strip(),
                            model=model,
                        )
                        selected_courses = select_top_courses_for_jd(
                            jd_text=jd_summary,
                            model=model,
                        )
                        st.session_state.selected_courses = selected_courses
                    new_tex = replace_columbia_coursework(new_tex, selected_courses or [])

                    selected_academic_projects = st.session_state.get(
                        "selected_academic_projects"
                    )
                    if "selected_academic_projects" not in st.session_state:
                        academic_file = DATA_DIR / DEFAULT_ACADEMIC_PROJECT_FILE
                        academic_projects = read_projects(academic_file)
                        jd_summary = summarize_job_description(
                            jd_text=st.session_state.get("jd_text", jd_text).strip(),
                            model=model,
                        )
                        selected_topics = select_top_academic_topics_for_jd(
                            jd_text=jd_summary,
                            project_list=academic_projects,
                            model=model,
                        )
                        selected_academic_projects = select_academic_projects_by_topics(
                            project_list=academic_projects,
                            selected_topics=selected_topics,
                        )
                        st.session_state.selected_academic_topics = selected_topics
                        st.session_state.selected_academic_projects = selected_academic_projects

                    new_tex = replace_academic_projects(
                        new_tex, selected_academic_projects or []
                    )
                    new_tex = tighten_spacing(new_tex)

                    cname = st.session_state.company_name
                    pname = st.session_state.position_name
                    slug = f"oranich_resume_{slugify(cname)}_{slugify(pname)}"

                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    tex_out = OUTPUT_DIR / f"{slug}.tex"
                    tex_out.write_text(new_tex, encoding="utf-8")

                    pdf_out = compile_to_pdf(tex_out)
                    cleanup_aux_files(tex_out)

                # Store bytes in session so download survives re-runs
                st.session_state.pdf_bytes = pdf_out.read_bytes()
                st.session_state.pdf_name = pdf_out.name
                st.session_state.tex_text = tex_out.read_text(encoding="utf-8")
                st.session_state.tex_name = tex_out.name
                st.success("Resume compiled successfully!")
            except Exception as exc:
                st.error(f"Compilation failed: {exc}")

        # ── Preview & Downloads ───────────────────────────────────────────
        if "pdf_bytes" in st.session_state:
            st.markdown('<p class="rt-eyebrow">Preview</p>', unsafe_allow_html=True)
            try:
                st.pdf(st.session_state.pdf_bytes, height=780)
            except Exception:
                st.info(
                    "Inline preview isn't available here — use **Download PDF** below "
                    "to open the compiled resume."
                )
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=st.session_state.pdf_name,
                    mime="application/pdf",
                    icon=":material/download:",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "Download TeX",
                    data=st.session_state.tex_text,
                    file_name=st.session_state.tex_name,
                    mime="text/plain",
                    icon=":material/code:",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
