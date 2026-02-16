"""Streamlit UI for the Resume Customization Workflow.

Run locally:  streamlit run main_code/app.py
Cloud:        Deploy via Docker to Google Cloud Run
"""

import json
import os
from pathlib import Path

import streamlit as st

from main_code.build_resume import (
    compile_to_pdf,
    cleanup_aux_files,
    replace_experience_bullets,
    slugify,
    tighten_spacing,
)
from main_code.resume_bullet_workflow import (
    DEFAULT_MODEL,
    MIN_BULLET_CHARS,
    MAX_BULLET_CHARS,
    parse_filename,
    run_all,
)

WORK_DIR = Path(os.getenv("RESUME_WORK_DIR", ".")).resolve()
DATA_DIR = Path(os.getenv("RESUME_DATA_DIR", WORK_DIR / "data")).resolve()
OUTPUT_DIR = Path(os.getenv("RESUME_OUTPUT_DIR", WORK_DIR / "output")).resolve()


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

    st.markdown(
        "<div style='text-align:center; margin-top:15vh'>"
        "<h2>Resume Builder</h2>"
        "<p style='color:#888'>Enter password to continue</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            pwd = st.text_input(
                "Password",
                type="password",
                label_visibility="collapsed",
                placeholder="Password",
            )
            if st.form_submit_button("Sign In", use_container_width=True):
                if pwd == password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
    return False


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar(work_files: list[Path]) -> tuple[str, bool]:
    """Render sidebar settings. Returns (model, log_prompts)."""
    with st.sidebar:
        st.header("Settings")
        model = st.text_input(
            "LLM Model",
            value=os.getenv("LITELLM_MODEL", DEFAULT_MODEL),
            help="LiteLLM model identifier (e.g. vertex_ai/gemini-2.5-flash)",
        )
        log_prompts = st.checkbox("Log prompts to disk", value=False)

        st.divider()
        st.subheader("Evidence Files")
        if work_files:
            for f in work_files:
                try:
                    company, min_b, max_b = parse_filename(f)
                    name = company.replace("_", " ").title()
                    st.markdown(f"**{name}** — {min_b}-{max_b} bullets")
                except ValueError:
                    st.text(f.name)
        else:
            st.warning("No work_*.json files found.")

        template = DATA_DIR / "main.tex"
        st.divider()
        if template.exists():
            st.markdown("**Template:** data/main.tex")
        else:
            st.error("data/main.tex not found.")

    return model, log_prompts


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

    for company, bullet_list in bullets.items():
        display = company.replace("_", " ").title()
        st.markdown(f"#### {display}")

        company_bullets: list[str] = []
        for i, bullet in enumerate(bullet_list):
            val = st.text_area(
                f"{display} — bullet {i + 1}",
                value=bullet,
                key=f"b_{company}_{i}",
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


# ── Main App ──────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Resume Builder",
        page_icon=":page_facing_up:",
        layout="wide",
    )

    st.markdown(
        "<style>"
        ".stTextArea textarea { font-size: 14px; }"
        ".block-container { padding-top: 2rem; }"
        "</style>",
        unsafe_allow_html=True,
    )

    if not check_password():
        return

    st.title("Resume Customization Tool")
    st.caption(
        "Generate ATS-optimized resume bullets tailored to any job description, "
        "then compile to PDF."
    )

    work_files = sorted(DATA_DIR.glob("work_*_*-*.json"))
    model, log_prompts = render_sidebar(work_files)
    template_path = DATA_DIR / "main.tex"

    # ── Step 1: JD ────────────────────────────────────────────────────────
    st.header("1 — Job Description")
    company_name, position_name, jd_text = render_jd_input()

    ready = bool(
        jd_text.strip() and company_name.strip() and work_files and template_path.exists()
    )

    # ── Step 2: Generate ──────────────────────────────────────────────────
    st.divider()
    st.header("2 — Generate Bullets")

    if st.button(
        "Generate ATS-Optimized Bullets",
        type="primary",
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
                bullets = run_all(
                    jd_path=jd_path,
                    directory=DATA_DIR,
                    model=model,
                    log_prompts=log_prompts,
                )
            st.session_state.bullets = bullets
            st.session_state.company_name = company_name.strip()
            st.session_state.position_name = position_name.strip()
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

    st.divider()
    st.header("3 — Review & Edit")
    edited_bullets = render_bullet_editor(st.session_state.bullets)

    st.divider()
    st.header("4 — Build PDF")

    if st.button("Compile Resume PDF", type="primary", use_container_width=True):
        try:
            with st.spinner("Injecting bullets and compiling LaTeX ..."):
                tex_content = template_path.read_text(encoding="utf-8")
                new_tex = replace_experience_bullets(tex_content, edited_bullets)
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

    # ── Downloads ─────────────────────────────────────────────────────────
    if "pdf_bytes" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download PDF",
                data=st.session_state.pdf_bytes,
                file_name=st.session_state.pdf_name,
                mime="application/pdf",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Download TeX",
                data=st.session_state.tex_text,
                file_name=st.session_state.tex_name,
                mime="text/plain",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
