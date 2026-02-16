# ATS Resume Builder

AI-powered workflow that generates tailored, ATS-optimized resume bullets from job descriptions using LLMs, then compiles professional PDF resumes.

## How It Works

### Workflow Overview

1. **Job Description Analysis** → Extract key signals (role type, required skills, domain keywords)
2. **Per-Company Generation** → Generate tailored bullets for each work experience using validated LLM calls
3. **Validation & Repair** → Enforce constraints (length, format, uniqueness) with automatic retry loops
4. **Template Injection** → Inject generated bullets into LaTeX resume template
5. **PDF Compilation** → Compile to professional PDF using TeX

### Key Features

- **ATS Optimization**: Actively reframes work using JD keywords and terminology for maximum ATS relevance
- **Per-Company Generation**: Each company's bullets are generated separately with full validation/repair cycles
- **Constraint Enforcement**: Hard limits on bullet length (190-220 chars), unique action verbs, measurable impact placement
- **Evidence-Based**: Stays grounded in provided project JSON evidence — no fabrication
- **Lexical Diversity**: Varies vocabulary, avoids repetition, spreads tools across bullets

### Architecture

```
┌─────────────────┐
│  Job Description│ → Extract JD signals (role, skills, keywords)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Work Evidence   │ → Load project JSON files (work_*.json)
│ (JSON files)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Per-Company Bullet Generation        │
│ • Generate bullets per company       │
│ • Validate constraints               │
│ • Repair if needed (max 2 attempts) │
│ • Track used verbs across companies  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ LaTeX Template  │ → Inject bullets into main.tex
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PDF Output      │ → Compiled resume PDF
└─────────────────┘
```

### Generation Pipeline

**Input Format:**
- Job description: Header+body format (`company_name: X`, `position_name: Y`, `---`, `JD text`) or JSON
- Work evidence: JSON files with project details, actions, results, tools, constraints

**Generation Process:**
1. **JD Signal Extraction**: Uses LLM to analyze JD and extract role type, key skills, domain keywords
2. **Prompt Construction**: Builds system/user prompts with:
   - JD analysis signals
   - Project evidence
   - Hard constraints (length, format, uniqueness)
   - Tailoring instructions (reframe with JD keywords)
3. **LLM Call**: Vertex AI (Gemini 2.5 Pro) generates numbered bullets
4. **Validation**: Checks bullet count, character length, verb uniqueness, impact placement
5. **Repair Loop**: If validation fails, sends repair prompt with issues and regenerates (up to 2 attempts)
6. **Output**: Canonical numbered list of bullets

**Output Format:**
- Numbered bullets (1. ..., 2. ..., etc.)
- Each bullet: 190-220 characters
- Unique starting action verbs per company
- Measurable impact at end (when available)

## Project Structure

```
.
├── main_code/              # Application source code
│   ├── resume_bullet_workflow.py  # Core generation logic
│   ├── build_resume.py            # End-to-end pipeline
│   └── app.py                     # Streamlit UI
├── data/                   # Input files
│   ├── JD.txt             # Job description (header+body or JSON)
│   ├── main.tex           # LaTeX resume template
│   └── work_*.json        # Work experience evidence files
├── output/                 # Generated artifacts
│   ├── prompt_logs/       # LLM prompt/response logs
│   └── *.tex, *.pdf       # Generated resumes
└── reference_resume/       # Reference PDFs
```

## Quick Start

### Setup

```bash
uv sync
export VERTEXAI_PROJECT="your-project"
export VERTEXAI_LOCATION="us-central1"
```

### Generate Resume

```bash
# Full pipeline (bullets → TeX → PDF)
uv run build-resume --log-prompts

# Generate bullets only (all companies)
uv run resume-bullets --all --jd data/JD.txt

# Generate bullets for one company
uv run resume-bullets --project-file data/work_agoda_2-2.json --jd data/JD.txt
```

### Streamlit UI

```bash
uv run streamlit run main_code/app.py
```

## Configuration

- **Model**: Defaults to `vertex_ai/gemini-2.5-pro` (configurable via `--model` or `LITELLM_MODEL`)
- **Bullet Length**: 190-220 characters (hard constraint)
- **Max Repair Attempts**: 2 per company
- **Output Directory**: `output/` (configurable via `--output-dir`)
