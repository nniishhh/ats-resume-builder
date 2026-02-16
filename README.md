# Resume Customization Workflow

Generate ATS-optimized resume bullets from a job description and compile a tailored resume PDF.

## Project Structure

- `main_code/` - application and CLI source code
- `data/` - JD, work evidence JSON files, and LaTeX template
- `output/` - generated `.tex`, `.pdf`, and prompt logs
- `reference_resume/` - reference resume PDFs

## Requirements

- Python 3.10+
- `uv`
- TeX compiler (`tectonic` or `xelatex`) for PDF compilation
- Vertex AI credentials and access

## Setup

```bash
uv sync
export VERTEXAI_PROJECT="ieor-oncloud"
export VERTEXAI_LOCATION="us-central1"
```

## Run Streamlit App

```bash
uv run streamlit run main_code/app.py
```

## CLI Commands

### Build full resume (bullets + TeX + PDF)

```bash
uv run build-resume --log-prompts
```

Defaults:
- JD: `data/JD.txt`
- Template: `data/main.tex`
- Output: `output/`

### Generate bullets for one company

```bash
uv run resume-bullets --project-file data/work_agoda_2-2.json --jd data/JD.txt --log-prompts
```

### Generate bullets for all companies (JSON output)

```bash
uv run resume-bullets --all --jd data/JD.txt --log-prompts
```

## Docker

Build:

```bash
docker build -t resume-customization .
```

Run:

```bash
docker run -p 8080:8080 \
  -e VERTEXAI_PROJECT="ieor-oncloud" \
  -e VERTEXAI_LOCATION="us-central1" \
  resume-customization
```
