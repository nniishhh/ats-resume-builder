# `data/` — local inputs

Everything in here except `main.example.tex` and this file is **gitignored**. The pipeline reads
real work history, job descriptions, and a skills inventory at runtime, and none of that belongs
in a public repo.

To run the pipeline you need to create these locally.

## `main.tex`

Copy `main.example.tex` to `main.tex` and fill in your details. It documents the four regex
anchors the pipeline injects into — keep them intact.

## `work_<company>_<min>-<max>.json`

Per-role evidence. The filename encodes the company key and how many bullets to generate, so
`work_acme_2-3.json` produces 2–3 bullets for a heading in `main.tex` containing "Acme".

An array of project objects:

```json
[
  {
    "project_id": "acme_pipeline_rebuild",
    "title": "Short human title",
    "company": "Acme Corp",
    "problem": "What was wrong, and why it mattered.",
    "actions": ["What you did", "Another thing you did"],
    "results": [
      "Cut processing time 40% (framing variants: 40% faster processing)"
    ],
    "tools": ["Python", "SQL"],
    "keywords": ["Data Engineering", "Automation"],
    "example_bullets": ["A human-proofread bullet. Treated as authoritative."],
    "harvested_bullets": ["Alternate phrasing, used for inspiration."]
  }
]
```

Notes:

- `results` entries may carry `(framing variants: ...)` — alternative truthful phrasings of the
  **same** metric. The generator picks exactly one based on JD fit.
- `example_bullets` are trusted references; the generator leans on their phrasing.
- Every number that appears in a generated bullet must appear somewhere in this file, or the
  grounding validator rejects it. Metrics live here, not in the prompt.

## `proj_academic_<min>-<max>.json`

Project pool, selected from rather than generated:

```json
[
  { "Topic": "Your University - Project Name (Year)",
    "Bullet": ["The bullet text, used verbatim."],
    "Link": "github.com/you/repo" }
]
```

## `skills_inventory.json`

The Skills whitelist. The selector may only choose names that appear here verbatim; anything
else is dropped in code and logged in the build report. Each skill carries an `evidence` pointer
back to a `project_id`, a repo, or `user-confirmed`.

**Add the evidence before adding the skill.** This file is the reason an unsupported skill cannot
reach a PDF, so it only works if entries are honest.

```json
{
  "categories": [
    { "id": "programming", "label": "Programming & Data",
      "skills": [ { "name": "Python", "evidence": "acme_pipeline_rebuild" } ] }
  ]
}
```

## `JD.txt`

The target job description. Either plain text, or a header block:

```
company_name: Acme Corp
position_name: Data Scientist
---
<job description text>
```
