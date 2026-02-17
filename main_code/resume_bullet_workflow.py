import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import litellm


FILENAME_PATTERN = re.compile(
    r"^work_(?P<company>.+)_(?P<min_bullets>\d+)-(?P<max_bullets>\d+)\.json$"
)
DEFAULT_MODEL = "vertex_ai/gemini-3-pro-preview"
MIN_BULLET_CHARS = 190
MAX_BULLET_CHARS = 230
MAX_GENERATION_ATTEMPTS = 2
CHARS_PER_TOKEN = 4
THINKING_MULTIPLIER = 10
TOKEN_BUFFER = 1000
MIN_MAX_TOKENS = 4000
DEFAULT_TOP_COURSE_COUNT = 4
DEFAULT_COLUMBIA_COURSES = [
    "Applied machine learning",
    "Optimization models",
    "Analytics on the cloud (Spark)",
    "Business analytics",
    "Operations strategy",
    "Stochastic models",
    "Project management",
    "Agentic AI",
]
GENERATION_MODES: Tuple[Literal["single_prompt", "sequential"], ...] = (
    "single_prompt",
    "sequential",
)
DEFAULT_GENERATION_MODE: Literal["single_prompt", "sequential"] = "sequential"
GLOBAL_VERTEX_PREVIEW_MODELS = {
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
}


def parse_filename(project_file: Path) -> Tuple[str, int, int]:
    match = FILENAME_PATTERN.match(project_file.name)
    if not match:
        raise ValueError(
            "Project filename must match: work_<company>_<min>-<max>.json"
        )

    company = match.group("company")
    min_bullets = int(match.group("min_bullets"))
    max_bullets = int(match.group("max_bullets"))
    if min_bullets <= 0 or max_bullets <= 0 or min_bullets > max_bullets:
        raise ValueError("Invalid bullet range in filename.")
    return company, min_bullets, max_bullets


def _parse_header_body(content: str) -> dict | None:
    """Parse a header+body JD format separated by '---'.

    Expected format::

        company_name: Acme
        position_name: SWE Intern
        ---
        Full job description text …

    Returns dict with company_name, position_name, job_description
    or None if the format doesn't match.
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


def read_jd(jd_path: Path) -> str:
    """Read JD file (JSON, header+body, or plain text) and return the JD text."""
    content = jd_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"JD file is empty: {jd_path}")

    # Try JSON first
    try:
        data = json.loads(content)
        return data.get("job_description", content)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try header+body format (company_name: … / --- / body)
    parsed = _parse_header_body(content)
    if parsed:
        return parsed.get("job_description", content)

    # Fall back to raw text
    return content


def read_projects(project_path: Path) -> List[Dict[str, Any]]:
    raw = project_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Project JSON is empty: {project_path}")

    data = json.loads(raw)
    if not isinstance(data, list) or not data:
        raise ValueError("Project JSON must be a non-empty array.")
    return data


def extract_jd_signals(jd_text: str, model: str) -> Dict[str, Any]:
    system_prompt = (
        "You are a precise job-description analyzer. "
        "Return ONLY valid JSON with no extra text."
    )
    user_prompt = json.dumps(
        {
            "task": "Analyze this job description and extract structured signals.",
            "output_schema": {
                "role_type": "string — the job title / role type",
                "required_skills": "list of strings — technical skills, tools, languages mentioned",
                "domain_keywords": "list of strings — industry/domain terms and concepts",
            },
            "rules": [
                "Extract all technical skills, tools, frameworks, and languages.",
                "Extract domain-specific terms (e.g. autonomous vehicles, logistics, fintech).",
                "Infer the role type from the title and description.",
                "Return ONLY the JSON object. No markdown fences, no explanation.",
            ],
            "job_description": jd_text,
        },
        ensure_ascii=True,
    )

    raw = call_vertex_litellm(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    try:
        signals = json.loads(cleaned)
    except json.JSONDecodeError:
        signals = {
            "role_type": "Unknown",
            "required_skills": [],
            "domain_keywords": [],
        }

    for key in ("role_type", "required_skills", "domain_keywords"):
        if key not in signals:
            signals[key] = [] if key != "role_type" else "Unknown"

    return signals


def _normalize_course_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _extract_selected_courses_from_raw(
    raw_output: str,
    course_pool: Sequence[str],
) -> List[str]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    candidates: List[str] = []
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, list):
            candidates = [str(item).strip() for item in payload]
        elif isinstance(payload, dict):
            for key in ("selected_courses", "courses", "top_courses"):
                value = payload.get(key)
                if isinstance(value, list):
                    candidates = [str(item).strip() for item in value]
                    break
    except json.JSONDecodeError:
        pass

    if not candidates:
        for line in cleaned.splitlines():
            item = re.sub(r"^\s*(?:\d+[.)-]?|[-*])\s*", "", line).strip()
            if item:
                candidates.append(item)

    canonical_by_name = {
        _normalize_course_name(course): course for course in course_pool
    }
    selected: List[str] = []
    for candidate in candidates:
        normalized = _normalize_course_name(candidate)
        matched = canonical_by_name.get(normalized)
        if not matched:
            for normalized_course, original in canonical_by_name.items():
                if normalized and (
                    normalized in normalized_course or normalized_course in normalized
                ):
                    matched = original
                    break
        if matched and matched not in selected:
            selected.append(matched)

    return selected


def _fallback_rank_courses(
    jd_text: str,
    course_pool: Sequence[str],
    top_k: int,
) -> List[str]:
    jd_lower = jd_text.lower()
    keyword_hints: Dict[str, List[str]] = {
        "Applied machine learning": [
            "machine learning",
            "ml",
            "model",
            "predictive",
            "classification",
            "regression",
            "deep learning",
            "nlp",
            "llm",
        ],
        "Optimization models": [
            "optimization",
            "optimize",
            "linear programming",
            "integer programming",
            "solver",
            "objective",
            "constraints",
        ],
        "Analytics on the cloud (Spark)": [
            "spark",
            "cloud",
            "distributed",
            "databricks",
            "big data",
            "pyspark",
            "etl",
        ],
        "Business analytics": [
            "analytics",
            "dashboard",
            "kpi",
            "insight",
            "business",
            "metrics",
        ],
        "Operations strategy": [
            "operations",
            "strategy",
            "process improvement",
            "supply chain",
            "efficiency",
            "planning",
        ],
        "Stochastic models": [
            "stochastic",
            "probability",
            "random",
            "uncertainty",
            "monte carlo",
            "markov",
        ],
        "Project management": [
            "project management",
            "stakeholder",
            "timeline",
            "delivery",
            "cross-functional",
            "roadmap",
        ],
        "Agentic AI": [
            "agentic",
            "agents",
            "autonomous",
            "tool use",
            "orchestration",
            "llm agents",
        ],
    }

    scored: List[Tuple[int, int, str]] = []
    for idx, course in enumerate(course_pool):
        score = 0
        for keyword in keyword_hints.get(course, []):
            if keyword in jd_lower:
                score += 2 if " " in keyword else 1
        scored.append((score, idx, course))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [course for _, _, course in scored]
    if all(score == 0 for score, _, _ in scored):
        ranked = list(course_pool)
    return ranked[:top_k]


def select_top_courses_for_jd(
    jd_text: str,
    model: str,
    courses: Sequence[str] | None = None,
    top_k: int = DEFAULT_TOP_COURSE_COUNT,
) -> List[str]:
    course_pool = list(courses) if courses else list(DEFAULT_COLUMBIA_COURSES)
    if not course_pool:
        return []
    top_k = max(1, min(top_k, len(course_pool)))

    system_prompt = (
        "You are a precise course-matching assistant. "
        "Choose the most job-relevant courses from the provided list only. "
        "Return ONLY valid JSON with no extra text."
    )
    user_prompt = {
        "task": "Select the top job-relevant courses for this job description.",
        "selection_rules": [
            f"Select exactly {top_k} course names.",
            "Use only names from the provided course list.",
            "Prioritize direct technical and domain fit to the JD requirements.",
            "Do not invent or rename courses.",
        ],
        "output_schema": {"selected_courses": [f"exactly {top_k} course names"]},
        "courses": course_pool,
        "job_description": jd_text,
    }

    try:
        raw = call_vertex_litellm(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        selected = _extract_selected_courses_from_raw(raw, course_pool)
    except Exception:
        selected = []

    fallback = _fallback_rank_courses(jd_text=jd_text, course_pool=course_pool, top_k=top_k)
    merged = selected + [course for course in fallback if course not in selected]
    return merged[:top_k]


def extract_numbered_bullets(text: str) -> List[str]:
    bullets: List[str] = []
    current: List[str] = []

    for line in text.strip().splitlines():
        numbered = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if numbered:
            if current:
                bullets.append(" ".join(part.strip() for part in current).strip())
                current = []
            current.append(numbered.group(1).strip())
        elif current and line.strip():
            current.append(line.strip())

    if current:
        bullets.append(" ".join(part.strip() for part in current).strip())
    return [bullet for bullet in bullets if bullet]


def extract_starting_verbs(bullets: Sequence[str]) -> List[str]:
    verbs: List[str] = []
    for bullet in bullets:
        match = re.match(r"^\s*([A-Za-z]+)", bullet)
        if not match:
            continue
        verb = match.group(1).lower()
        if verb:
            verbs.append(verb)
    return verbs


def validate_bullets(
    bullets: List[str],
    min_bullets: int,
    max_bullets: int,
    forbidden_verbs: Sequence[str] | None = None,
) -> List[str]:
    issues: List[str] = []
    if len(bullets) < min_bullets or len(bullets) > max_bullets:
        issues.append(f"Bullet count must be between {min_bullets} and {max_bullets}.")

    starts = extract_starting_verbs(bullets)
    if len(starts) != len(set(starts)):
        issues.append("Starting action verbs must be unique across bullets.")
    if forbidden_verbs:
        forbidden = {verb.lower() for verb in forbidden_verbs}
        repeated = [verb for verb in starts if verb in forbidden]
        if repeated:
            repeated_unique = sorted(set(repeated))
            issues.append(
                "Starting action verbs must not reuse previously used verbs: "
                + ", ".join(repeated_unique)
                + "."
            )

    if len(set(bullets)) != len(bullets):
        issues.append("Bullets must not duplicate ideas.")

    short_bullets = [
        idx + 1 for idx, bullet in enumerate(bullets) if len(bullet) < MIN_BULLET_CHARS
    ]
    if short_bullets:
        issues.append(
            f"Each bullet must be at least {MIN_BULLET_CHARS} characters "
            f"(too short: {short_bullets})."
        )
    long_bullets = [
        idx + 1 for idx, bullet in enumerate(bullets) if len(bullet) > MAX_BULLET_CHARS
    ]
    if long_bullets:
        issues.append(
            f"Each bullet must be at most {MAX_BULLET_CHARS} characters "
            f"(too long: {long_bullets})."
        )

    return issues


def canonize_numbered_list(bullets: List[str]) -> str:
    return "\n".join(f"{idx}. {bullet}" for idx, bullet in enumerate(bullets, start=1))


def _normalize_vertex_model_name(model: str) -> str:
    if model.startswith("vertex_ai/"):
        return model.split("/", 1)[1].strip().lower()
    return model.strip().lower()


def _is_global_vertex_preview_model(model: str) -> bool:
    return _normalize_vertex_model_name(model) in GLOBAL_VERTEX_PREVIEW_MODELS


def _is_location_mismatch_error(exc: Exception) -> bool:
    text = str(exc).lower()
    location_terms = ("location", "region", "global")
    failure_terms = (
        "not found",
        "unsupported",
        "invalid",
        "unavailable",
        "access",
        "permission",
    )
    return any(term in text for term in location_terms) and any(
        term in text for term in failure_terms
    )


def call_vertex_litellm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    reasoning_effort: str = "high",
) -> str:
    vertex_project = os.getenv("VERTEXAI_PROJECT")
    vertex_location = os.getenv("VERTEXAI_LOCATION", "global")
    if not vertex_project:
        raise EnvironmentError(
            "Set VERTEXAI_PROJECT environment variable."
        )

    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "vertex_project": vertex_project,
        "vertex_location": vertex_location,
        "timeout": 150,
        "reasoning_effort": reasoning_effort,
    }

    try:
        response = litellm.completion(**request_kwargs)
    except Exception as exc:
        can_retry_global = (
            vertex_location != "global"
            and _is_global_vertex_preview_model(model)
            and _is_location_mismatch_error(exc)
        )
        if not can_retry_global:
            raise
        request_kwargs["vertex_location"] = "global"
        response = litellm.completion(**request_kwargs)

    return (
        response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    )


def write_prompt_log(
    system_prompt: str,
    user_prompt: Dict[str, Any],
    stage: str,
) -> Path:
    base_dir = Path(os.getenv("RESUME_WORK_DIR", Path.cwd()))
    output_dir = Path(os.getenv("RESUME_OUTPUT_DIR", base_dir / "output"))
    log_dir = output_dir / "prompt_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.json"
    payload = {
        "stage": stage,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    log_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return log_path


def generate_bullets(
    jd_text: str,
    project_file: Path,
    projects: List[Dict[str, Any]],
    model: str,
    log_prompts: bool = False,
    used_verbs: List[str] | None = None,
) -> str:
    company, min_bullets, max_bullets = parse_filename(project_file)
    jd_signals = extract_jd_signals(jd_text, model=model)
    tokens_per_bullet = (MAX_BULLET_CHARS // CHARS_PER_TOKEN) * THINKING_MULTIPLIER
    max_tokens = max(MIN_MAX_TOKENS, max_bullets * tokens_per_bullet + TOKEN_BUFFER)

    system_prompt = f"""You write concise, ATS-optimized resume bullets grounded strictly in the provided project evidence and job description.

========================================
OUTPUT CONTRACT (NON-NEGOTIABLE)
========================================
- Return ONLY a numbered list.
- No markdown, commentary, explanations, or code fences.
- Format: "1. First bullet\\n2. Second bullet"

========================================
HARD LENGTH RULE (NON-NEGOTIABLE)
========================================
- Each bullet MUST be between {MIN_BULLET_CHARS} and {MAX_BULLET_CHARS} characters (including spaces).
- Bullets outside this range are invalid.

========================================
TRUTHFULNESS
========================================
- Do NOT invent metrics, tools, scope, stakeholders, or outcomes.
- Every statement must be logically supported by the evidence.
- You may paraphrase and reframe, but never fabricate.

========================================
ADAPTIVE FRAMING RULE
========================================
Let the job description determine what to emphasize and how to structure bullets.
- If the JD emphasizes ownership, strategy, prioritization, or cross-functional delivery, structure bullets to highlight leadership, decisions, and business outcomes.
- If the JD emphasizes technical execution, analytics, or modeling, structure bullets to highlight methods, systems, and measurable impact.
Choose the most appropriate framing without adding unsupported details.

========================================
PROFESSIONAL ABSTRACTION
========================================
You may elevate specific implementations into accurate higher-level professional terminology when supported by evidence (e.g., "pipeline," "system," "AI","ML").
Do NOT exaggerate beyond what the evidence supports.

========================================
JOB DESCRIPTION ALIGNMENT
========================================
- Use the JD to guide emphasis and terminology.
- Prefer JD-aligned language only when it fits the evidence.
- Do NOT force keywords that are not logically connected to the work.

========================================
BULLET ORDERING
========================================
- Order bullets by strongest alignment to the JD.
- If relevance is similar, order by measurable impact and scale.

========================================
STYLE PRINCIPLES
========================================
- Prefer concrete nouns and outcomes over vague corporate phrasing.
- Use strong, varied action verbs.
- When a bullet includes a numeric or measurable result, you should place that result at the end of the sentence.
- Avoid repetition."""

    user_prompt = {
        "task": f"Generate {min_bullets}-{max_bullets} resume bullets for {company}.",
        "company": company,
        "min_bullets": min_bullets,
        "max_bullets": max_bullets,
        "jd_analysis": jd_signals,
        "job_description": jd_text,
        "project_evidence": projects,
        "already_used_verbs": used_verbs or [],
    }
    # Minimal constraints for repair flow
    repair_constraints = {
        "company": company,
        "min_bullets": min_bullets,
        "max_bullets": max_bullets,
        "HARD_LIMIT_min_characters_per_bullet": MIN_BULLET_CHARS,
        "HARD_LIMIT_max_characters_per_bullet": MAX_BULLET_CHARS,
    }
    if log_prompts:
        log_path = write_prompt_log(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage="initial",
        )
        print(f"[prompt-log] initial prompts saved to: {log_path}", file=sys.stderr)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
    ]
    first_pass = call_vertex_litellm(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    candidate_bullets = extract_numbered_bullets(first_pass)
    issues = validate_bullets(
        candidate_bullets,
        min_bullets,
        max_bullets,
        forbidden_verbs=used_verbs,
    )
    if not issues:
        return canonize_numbered_list(candidate_bullets)

    latest_output = first_pass
    for attempt in range(2, MAX_GENERATION_ATTEMPTS + 1):
        repair_payload = {
            "instruction": (
                "Regenerate the entire output to satisfy all constraints exactly. "
                "Do not explain. Return only a numbered list."
            ),
            "attempt": attempt,
            "issues": issues,
            "previous_output": latest_output,
            "constraints": repair_constraints,
            "already_used_verbs": used_verbs or [],
            "project_evidence": projects,
            "output_rule": "Numbered list only. No explanations.",
        }
        repair_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=True)},
        ]
        if log_prompts:
            repair_log_path = write_prompt_log(
                system_prompt=system_prompt,
                user_prompt=repair_payload,
                stage="repair",
            )
            print(
                f"[prompt-log] repair prompts saved to: {repair_log_path}", file=sys.stderr
            )

        latest_output = call_vertex_litellm(
            model=model,
            messages=repair_messages,
            temperature=0.15,
            max_tokens=max_tokens,
        )
        candidate_bullets = extract_numbered_bullets(latest_output)
        issues = validate_bullets(
            candidate_bullets,
            min_bullets,
            max_bullets,
            forbidden_verbs=used_verbs,
        )
        if not issues:
            return canonize_numbered_list(candidate_bullets)

    if candidate_bullets:
        print(
            "[warning] Returning latest model output after max attempts; "
            f"remaining issues: {issues}",
            file=sys.stderr,
        )
        return canonize_numbered_list(candidate_bullets)

    raise ValueError("Model did not return parseable numbered bullets.")


def generate_all_bullets(
    jd_text: str,
    company_data: List[Dict[str, Any]],
    model: str,
    log_prompts: bool = False,
) -> Dict[str, List[str]]:
    jd_signals = extract_jd_signals(jd_text, model=model)

    total_bullets = sum(c["max_bullets"] for c in company_data)
    tokens_per_bullet = (MAX_BULLET_CHARS // CHARS_PER_TOKEN) * THINKING_MULTIPLIER
    max_tokens = max(MIN_MAX_TOKENS, total_bullets * tokens_per_bullet + TOKEN_BUFFER)

    system_prompt = f"""You write concise, ATS-optimized resume bullets grounded strictly in the provided project evidence and job description.

========================================
OUTPUT CONTRACT (NON-NEGOTIABLE)
========================================
- Return ONLY valid JSON.
- No markdown, commentary, explanations, or code fences.
- JSON must be a single object:
  - Keys = company names (strings)
  - Values = arrays of bullet strings

Example:
{{"Company A": ["bullet 1", "bullet 2"]}}


========================================
HARD LENGTH RULE (NON-NEGOTIABLE)
========================================
- Each bullet MUST be between {MIN_BULLET_CHARS} and {MAX_BULLET_CHARS} characters (including spaces).
- Bullets outside this range are invalid.

========================================
TRUTHFULNESS
========================================
- Do NOT invent metrics, tools, scope, stakeholders, or outcomes.
- Every statement must be logically supported by the evidence.
- You may paraphrase and reframe, but never fabricate.

========================================
ADAPTIVE FRAMING RULE
========================================
Let the job description determine what to emphasize and how to structure bullets.
- If the JD emphasizes ownership, strategy, prioritization, or cross-functional delivery, structure bullets to highlight leadership, decisions, and business outcomes.
- If the JD emphasizes technical execution, analytics, or modeling, structure bullets to highlight methods, systems, and measurable impact.
Choose the most appropriate framing without adding unsupported details.

========================================
PROFESSIONAL ABSTRACTION
========================================
You may elevate specific implementations into accurate higher-level professional terminology when supported by evidence (e.g., "pipeline," "system," "AI","ML").
Do NOT exaggerate beyond what the evidence supports.

========================================
JOB DESCRIPTION ALIGNMENT
========================================
- Use the JD to guide emphasis and terminology.
- Prefer JD-aligned language only when it fits the evidence.
- Do NOT force keywords that are not logically connected to the work.

========================================
BULLET ORDERING
========================================
- Within each company, order bullets by strongest alignment to the JD.
- If relevance is similar, order by measurable impact and scale.

========================================
STYLE PRINCIPLES
========================================
- Prefer concrete nouns and outcomes over vague corporate phrasing.
- Use strong, varied action verbs.
- When a bullet includes a numeric or measurable result, you should place that result at the end of the sentence.
- Avoid repetition within each company."""

    companies_spec = []
    for c in company_data:
        companies_spec.append({
            "company": c["company"],
            "min_bullets": c["min_bullets"],
            "max_bullets": c["max_bullets"],
            "project_evidence": c["projects"],
        })

    user_prompt = {
        "task": "Generate resume bullets for ALL companies below in one JSON response.",
        "jd_analysis": jd_signals,
        "job_description": jd_text,
        "companies": companies_spec,
    }

    if log_prompts:
        log_path = write_prompt_log(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage="initial",
        )
        print(f"[prompt-log] initial prompts saved to: {log_path}", file=sys.stderr)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
    ]

    raw = call_vertex_litellm(model=model, messages=messages, max_tokens=max_tokens)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    try:
        results = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError("Model did not return valid JSON.")

    if not isinstance(results, dict):
        raise ValueError("Model output is not a JSON object.")

    return results


def run_all(
    jd_path: Path,
    directory: Path,
    model: str,
    log_prompts: bool,
    generation_mode: Literal["single_prompt", "sequential"] = DEFAULT_GENERATION_MODE,
) -> Dict[str, List[str]]:
    jd_text = read_jd(jd_path)
    project_files = sorted(directory.glob("work_*_*-*.json"))
    if not project_files:
        raise FileNotFoundError(f"No work_*_<min>-<max>.json files found in {directory}")

    if generation_mode not in GENERATION_MODES:
        raise ValueError(
            f"Invalid generation_mode '{generation_mode}'. "
            f"Expected one of: {', '.join(GENERATION_MODES)}."
        )

    company_data: List[Dict[str, Any]] = []
    for project_file in project_files:
        company, min_b, max_b = parse_filename(project_file)
        projects = read_projects(project_file)
        company_data.append({
            "company": company,
            "min_bullets": min_b,
            "max_bullets": max_b,
            "projects": projects,
            "project_file": project_file,
        })
        print(
            f"[info] Loaded {project_file.name} ({company}, {min_b}-{max_b} bullets)",
            file=sys.stderr,
        )

    if generation_mode == "single_prompt":
        print("[info] Generation mode: single_prompt", file=sys.stderr)
        print("[info] Generating all bullets in one call ...", file=sys.stderr)
        generate_input = [
            {
                "company": c["company"],
                "min_bullets": c["min_bullets"],
                "max_bullets": c["max_bullets"],
                "projects": c["projects"],
            }
            for c in company_data
        ]
        return generate_all_bullets(
            jd_text=jd_text,
            company_data=generate_input,
            model=model,
            log_prompts=log_prompts,
        )

    print("[info] Generation mode: sequential", file=sys.stderr)
    results: Dict[str, List[str]] = {}
    used_verbs: List[str] = []

    for c in company_data:
        company = c["company"]
        project_file = c["project_file"]
        projects = c["projects"]
        print(f"[info] Generating bullets for {company} ...", file=sys.stderr)
        output = generate_bullets(
            jd_text=jd_text,
            project_file=project_file,
            projects=projects,
            model=model,
            log_prompts=log_prompts,
            used_verbs=used_verbs,
        )
        company_bullets = extract_numbered_bullets(output)
        results[company] = company_bullets

        company_verbs = extract_starting_verbs(company_bullets)
        for verb in company_verbs:
            if verb not in used_verbs:
                used_verbs.append(verb)

        print(
            f"[info] {company} done. Tracked starting verbs: {used_verbs}",
            file=sys.stderr,
        )

    return results


def run_all_with_course_selection(
    jd_path: Path,
    directory: Path,
    model: str,
    log_prompts: bool,
    generation_mode: Literal["single_prompt", "sequential"] = DEFAULT_GENERATION_MODE,
    courses: Sequence[str] | None = None,
    top_k: int = DEFAULT_TOP_COURSE_COUNT,
) -> Tuple[Dict[str, List[str]], List[str]]:
    jd_text = read_jd(jd_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        bullets_future = executor.submit(
            run_all,
            jd_path,
            directory,
            model,
            log_prompts,
            generation_mode,
        )
        courses_future = executor.submit(
            select_top_courses_for_jd,
            jd_text,
            model,
            courses,
            top_k,
        )
        bullets = bullets_future.result()
        selected_courses = courses_future.result()
    return bullets, selected_courses


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate resume bullets using Vertex AI via LiteLLM."
    )
    parser.add_argument(
        "--jd",
        type=Path,
        default=Path("data/JD.txt"),
        help="Path to job description text file.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--project-file",
        type=Path,
        help="Path to a single work_<company>_<min>-<max>.json file.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all work_*.json files in the JD directory and output JSON.",
    )
    group.add_argument(
        "--top-courses-only",
        action="store_true",
        help="Return only the top JD-relevant course names.",
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
        help="Write system/user prompts to temporary log files.",
    )
    parser.add_argument(
        "--generation-mode",
        type=str,
        choices=list(GENERATION_MODES),
        default=DEFAULT_GENERATION_MODE,
        help=(
            "How to generate bullets for --all: "
            "'single_prompt' (all companies in one prompt) or "
            "'sequential' (one company at a time with verb tracking)."
        ),
    )
    parser.add_argument(
        "--courses",
        nargs="*",
        default=DEFAULT_COLUMBIA_COURSES,
        help="Course names to rank for JD relevance (default: built-in Columbia list).",
    )
    parser.add_argument(
        "--top-k-courses",
        type=int,
        default=DEFAULT_TOP_COURSE_COUNT,
        help=f"Number of courses to return (default: {DEFAULT_TOP_COURSE_COUNT}).",
    )
    args = parser.parse_args()

    try:
        if args.top_courses_only:
            jd_text = read_jd(args.jd)
            selected_courses = select_top_courses_for_jd(
                jd_text=jd_text,
                model=args.model,
                courses=args.courses,
                top_k=args.top_k_courses,
            )
            print(json.dumps(selected_courses, indent=2, ensure_ascii=False))
        elif args.all:
            directory = args.jd.parent
            results = run_all(
                jd_path=args.jd,
                directory=directory,
                model=args.model,
                log_prompts=args.log_prompts,
                generation_mode=args.generation_mode,
            )
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            jd_text = read_jd(args.jd)
            projects = read_projects(args.project_file)
            output = generate_bullets(
                jd_text=jd_text,
                project_file=args.project_file,
                projects=projects,
                model=args.model,
                log_prompts=args.log_prompts,
            )
            print(output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
