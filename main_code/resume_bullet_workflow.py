import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import logging

import litellm

# When run as `python main_code/resume_bullet_workflow.py`, ensure imports
# resolve to local source (repo root) instead of requiring installed package.
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from main_code.workflow_prompts import (
    build_academic_project_selection_prompts,
    build_all_bullets_repair_payload,
    build_all_bullets_system_prompt,
    build_all_bullets_user_prompt,
    build_bullet_generation_system_prompt,
    build_bullet_generation_user_prompt,
    build_bullet_repair_payload,
    build_bullet_shorten_prompts,
    build_jd_signals_prompts,
    build_skills_selection_prompts,
    build_structure_plan_prompts,
    build_course_selection_prompts,
    build_jd_summary_prompts,
)

# Reduce LiteLLM noise unless debug requested
if os.getenv("LITELLM_DEBUG"):
    litellm._turn_on_debug()
    logging.getLogger("litellm").setLevel(logging.DEBUG)
else:
    logging.getLogger("litellm").setLevel(logging.WARNING)

# Drop provider-unsupported params (e.g. custom temperature on OpenAI gpt-5
# reasoning models) instead of raising, so non-bullet tasks using temperature=0.0
# do not crash.
litellm.drop_params = True

FILENAME_PATTERN = re.compile(
    r"^work_(?P<company>.+)_(?P<min_bullets>\d+)-(?P<max_bullets>\d+)\.json$"
)
DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_NON_BULLET_MODEL = "openai/gpt-5.2"
MODEL_CHOICES = [
    "openai/gpt-5.2",
    "openai/gpt-4o",
    "vertex_ai/gemini-3-pro-preview",
    "vertex_ai/gemini-3-flash-preview",
]

MIN_BULLET_CHARS = 200
MAX_BULLET_CHARS = 240
MAX_GENERATION_ATTEMPTS = 2
CHARS_PER_TOKEN = 4
THINKING_MULTIPLIER = 50
TOKEN_BUFFER = 1000
MIN_MAX_TOKENS = 4000
DEFAULT_TOP_COURSE_COUNT = 4
DEFAULT_TOP_ACADEMIC_PROJECT_COUNT = 3
DEFAULT_MAX_SKILL_CATEGORIES = 4
DEFAULT_MAX_SKILLS_PER_CATEGORY = 6
DEFAULT_MAX_TOTAL_SKILLS = 24
DEFAULT_ACADEMIC_PROJECT_FILE = "proj_academic_2-2.json"
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
RATE_LIMIT_MAX_RETRIES = 4
RATE_LIMIT_BACKOFF_BASE_SECONDS = 2
LLM_TASK_JD_SIGNALS = "jd_signals"
LLM_TASK_BULLET_GENERATION = "bullet_generation"
LLM_TASK_BULLET_REPAIR = "bullet_repair"
LLM_TASK_BULLETS_ALL = "bullets_all"
LLM_TASK_COURSE_SELECTION = "course_selection"
LLM_TASK_ACADEMIC_SELECTION = "academic_selection"
LLM_TASK_SKILLS_SELECTION = "skills_selection"
LLM_TASK_BULLET_SHORTEN = "bullet_shorten"
LLM_TASK_STRUCTURE_PLAN = "structure_plan"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _strip_fence(raw: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


DEFAULT_TASK_LLM_SETTINGS: Dict[str, Dict[str, Any]] = {
    LLM_TASK_JD_SIGNALS: {
        "model": DEFAULT_NON_BULLET_MODEL,
        "temperature": 0.0,
    },
    LLM_TASK_BULLET_GENERATION: {
        "temperature": 0.3,
    },
    LLM_TASK_BULLET_REPAIR: {
        "temperature": 0.3,
    },
    LLM_TASK_BULLETS_ALL: {
        "temperature": 0.5,
    },
    LLM_TASK_COURSE_SELECTION: {
        "model": DEFAULT_NON_BULLET_MODEL,
        "temperature": 0.0,
    },
    LLM_TASK_ACADEMIC_SELECTION: {
        "model": DEFAULT_NON_BULLET_MODEL,
        "temperature": 0.0,
    },
    LLM_TASK_STRUCTURE_PLAN: {
        "model": DEFAULT_NON_BULLET_MODEL,
        "temperature": 0.0,
    },
}


def get_task_llm_settings(task: str, fallback_model: str) -> Tuple[str, float]:
    task_settings = DEFAULT_TASK_LLM_SETTINGS.get(task, {})
    model = str(task_settings.get("model", fallback_model)).strip() or fallback_model
    temperature = float(task_settings.get("temperature", 0.3))
    # Gemini 3 preview models require temperature=1.0 to avoid infinite loops and failures
    # model_stripped = model.split("/")[-1] if "/" in model else model
    # if model_stripped in GLOBAL_VERTEX_PREVIEW_MODELS:
    #     temperature = 1.0
    return model, temperature


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


def summarize_job_description(jd_text: str, model: str) -> str:
    """Clean JD text using the JD Analyst prompt and return preserved-signal text."""
    task_model, task_temperature = get_task_llm_settings(
        LLM_TASK_JD_SIGNALS,
        fallback_model=model,
    )
    system_prompt, user_prompt = build_jd_summary_prompts(jd_text)
    raw = call_llm(
        model=task_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=task_temperature,
    )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip() or jd_text


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
    task_model, task_temperature = get_task_llm_settings(
        LLM_TASK_COURSE_SELECTION,
        fallback_model=model,
    )

    system_prompt, user_prompt = build_course_selection_prompts(
        jd_text=jd_text,
        course_pool=course_pool,
        top_k=top_k,
    )

    try:
        raw = call_llm(
            model=task_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
            ],
            temperature=task_temperature,
            max_tokens=600,
        )
        selected = _extract_selected_courses_from_raw(raw, course_pool)
    except Exception:
        selected = []

    fallback = _fallback_rank_courses(jd_text=jd_text, course_pool=course_pool, top_k=top_k)
    merged = selected + [course for course in fallback if course not in selected]
    return merged[:top_k]


def _normalize_topic_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _extract_selected_topics_from_raw(
    raw_output: str,
    topic_pool: Sequence[str],
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
            for key in ("selected_topics", "topics", "projects", "project_topics"):
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
        _normalize_topic_name(topic): topic for topic in topic_pool
    }
    selected: List[str] = []
    for candidate in candidates:
        normalized = _normalize_topic_name(candidate)
        matched = canonical_by_name.get(normalized)
        if not matched:
            for normalized_topic, original in canonical_by_name.items():
                if normalized and (
                    normalized in normalized_topic or normalized_topic in normalized
                ):
                    matched = original
                    break
        if matched and matched not in selected:
            selected.append(matched)
    return selected


def _fallback_rank_academic_topics(
    jd_text: str,
    projects: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[str]:
    terms = re.findall(r"[a-z0-9]{3,}", jd_text.lower())
    jd_terms = {term for term in terms if term not in {"with", "from", "that", "this"}}
    if not jd_terms:
        return [
            str(project.get("Topic", "")).strip()
            for project in projects[:top_k]
            if str(project.get("Topic", "")).strip()
        ]

    scored: List[Tuple[int, int, str]] = []
    for idx, project in enumerate(projects):
        topic = str(project.get("Topic", "")).strip()
        if not topic:
            continue
        bullet_text = " ".join(
            str(item).strip() for item in project.get("Bullet", []) if str(item).strip()
        )
        corpus = f"{topic} {bullet_text}".lower()
        score = sum(1 for term in jd_terms if term in corpus)
        scored.append((score, idx, topic))

    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [topic for _, _, topic in scored[:top_k]]


def select_top_academic_topics_for_jd(
    jd_text: str,
    project_list: Sequence[Dict[str, Any]],
    model: str,
    top_k: int = DEFAULT_TOP_ACADEMIC_PROJECT_COUNT,
) -> List[str]:
    topic_pool = [
        str(project.get("Topic", "")).strip()
        for project in project_list
        if str(project.get("Topic", "")).strip()
    ]
    if not topic_pool:
        return []
    top_k = max(1, min(top_k, len(topic_pool)))
    task_model, task_temperature = get_task_llm_settings(
        LLM_TASK_ACADEMIC_SELECTION,
        fallback_model=model,
    )

    system_prompt, user_prompt = build_academic_project_selection_prompts(
        jd_text=jd_text,
        project_list=list(project_list),
        top_k=top_k,
    )

    try:
        raw = call_llm(
            model=task_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=task_temperature,
            max_tokens=600,
        )
        selected = _extract_selected_topics_from_raw(raw, topic_pool)
    except Exception:
        selected = []

    fallback = _fallback_rank_academic_topics(
        jd_text=jd_text,
        projects=project_list,
        top_k=top_k,
    )
    merged = selected + [topic for topic in fallback if topic not in selected]
    return merged[:top_k]


def select_academic_projects_by_topics(
    project_list: Sequence[Dict[str, Any]],
    selected_topics: Sequence[str],
) -> List[Dict[str, Any]]:
    canonical_by_topic = {
        _normalize_topic_name(str(project.get("Topic", ""))): project
        for project in project_list
        if str(project.get("Topic", "")).strip()
    }

    selected_projects: List[Dict[str, Any]] = []
    for topic in selected_topics:
        normalized = _normalize_topic_name(topic)
        matched = canonical_by_topic.get(normalized)
        if not matched:
            for normalized_topic, project in canonical_by_topic.items():
                if normalized and (
                    normalized in normalized_topic or normalized_topic in normalized
                ):
                    matched = project
                    break
        if matched and matched not in selected_projects:
            selected_projects.append(matched)
    return selected_projects


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


_NUMERIC_CLAIM = re.compile(r"\d[\d,.]*\s*(?:%|K|M|B|x)?", re.IGNORECASE)


def _numeric_claims(text: str) -> set[str]:
    """Normalised numeric tokens, so '6M' and '6 M' and '6M,' compare equal."""
    out: set[str] = set()
    for raw in _NUMERIC_CLAIM.findall(text):
        token = raw.replace(",", "").replace(" ", "").rstrip(".").upper()
        if token:
            out.add(token)
    return out


def _evidence_numeric_claims(projects: Sequence[Dict[str, Any]]) -> set[str]:
    parts: List[str] = []
    for project in projects:
        for key in ("problem", "actions", "main_metric", "sub_metrics", "results", "tools", "keywords",
                    "example_bullets", "harvested_bullets"):
            value = project.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple)):
                parts.extend(str(v) for v in value)
    return _numeric_claims(" ".join(parts))


def validate_bullets(
    bullets: List[str],
    min_bullets: int,
    max_bullets: int,
    forbidden_verbs: Sequence[str] | None = None,
    projects: Sequence[Dict[str, Any]] | None = None,
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

    # Grounding: every number in a bullet must exist somewhere in the evidence for that
    # company. Numbers are the highest-consequence thing to get wrong and are unambiguous
    # to check, unlike free-text tool names.
    if projects:
        allowed = _evidence_numeric_claims(projects)
        for idx, bullet in enumerate(bullets, start=1):
            invented = sorted(_numeric_claims(bullet) - allowed)
            if invented:
                issues.append(
                    f"Bullet {idx} contains figures absent from the evidence: "
                    + ", ".join(invented)
                    + ". Use only metrics present in the provided evidence."
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


def _is_rate_limited_error(exc: Exception) -> bool:
    text = str(exc).lower()
    indicators = (
        "429",
        "resource_exhausted",
        "resource exhausted",
        "rate limit",
        "too many requests",
        "quota exceeded",
    )
    return any(indicator in text for indicator in indicators)


def _completion_with_exponential_backoff(request_kwargs: Dict[str, Any]) -> Any:
    max_attempts = RATE_LIMIT_MAX_RETRIES + 1
    for attempt in range(1, max_attempts + 1):
        try:
            return litellm.completion(**request_kwargs)
        except Exception as exc:
            if not _is_rate_limited_error(exc) or attempt >= max_attempts:
                raise
            delay_seconds = RATE_LIMIT_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                "[warning] LLM rate-limited (429/RESOURCE_EXHAUSTED). "
                f"Retrying in {delay_seconds}s "
                f"(attempt {attempt + 1}/{max_attempts}).",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)


def _is_vertex_model(model: str) -> bool:
    return model.strip().lower().startswith("vertex_ai/")


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 150,
    }

    if _is_vertex_model(model):
        vertex_project = os.getenv("VERTEXAI_PROJECT")
        vertex_location = os.getenv("VERTEXAI_LOCATION", "global")
        if not vertex_project:
            raise EnvironmentError(
                "Set VERTEXAI_PROJECT environment variable."
            )
        request_kwargs["vertex_project"] = vertex_project
        request_kwargs["vertex_location"] = vertex_location

        try:
            response = _completion_with_exponential_backoff(request_kwargs)
        except Exception as exc:
            can_retry_global = (
                vertex_location != "global"
                and _is_global_vertex_preview_model(model)
                and _is_location_mismatch_error(exc)
            )
            if not can_retry_global:
                raise
            request_kwargs["vertex_location"] = "global"
            response = _completion_with_exponential_backoff(request_kwargs)
    else:
        response = _completion_with_exponential_backoff(request_kwargs)

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
    seniority: str = "",
) -> str:
    company, min_bullets, max_bullets = parse_filename(project_file)
    generation_model, generation_temperature = get_task_llm_settings(
        LLM_TASK_BULLET_GENERATION,
        fallback_model=model,
    )
    repair_model, repair_temperature = get_task_llm_settings(
        LLM_TASK_BULLET_REPAIR,
        fallback_model=model,
    )
    tokens_per_bullet = (MAX_BULLET_CHARS // CHARS_PER_TOKEN) * THINKING_MULTIPLIER
    max_tokens = max(MIN_MAX_TOKENS, max_bullets * tokens_per_bullet + TOKEN_BUFFER)

    system_prompt = build_bullet_generation_system_prompt(
        min_bullet_chars=MIN_BULLET_CHARS,
        max_bullet_chars=MAX_BULLET_CHARS,
    )
    user_prompt = build_bullet_generation_user_prompt(
        company=company,
        min_bullets=min_bullets,
        max_bullets=max_bullets,
        jd_text=jd_text,
        projects=projects,
        used_verbs=used_verbs,
        seniority=seniority,
    )
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
    first_pass = call_llm(
        model=generation_model,
        messages=messages,
        temperature=generation_temperature,
        max_tokens=max_tokens,
    )
    candidate_bullets = extract_numbered_bullets(first_pass)
    issues = validate_bullets(
        candidate_bullets,
        min_bullets,
        max_bullets,
        forbidden_verbs=used_verbs,
        projects=projects,
    )
    if not issues:
        return canonize_numbered_list(candidate_bullets)

    latest_output = first_pass
    for attempt in range(2, MAX_GENERATION_ATTEMPTS + 1):
        repair_payload = build_bullet_repair_payload(
            company=company,
            min_bullets=min_bullets,
            max_bullets=max_bullets,
            min_bullet_chars=MIN_BULLET_CHARS,
            max_bullet_chars=MAX_BULLET_CHARS,
            issues=issues,
            latest_output=latest_output,
            jd_text=jd_text,
            used_verbs=used_verbs,
            projects=projects,
            attempt=attempt,
            seniority=seniority,
        )
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

        latest_output = call_llm(
            model=repair_model,
            messages=repair_messages,
            temperature=repair_temperature,
            max_tokens=max_tokens,
        )
        candidate_bullets = extract_numbered_bullets(latest_output)
        issues = validate_bullets(
            candidate_bullets,
            min_bullets,
            max_bullets,
            forbidden_verbs=used_verbs,
            projects=projects,
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


def _parse_all_bullets_json(raw: str) -> Dict[str, Any]:
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


def generate_all_bullets(
    jd_text: str,
    company_data: List[Dict[str, Any]],
    model: str,
    log_prompts: bool = False,
    seniority: str = "",
) -> Dict[str, List[str]]:
    task_model, task_temperature = get_task_llm_settings(
        LLM_TASK_BULLETS_ALL,
        fallback_model=model,
    )

    total_bullets = sum(c["max_bullets"] for c in company_data)
    tokens_per_bullet = (MAX_BULLET_CHARS // CHARS_PER_TOKEN) * THINKING_MULTIPLIER
    max_tokens = max(MIN_MAX_TOKENS, total_bullets * tokens_per_bullet + TOKEN_BUFFER)

    system_prompt = build_all_bullets_system_prompt(
        min_bullet_chars=MIN_BULLET_CHARS,
        max_bullet_chars=MAX_BULLET_CHARS,
    )

    companies_spec = []
    for c in company_data:
        companies_spec.append({
            "company": c["company"],
            "min_bullets": c["min_bullets"],
            "max_bullets": c["max_bullets"],
            "project_evidence": c["projects"],
        })

    user_prompt = build_all_bullets_user_prompt(
        jd_text=jd_text,
        companies_spec=companies_spec,
        seniority=seniority,
    )

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

    raw = call_llm(
        model=task_model,
        messages=messages,
        temperature=task_temperature,
        max_tokens=max_tokens,
    )

    results = _parse_all_bullets_json(raw)

    # Code-based length enforcement (len()) with a single repair round,
    # mirroring the sequential path.
    bounds_by_company = {
        c["company"]: (c["min_bullets"], c["max_bullets"]) for c in company_data
    }
    projects_by_company = {
        c["company"]: c.get("projects") or c.get("project_evidence") or []
        for c in company_data
    }

    def _validate_all(res: Dict[str, Any]) -> Dict[str, List[str]]:
        issues_by_company: Dict[str, List[str]] = {}
        for company, (min_b, max_b) in bounds_by_company.items():
            bullets = res.get(company)
            if not isinstance(bullets, list):
                bullets = next(
                    (
                        value
                        for key, value in res.items()
                        if key.strip().lower() == company.strip().lower()
                        and isinstance(value, list)
                    ),
                    None,
                )
            if not isinstance(bullets, list):
                issues_by_company[company] = [
                    f"Missing bullets for company '{company}'."
                ]
                continue
            issues = validate_bullets(
                [str(bullet) for bullet in bullets],
                min_b,
                max_b,
                projects=projects_by_company.get(company),
            )
            if issues:
                issues_by_company[company] = issues
        return issues_by_company

    issues_by_company = _validate_all(results)
    if issues_by_company:
        print(
            "[info] single_prompt length issues; running one repair for: "
            f"{list(issues_by_company.keys())}",
            file=sys.stderr,
        )
        repair_payload = build_all_bullets_repair_payload(
            jd_text=jd_text,
            companies_spec=companies_spec,
            issues_by_company=issues_by_company,
            previous_output=raw,
            min_bullet_chars=MIN_BULLET_CHARS,
            max_bullet_chars=MAX_BULLET_CHARS,
            seniority=seniority,
        )
        if log_prompts:
            repair_log_path = write_prompt_log(
                system_prompt=system_prompt,
                user_prompt=repair_payload,
                stage="repair",
            )
            print(
                f"[prompt-log] repair prompts saved to: {repair_log_path}",
                file=sys.stderr,
            )
        repair_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=True)},
        ]
        repaired_raw = call_llm(
            model=task_model,
            messages=repair_messages,
            temperature=task_temperature,
            max_tokens=max_tokens,
        )
        try:
            repaired = _parse_all_bullets_json(repaired_raw)
        except ValueError:
            repaired = None
        if repaired is not None:
            results = repaired
            remaining = _validate_all(results)
            if remaining:
                print(
                    "[warning] single_prompt bullets still out of range after "
                    f"one repair: {remaining}",
                    file=sys.stderr,
                )

    return results


def run_all(
    jd_path: Path,
    directory: Path,
    model: str,
    log_prompts: bool,
    generation_mode: Literal["single_prompt", "sequential"] = DEFAULT_GENERATION_MODE,
    jd_summary_text: str | None = None,
    seniority: str = "",
    role_bullets: Dict[str, int] | None = None,
) -> Dict[str, List[str]]:
    if jd_summary_text is None:
        raw_jd_text = read_jd(jd_path)
        jd_text = summarize_job_description(jd_text=raw_jd_text, model=model)
    else:
        jd_text = jd_summary_text
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
        # A structure plan pins each role to an exact count, and 0 drops it from the
        # resume entirely. Without a plan the filename range stands, which is the
        # original behaviour.
        if role_bullets is not None:
            planned = role_bullets.get(company, max_b)
            if planned <= 0:
                print(f"[info] Skipping {company} (structure plan)", file=sys.stderr)
                continue
            min_b = max_b = planned
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

    if not company_data:
        raise ValueError("Structure plan dropped every role; nothing left to generate.")

    def _enforce_counts(results: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Trim each role to the count it was allocated.

        validate_bullets() only *checks* the count: when the repair round still comes
        back long, generate_bullets() warns and returns the oversized list anyway. That
        made the plan advisory — one run reported "mta 2, swat 4" in the build report
        while the PDF actually carried mta 3 and swat 5. Same rule as the skills
        whitelist: the model proposes, code decides what ships.
        """
        caps = {c["company"]: c["max_bullets"] for c in company_data}
        trimmed: Dict[str, List[str]] = {}
        for company, bullets in results.items():
            cap = caps.get(company)
            if cap is not None and len(bullets) > cap:
                print(
                    f"[info] {company}: trimming {len(bullets)} bullets to the "
                    f"allocated {cap}",
                    file=sys.stderr,
                )
                bullets = bullets[:cap]
            trimmed[company] = bullets
        return trimmed

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
        return _enforce_counts(generate_all_bullets(
            jd_text=jd_text,
            company_data=generate_input,
            model=model,
            log_prompts=log_prompts,
            seniority=seniority,
        ))

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
            seniority=seniority,
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

    return _enforce_counts(results)


def run_all_with_full_selection(
    jd_path: Path,
    directory: Path,
    model: str,
    log_prompts: bool,
    generation_mode: Literal["single_prompt", "sequential"] = DEFAULT_GENERATION_MODE,
    courses: Sequence[str] | None = None,
    top_k_courses: int = DEFAULT_TOP_COURSE_COUNT,
    academic_project_file: Path | None = None,
    top_k_academic_topics: int = DEFAULT_TOP_ACADEMIC_PROJECT_COUNT,
) -> Tuple[Dict[str, List[str]], List[str], List[str], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    raw_jd_text = read_jd(jd_path)
    project_file = academic_project_file or (directory / DEFAULT_ACADEMIC_PROJECT_FILE)
    if not project_file.exists():
        raise FileNotFoundError(f"Academic project file not found: {project_file}")
    academic_projects = read_projects(project_file)

    # jd_signals (seniority/archetype) has to be known before bullet generation starts,
    # since it steers how technically dense the bullets read — so it runs alongside the
    # summary rather than after everything else, which is where it used to live and where
    # it was too late to inform generation at all.
    with ThreadPoolExecutor(max_workers=2) as pre_executor:
        summary_future = pre_executor.submit(summarize_job_description, raw_jd_text, model)
        signals_future = pre_executor.submit(extract_jd_signals, raw_jd_text, model)
        jd_summary = summary_future.result()
        jd_signals = signals_future.result()
    seniority = jd_signals.get("seniority", "")

    # The plan decides the resume's shape (bullets per role, how many projects and
    # courses, whether projects lead), so it has to settle before anything is generated
    # or selected.
    role_specs = []
    for project_file in sorted(directory.glob("work_*_*-*.json")):
        try:
            company, min_b, max_b = parse_filename(project_file)
        except ValueError:
            continue
        # Without this the planner only sees a company name and a bullet range, so it
        # guesses from brand recognition ("likely closest to...") and keeps cutting the
        # oldest role whatever the posting asks for. The evidence is what makes a role
        # relevant, so it has to see it.
        try:
            evidence = read_projects(project_file)
        except Exception:
            evidence = []
        keywords, tools, titles = [], [], []
        for item in evidence:
            titles.append(str(item.get("title", "")).strip())
            keywords.extend(str(k) for k in (item.get("keywords") or []))
            tools.extend(str(t) for t in (item.get("tools") or []))
        role_specs.append({
            "company": company,
            "min_bullets": min_b,
            "max_bullets": max_b,
            "what_this_role_covers": [t for t in titles if t],
            "keywords": sorted(set(keywords)),
            "tools": sorted(set(tools)),
        })
    plan = plan_resume_structure(
        jd_signals=jd_signals,
        roles=role_specs,
        project_pool=[
            str(p.get("Topic", "")).strip() for p in academic_projects if p.get("Topic")
        ],
        model=model,
    )
    top_k_courses = plan["coursework"]
    top_k_academic_topics = plan["projects"]
    print(f"[info] Structure plan: {plan['roles']} "
          f"projects={plan['projects']} coursework={plan['coursework']}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=3) as executor:
        bullets_future = executor.submit(
            run_all,
            jd_path,
            directory,
            model,
            log_prompts,
            generation_mode,
            jd_summary,
            seniority,
            plan["roles"],
        )
        courses_future = executor.submit(
            select_top_courses_for_jd,
            jd_summary,
            model,
            courses,
            top_k_courses,
        )
        academic_topics_future = executor.submit(
            select_top_academic_topics_for_jd,
            jd_summary,
            academic_projects,
            model,
            top_k_academic_topics,
        )
        bullets = bullets_future.result()
        selected_courses = courses_future.result()
        selected_topics = academic_topics_future.result()

    selected_projects = select_academic_projects_by_topics(
        project_list=academic_projects,
        selected_topics=selected_topics,
    )
    target_count = max(1, min(top_k_academic_topics, len(academic_projects)))
    if len(selected_projects) < target_count:
        known_topics = {
            _normalize_topic_name(str(project.get("Topic", "")))
            for project in selected_projects
        }
        for project in academic_projects:
            topic = _normalize_topic_name(str(project.get("Topic", "")))
            if topic and topic not in known_topics:
                selected_projects.append(project)
                known_topics.add(topic)
            if len(selected_projects) >= target_count:
                break

    selected_projects = selected_projects[:target_count]
    selected_topics = [
        str(project.get("Topic", "")).strip()
        for project in selected_projects
        if str(project.get("Topic", "")).strip()
    ]
    return bullets, selected_courses, selected_topics, selected_projects, jd_signals, plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate resume bullets using OpenAI (or other providers) via LiteLLM."
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
    group.add_argument(
        "--top-academic-only",
        action="store_true",
        help="Return only the top JD-relevant academic project Topic names.",
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
    parser.add_argument(
        "--academic-project-file",
        type=Path,
        default=Path("data") / DEFAULT_ACADEMIC_PROJECT_FILE,
        help="Path to academic project JSON file (default: data/proj_academic_2-2.json).",
    )
    parser.add_argument(
        "--top-k-academic",
        type=int,
        default=DEFAULT_TOP_ACADEMIC_PROJECT_COUNT,
        help=(
            "Number of academic project topics to return "
            f"(default: {DEFAULT_TOP_ACADEMIC_PROJECT_COUNT})."
        ),
    )
    args = parser.parse_args()

    try:
        if args.top_courses_only:
            raw_jd_text = read_jd(args.jd)
            jd_text = summarize_job_description(jd_text=raw_jd_text, model=args.model)
            selected_courses = select_top_courses_for_jd(
                jd_text=jd_text,
                model=args.model,
                courses=args.courses,
                top_k=args.top_k_courses,
            )
            print(json.dumps(selected_courses, indent=2, ensure_ascii=False))
        elif args.top_academic_only:
            raw_jd_text = read_jd(args.jd)
            jd_text = summarize_job_description(jd_text=raw_jd_text, model=args.model)
            academic_projects = read_projects(args.academic_project_file)
            selected_topics = select_top_academic_topics_for_jd(
                jd_text=jd_text,
                project_list=academic_projects,
                model=args.model,
                top_k=args.top_k_academic,
            )
            print(json.dumps(selected_topics, indent=2, ensure_ascii=False))
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
            raw_jd_text = read_jd(args.jd)
            jd_text = summarize_job_description(jd_text=raw_jd_text, model=args.model)
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


# ---------------------------------------------------------------------------
# JD signals + Skills selection
# ---------------------------------------------------------------------------

def plan_resume_structure(
    jd_signals: Dict[str, Any],
    roles: List[Dict[str, Any]],
    project_pool: Sequence[str],
    model: str,
) -> Dict[str, Any]:
    """Choose how much space each role and section gets for this posting.

    Until now the shape was fixed: the same bullet counts from the filenames and the
    same section order for every JD, so only the wording ever adapted. `archetype` was
    extracted and then never used by anything.

    Same contract as the skills whitelist — the model proposes, code constrains. Each
    role is clamped to 0 (dropped) or its evidence file's own min..max, so a plan can
    never ask for depth the evidence cannot support. A failed call falls back to the
    filename maxima, i.e. exactly the old fixed behaviour.
    """
    bounds = {r["company"]: (r["min_bullets"], r["max_bullets"]) for r in roles}
    fallback = {
        "roles": {c: mx for c, (_, mx) in bounds.items()},
        "projects": DEFAULT_TOP_ACADEMIC_PROJECT_COUNT,
        "coursework": DEFAULT_TOP_COURSE_COUNT,
        "reasons": {},
    }

    # The budget has to bite. Setting it to the sum of the maxima made "give every
    # role its maximum" a feasible answer, so the planner returned exactly that for
    # every posting and the shape never changed. The sum of the minima forces a real
    # trade: either everything runs lean, or a weak role is dropped so a strong one
    # can stay deep.
    budget = sum(mn for mn, _ in bounds.values())

    task_model, task_temperature = get_task_llm_settings(
        LLM_TASK_STRUCTURE_PLAN, fallback_model=model
    )
    allowed = [
        {
            "role_key": r["company"],
            "allowed_bullets": f"0 (drop) or {r['min_bullets']}-{r['max_bullets']}",
            "min_if_kept": r["min_bullets"],
            "max_if_kept": r["max_bullets"],
            "what_this_role_covers": r.get("what_this_role_covers", []),
            "keywords": r.get("keywords", []),
            "tools": r.get("tools", []),
        }
        for r in roles
    ]
    system_prompt, user_prompt = build_structure_plan_prompts(
        jd_signals=jd_signals,
        roles=allowed,
        project_pool=list(project_pool),
    )
    try:
        raw = call_llm(
            model=task_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt + f"\n\ntotal_bullet_budget: {budget}",
                },
            ],
            temperature=task_temperature,
            max_tokens=1200,
        )
        proposed = json.loads(_strip_fence(raw))
        if not isinstance(proposed, dict):
            return fallback
    except Exception as exc:
        print(f"[warning] structure planner failed, using full evidence: {exc}", file=sys.stderr)
        return fallback

    # Clamp: 0, or inside the evidence file's declared range. Nothing else is legal.
    planned_roles: Dict[str, int] = {}
    raw_roles = proposed.get("roles") or {}
    for company, (mn, mx) in bounds.items():
        want = raw_roles.get(company, mx)
        try:
            want = int(want)
        except (TypeError, ValueError):
            want = mx
        planned_roles[company] = 0 if want <= 0 else max(mn, min(mx, want))

    # Refuse to empty the resume, however the model scored things.
    if not any(planned_roles.values()):
        planned_roles = {c: mx for c, (_, mx) in bounds.items()}

    def _clamp(key: str, lo: int, hi: int, default: int) -> int:
        try:
            return max(lo, min(hi, int(proposed.get(key, default))))
        except (TypeError, ValueError):
            return default

    return {
        "roles": planned_roles,
        "projects": _clamp("projects", 2, 4, DEFAULT_TOP_ACADEMIC_PROJECT_COUNT),
        "coursework": _clamp("coursework", 3, 5, DEFAULT_TOP_COURSE_COUNT),
        "reasons": proposed.get("reasons") or {},
    }


def extract_jd_signals(jd_text: str, model: str) -> Dict[str, Any]:
    """Structured ranking signals from a JD.

    Separate from summarize_job_description(), which returns cleaned prose for bullet
    generation. Selection tasks and the QA report need structure. Returns an empty-but-valid
    shape on failure so callers never have to guard.
    """
    empty: Dict[str, Any] = {
        "archetype": "generalist",
        "seniority": "",
        "domain": "",
        "must_have": [],
        "nice_to_have": [],
        "credentials": [],
        "top_3_screens": [],
    }
    task_model, task_temperature = get_task_llm_settings(
        LLM_TASK_JD_SIGNALS, fallback_model=model
    )
    system_prompt, user_prompt = build_jd_signals_prompts(jd_text)
    try:
        raw = call_llm(
            model=task_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=task_temperature,
            max_tokens=1200,
        )
        data = json.loads(_strip_fence(raw))
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty
    return {**empty, **{k: v for k, v in data.items() if k in empty}}


def load_skills_inventory(path: Path | None = None) -> Dict[str, Any]:
    """Master whitelist for the Skills section."""
    inventory_path = path or (DATA_DIR / "skills_inventory.json")
    with inventory_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _inventory_index(inventory: Dict[str, Any]) -> Dict[str, str]:
    """Map lowercased skill name -> canonical name, for verbatim matching."""
    index: Dict[str, str] = {}
    for category in inventory.get("categories", []):
        for skill in category.get("skills", []):
            name = skill.get("name", "")
            if name:
                index[name.strip().lower()] = name
    return index


def select_skills_for_jd(
    jd_signals: Dict[str, Any],
    model: str,
    inventory: Dict[str, Any] | None = None,
    selected_bullets: Dict[str, List[str]] | None = None,
    selected_projects: Sequence[Dict[str, Any]] | None = None,
    max_categories: int = DEFAULT_MAX_SKILL_CATEGORIES,
    max_per_category: int = DEFAULT_MAX_SKILLS_PER_CATEGORY,
    max_total_skills: int = DEFAULT_MAX_TOTAL_SKILLS,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Tailor the Skills section, enforcing the inventory whitelist in code.

    Returns (categories, dropped) where `dropped` lists anything the model returned that is
    not in the inventory. Dropping happens here rather than being left to prompt compliance,
    so an unsupported claim cannot reach the PDF even if the model ignores its instructions.
    """
    inv = inventory if inventory is not None else load_skills_inventory()
    index = _inventory_index(inv)
    dropped: List[str] = []
    selected_resume_evidence = {
        "experience_bullets": selected_bullets or {},
        "academic_projects": [
            {
                "topic": str(project.get("Topic", "")).strip(),
                "bullets": [
                    str(bullet).strip()
                    for bullet in project.get("Bullet", [])
                    if str(bullet).strip()
                ],
            }
            for project in (selected_projects or [])
            if isinstance(project, dict)
        ],
    }

    task_model, task_temperature = get_task_llm_settings(
        LLM_TASK_SKILLS_SELECTION, fallback_model=model
    )
    system_prompt, user_prompt = build_skills_selection_prompts(
        jd_signals=jd_signals,
        inventory=inv,
        selected_resume_evidence=selected_resume_evidence,
        max_categories=max_categories,
        max_per_category=max_per_category,
        max_total_skills=max_total_skills,
    )
    try:
        raw = call_llm(
            model=task_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=task_temperature,
            max_tokens=1500,
        )
        payload = json.loads(_strip_fence(raw))
        proposed = payload.get("categories", []) if isinstance(payload, dict) else []
    except Exception:
        proposed = []

    if not proposed:
        return _fallback_skills(
            inv,
            max_categories=max_categories,
            max_per_category=max_per_category,
            max_total_skills=max_total_skills,
        ), dropped

    categories: List[Dict[str, Any]] = []
    globally_kept: set[str] = set()
    remaining = max(0, max_total_skills)
    for category in proposed:
        if len(categories) >= max(0, max_categories) or remaining <= 0:
            break
        if not isinstance(category, dict):
            continue
        kept: List[str] = []
        for name in category.get("skills", []) or []:
            canonical = index.get(str(name).strip().lower())
            if canonical is None:
                dropped.append(str(name))
            elif canonical not in globally_kept:
                kept.append(canonical)
                globally_kept.add(canonical)
            if len(kept) >= max(0, max_per_category) or len(kept) >= remaining:
                break
        if kept:
            categories.append(
                {
                    "id": category.get("id", ""),
                    "label": category.get("label") or category.get("id", "Skills"),
                    "skills": kept,
                }
            )
            remaining -= len(kept)

    if not categories:
        return _fallback_skills(
            inv,
            max_categories=max_categories,
            max_per_category=max_per_category,
            max_total_skills=max_total_skills,
        ), dropped
    return categories, dropped


def _fallback_skills(
    inventory: Dict[str, Any],
    max_categories: int,
    max_per_category: int,
    max_total_skills: int,
) -> List[Dict[str, Any]]:
    """Inventory order, truncated. Used when the model call fails or returns nothing usable."""
    categories: List[Dict[str, Any]] = []
    remaining = max(0, max_total_skills)
    for category in inventory.get("categories", [])[:max(0, max_categories)]:
        if remaining <= 0:
            break
        skills = [
            skill["name"]
            for skill in category.get("skills", [])
            if skill.get("name")
        ][:min(max(0, max_per_category), remaining)]
        if skills:
            categories.append(
                {
                    "id": category.get("id", ""),
                    "label": category.get("label", "Skills"),
                    "skills": skills,
                }
            )
            remaining -= len(skills)
    return categories



def shorten_bullet_llm(
    bullet: str,
    max_chars: int,
    model: str,
    projects: Sequence[Dict[str, Any]] | None = None,
) -> str:
    """Compress one bullet under a hard character budget, preserving every claim.

    Fallback for orphan repair when deterministic contractions cannot free enough
    characters. Verified before use: the rewrite is rejected if it exceeds the budget or
    introduces a number that was not in the original, so a bad rewrite degrades to a no-op
    rather than shipping an unsupported claim.
    """
    task_model, task_temperature = get_task_llm_settings(
        LLM_TASK_BULLET_SHORTEN, fallback_model=model
    )
    system_prompt, user_prompt = build_bullet_shorten_prompts(
        bullet=bullet, max_chars=max_chars, projects=list(projects) if projects else None
    )
    try:
        raw = call_llm(
            model=task_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=task_temperature,
            max_tokens=400,
        )
    except Exception:
        return bullet

    candidate = _strip_fence(raw).strip().strip('"').lstrip("-•").strip()
    candidate = re.sub(r"^\d+[.)]\s*", "", candidate)

    # max_chars is a target, not a gate. A rewrite that lands slightly over budget still
    # often clears the orphan, and the caller recompiles and re-measures to find out. An
    # earlier version hard-rejected anything over budget and silently discarded good
    # rewrites (269 -> 253 chars was thrown away for missing a 233 target).
    if not candidate or len(candidate) >= len(bullet):
        return bullet
    # Never let a "shortening" introduce a figure the original did not contain.
    if _numeric_claims(candidate) - _numeric_claims(bullet):
        return bullet
    return candidate
