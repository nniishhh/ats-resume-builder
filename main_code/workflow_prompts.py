import json
from typing import Any, Dict, List, Sequence, Tuple


def build_jd_signal_prompts(jd_text: str) -> Tuple[str, str]:
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
    return system_prompt, user_prompt


def build_course_selection_prompts(
    jd_text: str,
    course_pool: Sequence[str],
    top_k: int,
) -> Tuple[str, Dict[str, Any]]:
    system_prompt = (
        "You are a precise course-matching assistant. "
        "Choose the most job-relevant courses from the provided list only. "
        "Return ONLY valid JSON with no extra text."
    )
    user_prompt: Dict[str, Any] = {
        "task": "Select the top job-relevant courses for this job description.",
        "selection_rules": [
            f"Select exactly {top_k} course names.",
            "Use only names from the provided course list.",
            "Prioritize direct technical and domain fit to the JD requirements.",
            "Do not invent or rename courses.",
        ],
        "output_schema": {"selected_courses": [f"exactly {top_k} course names"]},
        "courses": list(course_pool),
        "job_description": jd_text,
    }
    return system_prompt, user_prompt


def build_bullet_generation_system_prompt(
    min_bullet_chars: int,
    max_bullet_chars: int,
) -> str:
    return f"""You write concise, ATS-optimized resume bullets grounded strictly in the provided project evidence and job description.

========================================
OUTPUT CONTRACT (NON-NEGOTIABLE)
========================================
- Return ONLY a numbered list.
- No markdown, commentary, explanations, or code fences.
- Format: "1. First bullet\\n2. Second bullet"

========================================
HARD LENGTH RULE (NON-NEGOTIABLE)
========================================
- Each bullet MUST be between {min_bullet_chars} and {max_bullet_chars} characters (including spaces).
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


def build_bullet_generation_user_prompt(
    company: str,
    min_bullets: int,
    max_bullets: int,
    jd_signals: Dict[str, Any],
    jd_text: str,
    projects: List[Dict[str, Any]],
    used_verbs: List[str] | None,
) -> Dict[str, Any]:
    return {
        "task": f"Generate {min_bullets}-{max_bullets} resume bullets for {company}.",
        "company": company,
        "min_bullets": min_bullets,
        "max_bullets": max_bullets,
        "jd_analysis": jd_signals,
        "job_description": jd_text,
        "project_evidence": projects,
        "already_used_verbs": used_verbs or [],
    }


def build_bullet_repair_payload(
    company: str,
    min_bullets: int,
    max_bullets: int,
    min_bullet_chars: int,
    max_bullet_chars: int,
    issues: List[str],
    latest_output: str,
    used_verbs: List[str] | None,
    projects: List[Dict[str, Any]],
    attempt: int,
) -> Dict[str, Any]:
    return {
        "instruction": (
            "Regenerate the entire output to satisfy all constraints exactly. "
            "Do not explain. Return only a numbered list."
        ),
        "attempt": attempt,
        "issues": issues,
        "previous_output": latest_output,
        "constraints": {
            "company": company,
            "min_bullets": min_bullets,
            "max_bullets": max_bullets,
            "HARD_LIMIT_min_characters_per_bullet": min_bullet_chars,
            "HARD_LIMIT_max_characters_per_bullet": max_bullet_chars,
        },
        "already_used_verbs": used_verbs or [],
        "project_evidence": projects,
        "output_rule": "Numbered list only. No explanations.",
    }


def build_all_bullets_system_prompt(
    min_bullet_chars: int,
    max_bullet_chars: int,
) -> str:
    return f"""You write concise, ATS-optimized resume bullets grounded strictly in the provided project evidence and job description.

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
- Each bullet MUST be between {min_bullet_chars} and {max_bullet_chars} characters (including spaces).
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


def build_all_bullets_user_prompt(
    jd_signals: Dict[str, Any],
    jd_text: str,
    companies_spec: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "task": "Generate resume bullets for ALL companies below in one JSON response.",
        "jd_analysis": jd_signals,
        "job_description": jd_text,
        "companies": companies_spec,
    }


def build_academic_project_selection_prompts(
    jd_text: str,
    project_list: List[Dict[str, Any]],
    top_k: int,
) -> Tuple[str, str]:
    system_prompt = (
        "You are a hiring manager.\n\n"
        f"From the provided project list, select the {top_k} MOST relevant projects based on the Job Description.\n\n"
        "Selection rules:\n"
        "- Prioritize direct skill and responsibility overlap.\n"
        "- Favor required qualifications over preferred.\n"
        "- Do NOT invent information.\n"
        f"- Return ONLY an array of {top_k} project Topic names.\n"
        "- No explanations. No extra text."
    )
    user_prompt = (
        "Job Description:\n"
        f"{jd_text}\n\n"
        "Project List:\n"
        f"{json.dumps(project_list, ensure_ascii=True)}\n\n"
        f"Return only the {top_k} most relevant Topic names."
    )
    return system_prompt, user_prompt
