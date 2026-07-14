import json
from typing import Any, Dict, List, Sequence, Tuple


def build_jd_summary_prompts(jd_text: str) -> Tuple[str, str]:
    system_prompt = (
        "You are a Job Description Analyst. Your task is to distill a Job Description down to its core requirements for resume tailoring.\n\n"
        "OBJECTIVE: Remove noise, but preserve ALL potential ranking signals.\n\n"
        "=========================================\n"
        "WHAT TO REMOVE (The \"Fluff\")\n"
        "=========================================\n"
        "- Company history, 'About Us', and generic marketing intros.\n"
        "- Employee benefits (health, 401k, gym, perks).\n"
        "- Legal disclaimers (EEO statements, background check warnings).\n"
        "- Generic buzzwords IF they stand alone (e.g., 'Passionate', 'Go-getter').\n\n"
        "=========================================\n"
        "WHAT TO KEEP (The \"Signal\")\n"
        "=========================================\n"
        "1. HARD SKILLS: All tools, languages, software, and methodologies.\n"
        "2. RESPONSIBILITIES: What the person will actually do day-to-day.\n"
        "3. DOMAIN CONTEXT: Specific industries or environments (e.g., 'High-frequency trading', 'Start-up environment', 'B2B SaaS').\n"
        "4. CONSTRAINTS & SCOPE: \n"
        "   - Hierarchy information (e.g., 'Reports to VP').\n"
        "   - Travel requirements or work shifts.\n"
        "   - Team size or budget responsibility.\n\n"
        "=========================================\n"
        "THE \"SAFE KEEP\" RULE\n"
        "=========================================\n"
        "If you are unsure whether a sentence is 'fluff' or a 'requirement', KEEP IT.\n"
        "It is better to include extra text than to delete a potential ATS keyword.\n\n"
        "Output only the cleaned text. Do not summarize; copy exact phrases."
    )
    user_prompt = (
        "Clean this Job Description, keeping all necessary requirements and context:\n\n"
        f"{jd_text}"
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
- Every bullet MUST satisfy this exact condition:
      {min_bullet_chars} <= len(bullet) <= {max_bullet_chars}
- len(bullet) is Python's character count of the bullet string, including spaces.
- If the condition is False for any bullet, that bullet is invalid and must be rewritten.

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
EVIDENCE FIELD GUIDE
========================================
- "example_bullets" are human-proofread, trusted references: treat their claims and phrasing as authoritative.
- "harvested_bullets" are additional strong variants: use them for phrasing inspiration, same trust as other evidence.
- Some "results" entries include "(framing variants: ...)": these are alternative truthful phrasings of the SAME metric. Pick exactly ONE framing per metric based on JD fit; never combine variants or state the metric twice.
- "keywords" are ATS vocabulary hints, not facts; weave them in only where the evidence supports them.

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
- Avoid repetition.

========================================
LEXICAL DIVERSITY RULE
========================================
- Avoid repeating the same connector or phrasing across bullets.
- Ensure lexical variety in verbs, connectors, and clause structure.
- Do NOT repeat the same tool in every bullet when context is already clear.
- After an initial explicit mention, tools may be abstracted into system-level phrasing.
- Reintroduce a tool only when it adds meaningful technical clarity or differentiation."""


def build_bullet_generation_user_prompt(
    company: str,
    min_bullets: int,
    max_bullets: int,
    jd_text: str,
    projects: List[Dict[str, Any]],
    used_verbs: List[str] | None,
) -> Dict[str, Any]:
    return {
        "task": f"Generate {min_bullets}-{max_bullets} resume bullets for {company}.",
        "company": company,
        "min_bullets": min_bullets,
        "max_bullets": max_bullets,
        "job_description_summary": jd_text,
        "project_evidence": projects,
        "already_used_verbs": used_verbs or [],
        "lexical_diversity_rules": [
            "Avoid repeating the same connector or phrasing across bullets.",
            "Ensure lexical variety in verbs, connectors, and clause structure.",
            "Do NOT repeat the same tool in every bullet when context is already clear.",
            "After an initial explicit mention, tools may be abstracted into system-level phrasing.",
            "Reintroduce a tool only when it adds meaningful technical clarity or differentiation.",
        ],
    }


def build_bullet_repair_payload(
    company: str,
    min_bullets: int,
    max_bullets: int,
    min_bullet_chars: int,
    max_bullet_chars: int,
    issues: List[str],
    latest_output: str,
    jd_text: str,
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
            "length_condition": (
                f"{min_bullet_chars} <= len(bullet) <= {max_bullet_chars}"
            ),
        },
        "job_description_summary": jd_text,
        "already_used_verbs": used_verbs or [],
        "project_evidence": projects,
        "lexical_diversity_rules": [
            "Avoid repeating the same connector or phrasing across bullets.",
            "Ensure lexical variety in verbs, connectors, and clause structure.",
            "Do NOT repeat the same tool in every bullet when context is already clear.",
            "After an initial explicit mention, tools may be abstracted into system-level phrasing.",
            "Reintroduce a tool only when it adds meaningful technical clarity or differentiation.",
        ],
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
- Every bullet MUST satisfy this exact condition:
      {min_bullet_chars} <= len(bullet) <= {max_bullet_chars}
- len(bullet) is Python's character count of the bullet string, including spaces.
- If the condition is False for any bullet, that bullet is invalid and must be rewritten.

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
EVIDENCE FIELD GUIDE
========================================
- "example_bullets" are human-proofread, trusted references: treat their claims and phrasing as authoritative.
- "harvested_bullets" are additional strong variants: use them for phrasing inspiration, same trust as other evidence.
- Some "results" entries include "(framing variants: ...)": these are alternative truthful phrasings of the SAME metric. Pick exactly ONE framing per metric based on JD fit; never combine variants or state the metric twice.
- "keywords" are ATS vocabulary hints, not facts; weave them in only where the evidence supports them.

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
- Avoid repetition within each company.

========================================
LEXICAL DIVERSITY RULE
========================================
- Avoid repeating the same connector or phrasing across bullets.
- Ensure lexical variety in verbs, connectors, and clause structure.
- Do NOT repeat the same tool in every bullet when context is already clear.
- After an initial explicit mention, tools may be abstracted into system-level phrasing.
- Reintroduce a tool only when it adds meaningful technical clarity or differentiation."""


def build_all_bullets_user_prompt(
    jd_text: str,
    companies_spec: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "task": "Generate resume bullets for ALL companies below in one JSON response.",
        "job_description_summary": jd_text,
        "companies": companies_spec,
        "lexical_diversity_rules": [
            "Avoid repeating the same connector or phrasing across bullets.",
            "Ensure lexical variety in verbs, connectors, and clause structure.",
            "Do NOT repeat the same tool in every bullet when context is already clear.",
            "After an initial explicit mention, tools may be abstracted into system-level phrasing.",
            "Reintroduce a tool only when it adds meaningful technical clarity or differentiation.",
        ],
    }


def build_all_bullets_repair_payload(
    jd_text: str,
    companies_spec: List[Dict[str, Any]],
    issues_by_company: Dict[str, List[str]],
    previous_output: str,
    min_bullet_chars: int,
    max_bullet_chars: int,
) -> Dict[str, Any]:
    return {
        "instruction": (
            "The previous JSON output violated constraints for some companies. "
            "Regenerate the ENTIRE JSON object for ALL companies so every bullet "
            "satisfies all constraints. Do not explain. Return only the JSON object."
        ),
        "length_condition": (
            f"{min_bullet_chars} <= len(bullet) <= {max_bullet_chars}"
        ),
        "issues_by_company": issues_by_company,
        "previous_output": previous_output,
        "job_description_summary": jd_text,
        "companies": companies_spec,
        "output_rule": (
            "Return only a single JSON object: keys = company names, "
            "values = arrays of bullet strings. No markdown, no code fences."
        ),
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
