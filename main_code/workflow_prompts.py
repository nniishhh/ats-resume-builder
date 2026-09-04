import json
from typing import Any, Dict, List, Sequence, Tuple

# Shared by every bullet-generation payload. Kept as one constant because the
# sequential and single_prompt paths, plus both of their repair rounds, all send
# the same rules — four hand-synced copies is how they drift apart.
LEXICAL_DIVERSITY_RULES = [
    "Do NOT repeat the same headline figure in consecutive bullets. If two bullets for "
    "one company both draw on the same scale metric (e.g. '10K daily deliveries'), state "
    "it in the stronger bullet and let the other stand on its own result.",
    "Avoid repeating the same connector or phrasing across bullets.",
    "Ensure lexical variety in verbs, connectors, and clause structure.",
    "Do NOT repeat the same tool in every bullet when context is already clear.",
    "After an initial explicit mention, tools may be abstracted into system-level phrasing.",
    "Reintroduce a tool only when it adds meaningful technical clarity or differentiation.",
]


def _bullet_system_prompt(
    min_bullet_chars: int,
    max_bullet_chars: int,
    output_contract: str,
    ordering_line: str,
    repetition_scope: str,
) -> str:
    """Shared system prompt for both generation modes.

    The two modes are identical apart from their output contract (numbered list vs
    JSON object) and whether ordering/repetition is scoped per company. Everything
    else — truthfulness, no-inflation, evidence handling, jargon calibration — must
    stay the same in both, so it lives here once instead of in two copies that have
    to be edited in lockstep.
    """
    return f"""You write concise, ATS-optimized resume bullets grounded strictly in the provided project evidence and job description.

========================================
OUTPUT CONTRACT (NON-NEGOTIABLE)
========================================
{output_contract}

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
NO INFLATION
========================================
Describe the work at the level the evidence describes it.
- Do NOT add a category of work that is not in the evidence (e.g. calling optimization work "predictive").
- Do NOT promote a technique into a broader family to match the JD (e.g. clustering -> "machine learning" if the evidence says clustering).
- Generic system nouns ("pipeline", "system", "framework") are fine when the evidence describes a built artifact.

========================================
EVIDENCE FIELD GUIDE
========================================
- "main_metric" is the default measurable result for that project. Prioritize it when the project is selected.
- "sub_metrics" are optional supporting figures. Add at most ONE sub-metric only when it materially strengthens alignment with the JD; never place multiple sub-metrics in the same bullet.
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
- {ordering_line}
- If relevance is similar, order by measurable impact and scale.

========================================
STYLE PRINCIPLES
========================================
- Prefer concrete nouns and outcomes over vague corporate phrasing.
- Use strong, varied action verbs.
- When a bullet includes a numeric or measurable result, you should place that result at the end of the sentence.
- Avoid repetition{repetition_scope}.

========================================
READER-CALIBRATED JARGON
========================================
Match technical density to "target_seniority" in the user payload. This never changes what
happened, what tools were used, or what the result was — only how technically dense the
phrasing is.
- "intern" / "junior": Favor plain, concrete language a generalist hiring manager can follow
  at a glance. Name tools directly (SQL, Python, Tableau) but avoid compressed technical
  shorthand (e.g. "agglomerative clustering", "MILP", "record-linkage workflow") unless the
  JD itself uses that exact term — describe the mechanism in plain words instead.
- "mid" or unset: Balance plain language with domain terminology where it is precise.
- "senior": Full technical density is expected — precise method, algorithm, and systems
  terminology is rewarded, not a liability.

========================================
LEXICAL DIVERSITY RULE
========================================
""" + "\n".join(f"- {rule}" for rule in LEXICAL_DIVERSITY_RULES)


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
    return _bullet_system_prompt(
        min_bullet_chars=min_bullet_chars,
        max_bullet_chars=max_bullet_chars,
        output_contract=(
            "- Return ONLY a numbered list.\n"
            "- No markdown, commentary, explanations, or code fences.\n"
            '- Format: "1. First bullet\\n2. Second bullet"'
        ),
        ordering_line="Order bullets by strongest alignment to the JD.",
        repetition_scope="",
    )


def build_bullet_generation_user_prompt(
    company: str,
    min_bullets: int,
    max_bullets: int,
    jd_text: str,
    projects: List[Dict[str, Any]],
    used_verbs: List[str] | None,
    seniority: str = "",
) -> Dict[str, Any]:
    return {
        "task": f"Generate {min_bullets}-{max_bullets} resume bullets for {company}.",
        "company": company,
        "min_bullets": min_bullets,
        "max_bullets": max_bullets,
        "job_description_summary": jd_text,
        "target_seniority": seniority or "mid",
        "project_evidence": projects,
        "already_used_verbs": used_verbs or [],
        "lexical_diversity_rules": LEXICAL_DIVERSITY_RULES,
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
    seniority: str = "",
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
        "target_seniority": seniority or "mid",
        "already_used_verbs": used_verbs or [],
        "project_evidence": projects,
        "lexical_diversity_rules": LEXICAL_DIVERSITY_RULES,
        "output_rule": "Numbered list only. No explanations.",
    }


def build_all_bullets_system_prompt(
    min_bullet_chars: int,
    max_bullet_chars: int,
) -> str:
    return _bullet_system_prompt(
        min_bullet_chars=min_bullet_chars,
        max_bullet_chars=max_bullet_chars,
        output_contract=(
            "- Return ONLY valid JSON.\n"
            "- No markdown, commentary, explanations, or code fences.\n"
            "- JSON must be a single object:\n"
            "  - Keys = company names (strings)\n"
            "  - Values = arrays of bullet strings\n"
            "\n"
            "Example:\n"
            '{"Company A": ["bullet 1", "bullet 2"]}\n'
        ),
        ordering_line="Within each company, order bullets by strongest alignment to the JD.",
        repetition_scope=" within each company",
    )


def build_all_bullets_user_prompt(
    jd_text: str,
    companies_spec: List[Dict[str, Any]],
    seniority: str = "",
) -> Dict[str, Any]:
    return {
        "task": "Generate resume bullets for ALL companies below in one JSON response.",
        "job_description_summary": jd_text,
        "target_seniority": seniority or "mid",
        "companies": companies_spec,
        "lexical_diversity_rules": LEXICAL_DIVERSITY_RULES,
    }


def build_all_bullets_repair_payload(
    jd_text: str,
    companies_spec: List[Dict[str, Any]],
    issues_by_company: Dict[str, List[str]],
    previous_output: str,
    min_bullet_chars: int,
    max_bullet_chars: int,
    seniority: str = "",
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
        "target_seniority": seniority or "mid",
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


def build_jd_signals_prompts(jd_text: str) -> Tuple[str, str]:
    """Structured ranking signals from a JD.

    Complements build_jd_summary_prompts (which returns cleaned prose for bullet generation).
    Downstream consumers need structure, not prose: the skills selector needs must-haves, the
    QA report needs keywords to check coverage against, and the trim ladder needs a relevance
    ranking so it drops the least relevant content rather than an arbitrary one.
    """
    system_prompt = (
        "You are a Job Description Analyst. Extract structured ranking signals for resume tailoring.\n\n"
        "Return ONLY valid JSON matching this schema, with no markdown or commentary:\n"
        "{\n"
        '  "archetype": one of "builder" | "analyst" | "researcher" | "ops" | "generalist",\n'
        '  "seniority": one of "intern" | "junior" | "mid" | "senior",\n'
        '  "domain": short phrase, e.g. "transit operations" or "adtech",\n'
        '  "must_have": [concrete skills/tools/methods stated as required],\n'
        '  "nice_to_have": [stated as preferred or bonus],\n'
        '  "credentials": [degree, major, certification and eligibility requirements],\n'
        '  "top_3_screens": [the three things this JD is really filtering on, in plain language]\n'
        "}\n\n"
        "RULES\n"
        "- credentials holds ONLY things a bullet cannot change: required degrees, fields of\n"
        "  study, certifications, work authorization, years of experience. Put them there and\n"
        "  NOT in must_have — must_have is scored against the finished resume, and a degree\n"
        "  requirement is either already met by the Education section or cannot be met at all.\n"
        "  Examples: \"Master's degree\", \"PhD\", \"Operations Research\", \"2+ years experience\".\n"
        "- must_have and nice_to_have hold SHORT KEYWORD PHRASES, not sentences.\n"
        "  Good: \"Python\", \"SQL\", \"predictive modeling\", \"model deployment\", \"Salesforce\".\n"
        "  Bad:  \"At least 2 years of Data Scientist experience.\", \"Strong Python and SQL skills.\"\n"
        "  Split compound requirements into separate entries: \"Strong Python and SQL skills\"\n"
        "  becomes [\"Python\", \"SQL\"].\n"
        "- Each entry must be a term that could plausibly appear verbatim in a resume, because\n"
        "  these are matched literally against the finished document to measure coverage.\n"
        "- Use the JD's own wording for tools and technologies; drop filler words like\n"
        "  \"strong\", \"expertise in\", \"experience with\", \"at least N years of\".\n"
        "- must_have holds only what the JD marks as required; everything softer goes in nice_to_have.\n"
        "- Do not invent requirements that are not in the text.\n"
        "- Keep each list under 12 entries; prefer the most discriminating ones."
    )
    user_prompt = f"Extract ranking signals from this job description:\n\n{jd_text}"
    return system_prompt, user_prompt


def build_structure_plan_prompts(
    jd_signals: Dict[str, Any],
    roles: List[Dict[str, Any]],
    project_pool: List[str],
) -> Tuple[str, str]:
    """Decide how much page space each role and section earns for this JD.

    Everything else in the pipeline tailors wording; the resume's *shape* was fixed
    (same bullet counts, same section order, every posting). This picks the shape.
    The model only proposes — allowed counts are clamped in code to what each
    evidence file can actually support, so it can never invent depth.
    """
    system_prompt = (
        "You allocate limited resume space across a candidate's roles and sections for one "
        "specific job posting. You are not writing anything — only deciding how much room "
        "each item earns.\n\n"
        "Return ONLY valid JSON, no markdown or commentary:\n"
        "{\n"
        '  "roles": {"<role_key>": <int bullets>, ...},\n'
        '  "projects": <int>,\n'
        '  "coursework": <int>,\n'
        '  "reasons": {"<role_key>": "<short reason>", ...}\n'
        "}\n\n"
        "RULES\n"
        "- One page is the budget. Giving one role more means another gets less.\n"
        "- For each role you may return 0 to drop it entirely, or any count within the\n"
        "  allowed range given for that role. Nothing in between is valid.\n"
        "- Judge each role by the evidence given for it — what_this_role_covers, keywords\n"
        "  and tools — NOT by the company's name or how recent it is. A less famous or\n"
        "  older role whose actual work matches the posting outranks a better-known one\n"
        "  whose work does not.\n"
        "- Drop a role only when its evidence genuinely does not serve this posting.\n"
        "- Weight by what the posting screens on, not by recency or brand.\n"
        "- Section order is fixed: Experience always comes before Academic Projects.\n"
        "  You are allocating space only, never reordering the page.\n"
        "- Prefer depth on the one or two roles that match, over thin coverage of everything.\n"
        "- reasons: one short clause per role explaining the count. This is shown to the "
        "candidate, so be concrete about the posting, not generic.\n"
        "- total_bullet_budget is BINDING and is deliberately less than the sum of every\n"
        "  role's maximum. You cannot give every role its maximum. Decide what this posting\n"
        "  actually screens on, spend there, and take the space from somewhere else — either\n"
        "  run the weaker roles at their minimum, or drop one to 0 so a strong role stays deep.\n"
        "  Returning the same allocation you would give any other posting means you have not\n"
        "  done the task."
    )
    user_prompt = json.dumps(
        {
            "job_signals": jd_signals,
            "roles": roles,
            "academic_project_pool": project_pool,
            "coursework_slots": {"min": 3, "max": 5},
            "project_slots": {"min": 2, "max": 4},
        },
        ensure_ascii=True,
    )
    return system_prompt, user_prompt


def build_skills_selection_prompts(
    jd_signals: Dict[str, Any],
    inventory: Dict[str, Any],
    selected_resume_evidence: Dict[str, Any] | None = None,
    max_categories: int = 4,
    max_per_category: int = 6,
    max_total_skills: int = 24,
) -> Tuple[str, str]:
    """Choose and order the Skills section from a fixed whitelist.

    The inventory is exhaustive by construction. The model selects and orders; it never
    authors. Anything it returns that is not in the inventory is dropped in code, so a
    hallucinated skill cannot reach the PDF even if the model ignores these instructions.
    """
    system_prompt = (
        "You are tailoring the Skills section of a resume to a job description.\n\n"
        "PURPOSE\n"
        "- Build a compact recruiter-and-ATS index of the candidate's strongest, most relevant skills.\n"
        "- The Experience and Academic Projects sections prove capability; Skills should make the\n"
        "  most important evidence easy to find, not reproduce the candidate's full inventory.\n\n"
        "CRITICAL CONSTRAINT\n"
        "- You may ONLY return skill names that appear VERBATIM in the provided inventory.\n"
        "- The inventory is exhaustive. If a skill the JD asks for is not in it, the candidate\n"
        "  does not have it. Omit it. Do NOT substitute a similar-sounding skill.\n"
        "- Never invent, rename, reword, or abbreviate a skill name.\n\n"
        "SELECTION\n"
        f"- Return at most {max_categories} categories.\n"
        f"- Return at most {max_per_category} skills per category.\n"
        f"- Return at most {max_total_skills} skills total across all categories.\n"
        "- Aim for 16-20 total skills when that many are directly relevant and supported. Use\n"
        "  fewer when the evidence is narrower; never pad.\n"
        "- Prioritize exact must_have matches from the JD signals.\n"
        "- Include a secondary or nice_to_have skill only when its inventory evidence is also\n"
        "  supported by a selected experience bullet or selected academic project.\n"
        "- Prefer concrete tools, languages, platforms, and specific methods over broad phrases\n"
        "  such as generic analysis, communication, or problem-solving labels.\n"
        "- Drop categories that are not relevant to this job; do not pad them.\n"
        "- Order categories most-relevant first, and skills within a category most-relevant first.\n"
        "- A skill being present in the inventory makes it eligible, not automatically worth listing.\n\n"
        "Return ONLY valid JSON:\n"
        '{"categories": [{"id": "<category id from inventory>", "label": "<label to print>",\n'
        '                 "skills": ["<verbatim inventory names>"]}]}'
    )
    user_prompt = json.dumps(
        {
            "job_signals": jd_signals,
            "inventory": inventory,
            "selected_resume_evidence": selected_resume_evidence or {},
            "reminder": "Every returned skill must match an inventory name character for character.",
        },
        ensure_ascii=True,
    )
    return system_prompt, user_prompt




def build_bullet_shorten_prompts(
    bullet: str,
    max_chars: int,
    projects: List[Dict[str, Any]] | None = None,
) -> Tuple[str, str]:
    """Compress one bullet that renders with an orphan line.

    Used only as a fallback after deterministic contractions fail, which happens when a
    bullet is already tight and simply runs a few characters past a line boundary. The
    rewrite must be lossless in substance: this runs after validation, so a dropped metric
    or invented claim here would bypass every earlier check.
    """
    system_prompt = (
        "Shorten a single resume bullet so it fits on fewer rendered lines.\n\n"
        "ABSOLUTE RULES\n"
        f"- The result MUST be at most {max_chars} characters.\n"
        "- Keep EVERY number, percentage, tool name, and proper noun exactly as written.\n"
        "- Do NOT add any fact, tool, metric, or claim that is not already in the bullet.\n"
        "- Do NOT change what was done, only how tersely it is described.\n"
        "- Keep the same opening verb.\n\n"
        "HOW TO SHORTEN\n"
        "- Cut filler and hedging first.\n"
        "- Collapse redundant clauses; prefer a noun phrase over a relative clause.\n"
        "- Drop the least informative trailing detail last, and only if still over budget.\n\n"
        "Return ONLY the rewritten bullet. No quotes, numbering, or commentary."
    )
    payload: Dict[str, Any] = {"bullet": bullet, "max_chars": max_chars}
    if projects:
        payload["evidence_for_reference_only"] = projects
    user_prompt = json.dumps(payload, ensure_ascii=True)
    return system_prompt, user_prompt
