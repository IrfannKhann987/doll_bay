# ai_nodes.py
import json
import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from langchain_openai import OpenAI

from prompts import (
    SAFETY_PROMPT,
    QUIZ_SUMMARY_PROMPT,
    PLAN_21D_PROMPT,
    COACH_PROMPT,
    QUIZ_GENERATOR_PROMPT,
    CANONICALIZE_PROMPT,
    WHY_DAY_PROMPT
)
from schemas import HabitState, SafetyResult, QuizSummary, Plan21D,QuizForm

load_dotenv()


def _get_model(name: str, default: str) -> str:
    """
    Read a model name from env, but NEVER return None or empty string.
    """
    value = os.getenv(name)
    if not value or value.strip().lower() == "none":
        return default
    return value.strip()

MODEL_JSON = os.getenv("OPENAI_MODEL_JSON", "gpt-4.1")
MODEL_TEXT = os.getenv("OPENAI_MODEL_TEXT", "gpt-4.1")
def _llm_json(
    prompt: str,
    max_tokens: int = 800,
    temperature: float = 0.5,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Call the JSON-optimized LLM and return a Python dict.
    Retries with slightly higher temperature and stronger JSON instructions if parsing fails.
    """
    for attempt in range(retries):
        llm = ChatOpenAI(
            model=MODEL_JSON,
            temperature=temperature + (attempt * 0.2),
            response_format={"type": "json_object"},
        )

        resp = llm.invoke(prompt).content

        try:
            return json.loads(resp)
        except Exception:
            # strengthen instructions & increase randomness
            prompt += (
                "\nReturn STRICT JSON. No commentary. "
                "Do NOT repeat previous suggestions."
            )
            continue

    # final fallback if everything fails
    return {}


def _json_llm(temperature: float = 0.3) -> ChatOpenAI:
    """
    Base JSON-optimized LLM (used with structured outputs).
    """
    return ChatOpenAI(
        model=MODEL_JSON,
        temperature=temperature,
    )


def _text_llm(temperature: float = 0.6) -> ChatOpenAI:
    """
    Base text LLM for the coach.
    """
    return ChatOpenAI(
        model=MODEL_TEXT,
        temperature=temperature,
    )



def canonicalize_habit_node(state: HabitState):
    user_raw = state.habit_description or ""

    prompt = CANONICALIZE_PROMPT.format(user_habit_raw=user_raw)
    data = _llm_json(prompt)

    # Fallback if model fails
    canonical = data.get("canonical_habit_name", user_raw)
    category = data.get("habit_category", "unknown")
    conf = data.get("confidence", "low")

    return {
        "canonical_habit_name": canonical,
        "habit_category": category,
        "canonical_confidence": conf,
    }


# ---------- Safety Node ----------

def safety_node(state: HabitState) -> Dict[str, Any]:
    """
    Classify the latest user text for safety and scope.

    Uses SafetyResult:
    - risk: "none" | "self_harm" | "eating_disorder" | "severe_addiction" | "violence" | "other"
    - action: "allow" | "block_and_escalate"
    - message: short, safe helper text
    """

    # Prefer the freshest user message; fall back to habit_description or empty string
    user_text = (
        getattr(state, "last_user_message", None)
        or getattr(state, "habit_description", None)
        or getattr(state, "user_input", "")
        or ""
    )

    prompt = SAFETY_PROMPT.format(user_text=user_text)

    llm = _json_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(SafetyResult)

    try:
        safety = structured_llm.invoke(prompt)
    except Exception:
        # Be conservative if safety fails: block & escalate instead of silently allowing
        safety = SafetyResult(
            risk="other",
            action="block_and_escalate",
            message=(
                "I’m here only for habit and behavior coaching, so I can’t safely respond to this. "
                "Please avoid medical, illegal, or harmful topics, and consider reaching out to a "
                "trusted person or local professional if you’re in distress."
            ),
        )

    return {"safety": safety}



def quiz_form_node(state: HabitState) -> Dict[str, Any]:
    """
    Generate a tailored 8–10 question MULTIPLE-CHOICE quiz based on the user's habit description.

    Guarantees:
    - The habit name / product (e.g. "Zyn", "TikTok", "porn") is preserved.
    - Questions are explicitly about THIS habit, not generic behavior.
    - Each question has EXACTLY 4 MCQ options.
    """
    habit_description = state.habit_description or ""

    prompt = QUIZ_GENERATOR_PROMPT.format(
        habit_description=habit_description
    )

    llm = ChatOpenAI(
        model=MODEL_JSON,
        temperature=0.4,
    )
    structured_llm = llm.with_structured_output(QuizForm)

    try:
        # Expecting the LLM to respect the QuizForm schema with MCQ options.
        quiz_form = structured_llm.invoke(prompt)
    except Exception:
        # Fallback that is STILL tied to the described habit
        habit_label = habit_description or "this habit"
        quiz_form = QuizForm(
            habit_name_guess=habit_label,
            questions=[
                {
                    "id": "q1",
                    "question": f"How often do you usually do {habit_label}?",
                    "helper_text": "Think about an average week.",
                    "options": [
                        {"id": "q1_a", "label": "Less than once a week", "helper_text": None},
                        {"id": "q1_b", "label": "1–3 times a week", "helper_text": None},
                        {"id": "q1_c", "label": "4–7 times a week", "helper_text": None},
                        {"id": "q1_d", "label": "Multiple times per day", "helper_text": None},
                    ],
                },
                {
                    "id": "q2",
                    "question": f"At what times of day does {habit_label} usually happen most?",
                    "helper_text": None,
                    "options": [
                        {"id": "q2_a", "label": "Morning", "helper_text": None},
                        {"id": "q2_b", "label": "Afternoon", "helper_text": None},
                        {"id": "q2_c", "label": "Evening", "helper_text": None},
                        {"id": "q2_d", "label": "Late night / before sleep", "helper_text": None},
                    ],
                },
                {
                    "id": "q3",
                    "question": f"Where are you most often when {habit_label} happens?",
                    "helper_text": "For example: bedroom, desk, bathroom, outside, with friends.",
                    "options": [
                        {"id": "q3_a", "label": "Mostly at home", "helper_text": None},
                        {"id": "q3_b", "label": "Mostly outside", "helper_text": None},
                        {"id": "q3_c", "label": "Mostly at work / study place", "helper_text": None},
                        {"id": "q3_d", "label": "It changes a lot", "helper_text": None},
                    ],
                },
                {
                    "id": "q4",
                    "question": f"Right before {habit_label}, you are usually…",
                    "helper_text": "Pick the closest option.",
                    "options": [
                        {"id": "q4_a", "label": "Bored or passing time", "helper_text": None},
                        {"id": "q4_b", "label": "Stressed / anxious / overwhelmed", "helper_text": None},
                        {"id": "q4_c", "label": "Lonely / sad", "helper_text": None},
                        {"id": "q4_d", "label": "Generally okay or even excited", "helper_text": None},
                    ],
                },
                {
                    "id": "q5",
                    "question": f"What usually triggers you to do {habit_label}?",
                    "helper_text": None,
                    "options": [
                        {"id": "q5_a", "label": "Seeing or having the product/device nearby", "helper_text": None},
                        {"id": "q5_b", "label": "Notifications, apps, or online content", "helper_text": None},
                        {"id": "q5_c", "label": "Specific people, places, or routines", "helper_text": None},
                        {"id": "q5_d", "label": "Mostly my internal feelings / thoughts", "helper_text": None},
                    ],
                },
                {
                    "id": "q6",
                    "question": f"How strong does the urge for {habit_label} feel when it appears?",
                    "helper_text": None,
                    "options": [
                        {"id": "q6_a", "label": "Mild – I can ignore it if needed", "helper_text": None},
                        {"id": "q6_b", "label": "Moderate – uncomfortable but manageable", "helper_text": None},
                        {"id": "q6_c", "label": "Strong – very hard to resist", "helper_text": None},
                        {"id": "q6_d", "label": "Overwhelming – I almost always give in", "helper_text": None},
                    ],
                },
                {
                    "id": "q7",
                    "question": f"Have you tried to cut down or stop {habit_label} before?",
                    "helper_text": None,
                    "options": [
                        {"id": "q7_a", "label": "No, this is my first serious attempt", "helper_text": None},
                        {"id": "q7_b", "label": "Yes, once or twice", "helper_text": None},
                        {"id": "q7_c", "label": "Yes, many times with short success", "helper_text": None},
                        {"id": "q7_d", "label": "Yes, but I never really committed", "helper_text": None},
                    ],
                },
                {
                    "id": "q8",
                    "question": f"Right now, how ready do you feel to change {habit_label}?",
                    "helper_text": None,
                    "options": [
                        {"id": "q8_a", "label": "Just exploring, not really ready", "helper_text": None},
                        {"id": "q8_b", "label": "Somewhat ready, but scared / unsure", "helper_text": None},
                        {"id": "q8_c", "label": "Ready to seriously reduce it", "helper_text": None},
                        {"id": "q8_d", "label": "Very ready – I want big change", "helper_text": None},
                    ],
                },
            ],
        )

    return {"quiz_form": quiz_form}


# ---------- Quiz Summary Node ----------

def quiz_summary_node(state: HabitState) -> Dict[str, Any]:
    """
    Convert:
    - original habit_description
    - AI-generated quiz_form
    - user_quiz_answers (now usually {question_id: option_id})

    into a compact QuizSummary JSON.
    """
    habit_description = state.habit_description or ""
    quiz_form_json = state.quiz_form.model_dump() if state.quiz_form else {}

    # Backwards compatible handling:
    # - If answers is a dict, use it as-is.
    # - If it's a string (old behaviour), wrap it under "freeform".
    raw_answers = getattr(state, "user_quiz_answers", None)

    if isinstance(raw_answers, dict):
        answers_payload = raw_answers
    elif isinstance(raw_answers, str) and raw_answers.strip():
        answers_payload = {"freeform": raw_answers}
    else:
        answers_payload = {}

    prompt = QUIZ_SUMMARY_PROMPT.format(
        habit_description=habit_description,
        quiz_form_json=json.dumps(quiz_form_json, ensure_ascii=False),
        user_quiz_answers=json.dumps(answers_payload, ensure_ascii=False),
    )

    llm = ChatOpenAI(
        model=MODEL_JSON,
        temperature=0.3,
    )
    structured_llm = llm.with_structured_output(QuizSummary)

    try:
        summary = structured_llm.invoke(prompt)
    except (ValidationError, Exception):
        # Defensive fallback – still honest, no hallucinated structure
        summary = QuizSummary(
            user_habit_raw=habit_description,
            canonical_habit_name=habit_description or "user habit",
            habit_category="other",
            category_confidence="low",
            product_type="unspecified",
            severity_level="mild",
            main_trigger="unknown",
            peak_times="unknown",
            common_locations="unknown",
            emotional_patterns="unclear",
            frequency_pattern="unknown",
            previous_attempts="not_clear",
            motivation_reason="user_wants_change",
            risk_situations="unknown",
        )

    return {"quiz_summary": summary}


def _category_guidance(summary: QuizSummary) -> str:
    """
    Rich category- and user-specific guidance so each habit type
    produces a structurally different 21-day plan.

    This is injected into the PLAN_21D_PROMPT as extra context.
    """

    cat = (summary.habit_category or "other").lower()
    severity = summary.severity_level
    name = summary.canonical_habit_name or summary.user_habit_raw or "the habit"
    raw = summary.user_habit_raw or ""
    trigger = summary.main_trigger or "unclear triggers"
    peak = summary.peak_times or "unclear peak times"
    loc = summary.common_locations or "unclear locations"
    emo = summary.emotional_patterns or "unclear emotional patterns"
    freq = summary.frequency_pattern or "unclear frequency"
    motive = summary.motivation_reason or "unclear motivation"
    risk = summary.risk_situations or "unclear risk situations"
    prev = summary.previous_attempts or "not clearly described"

    base_context = f"""
User-specific context:
- Exact wording: {raw}
- Canonical habit name: {name}
- Severity: {severity}
- Main trigger: {trigger}
- Peak times (when it MOST OFTEN happens, not the only possible times): {peak}
- Common locations (where it MOST OFTEN happens, not the only places): {loc}
- Emotional pattern: {emo}
- Frequency pattern: {freq}
- Motivation: {motive}
- High-risk situations: {risk}
- Previous attempts: {prev}

Global rules for using this information in the 21-day plan:
- Treat peak times and common locations as patterns or hotspots, NOT as exclusive rules.
- Do NOT assume the habit only happens at {loc} or only during {peak} unless the summary explicitly says so.
- At most about half of the daily tasks should explicitly mention a specific location like {loc}.
- At most about half of the daily tasks should explicitly mention a specific time window like {peak}.
- The remaining tasks should either be:
  - location-agnostic (work anywhere), or
  - clearly applicable across multiple contexts (home, outside, work, with friends, etc.).
- Use language like "often at home", "usually late at night", "especially in your room"
  instead of "only at home" or "always late at night", unless the summary literally says it ONLY happens there.
- Make sure at least a few tasks explicitly handle situations where the habit appears
  in other places or times than {loc} / {peak} (for example outside, with friends, or at random times).

Plan must explicitly reference these details across the 21 days, but in a BALANCED way that still works
if the habit also shows up in other places or times.
"""

    # Now add deep category-specific strategy
    if cat in ["nicotine_smoking", "nicotine_vaping", "nicotine_oral"]:
        cat_block = f"""
Category: Nicotine

Core strategy:
- Treat {name} as a dopamine and ritual loop, not just a chemical.
- Emphasize routines around peak times (for example {peak}), and environments like {loc}.
- Explicitly build friction around storage, access, purchase, and first use of the day.
- For oral products like pouches, include mouth and hand substitution tasks.
- For higher severity, include more aggressive environment restructuring and longer urge delays.

Must include across 21 days:
- At least 4 tasks about changing where {name} is kept or accessed.
- At least 4 tasks about first use of the day and last use window.
- At least 3 tasks about physical state regulation during withdrawal (sleep window, hydration, body movement).
- At least 3 tasks that use emotional patterns like {emo} to pre-empt urges.
"""

    elif cat == "pornography":
        cat_block = f"""
Category: Pornography / sexual content

Core strategy:
- Treat {name} as a privacy plus device plus emotional loop.
- Focus on device rules, room layout, and late-night behaviour, especially around {peak}.
- Explicitly design friction around entering high-risk locations such as {loc}.
- Use stimulus control (lights, door, blockers, charging locations) instead of just "willpower".
- Tie reflection tasks to shame cycles and emotion patterns like {emo}, but without using shame language.

Must include across 21 days:
- At least 4 tasks that change how and where the device is used.
- At least 3 tasks that pre-empt late-night or alone-time triggers.
- At least 3 tasks that redirect immediately after a strong urge into a specific alternative behaviour.
- At least 2 tasks that review a slip in a non-judgmental, purely diagnostic way.
"""

    elif cat in ["screen_time", "social_media", "gaming"]:
        cat_block = f"""
Category: Screen-based habit (social media, scrolling, or gaming)

Core strategy:
- Treat {name} as an algorithm plus environment plus boredom loop.
- Focus on first and last 30 minutes of the day, especially if peak times include {peak}.
- Redesign notification logic, home screen layout, and app availability.
- Use strong "screen zones" and "screen windows" instead of unrealistic total bans.
- Tie replacement activities to the motivation: {motive}.

Must include across 21 days:
- At least 3 tasks modifying notifications, app positions, or app removal.
- At least 3 tasks that change morning behaviour before the first use.
- At least 3 tasks that change evening behaviour and pre-sleep routines.
- At least 3 tasks that deliberately swap a high-risk scrolling window with something aligned to {motive}.
"""

    elif cat in ["alcohol", "cannabis"]:
        cat_block = f"""
Category: Substance use (alcohol or cannabis)

Core strategy:
- Treat {name} as a context plus people plus emotional regulation loop.
- Focus on social settings, routes, and specific times like {peak}.
- Include clear "no-use" contexts and re-routing strategies for high-risk places like {loc}.
- Include craving delay plus alternative rituals at the exact times they usually use.
- Tie medium-term tasks to motivation {motive} and long-term identity.

Must include across 21 days:
- At least 3 tasks that alter routes or places that usually lead to use.
- At least 3 tasks that create explicit "no-use" rules in specific contexts.
- At least 3 tasks focused on high-risk situations described as {risk}.
- At least 2 tasks rehearsing what to do during a social invite or stress spike.
"""

    elif cat in ["sugar", "food_overeating"]:
        cat_block = f"""
Category: Food / sugar / overeating

Core strategy:
- Treat {name} as a kitchen plus shopping plus emotional soothing loop.
- Focus on visibility and proximity of foods, especially around locations like {loc}.
- Tie tasks to emotional states like {emo} and times like {peak}.
- Include shopping list and preparation changes that reduce impulsive access.
- Use small plate, portion, and environment tricks rather than "never eat X again" rules.

Must include across 21 days:
- At least 3 tasks about shopping or preparing alternatives in advance.
- At least 3 tasks about changing visibility and proximity of trigger foods.
- At least 3 tasks about emotional check-ins before eating in high-risk moments.
- At least 2 tasks about how to handle evenings or specific risk situations like {risk}.
"""

    elif cat in ["shopping_spending", "gambling"]:
        cat_block = f"""
Category: Spending / gambling

Core strategy:
- Treat {name} as a excitement plus access plus impulse loop.
- Focus on financial access: cards, apps, cash, sites, groups.
- Use strong pre-commitment rules, delays, and visibility of consequences.
- Tie specific tasks to high-risk times or contexts like {peak} and {risk}.
- Use replacement forms of excitement or reward that are lower-risk.

Must include across 21 days:
- At least 3 tasks about restricting or delaying financial access.
- At least 3 tasks about changing what happens in the 10–20 minutes before spending or betting.
- At least 2 tasks about reviewing a past spending or gambling episode analytically, not emotionally.
- At least 2 tasks that explicitly reinforce the motivation: {motive}.
"""

    elif cat == "procrastination":
        cat_block = f"""
Category: Procrastination

Core strategy:
- Treat {name} as avoidance of a specific type of work or feeling.
- Tie tasks directly to the kind of work they avoid most (for example study or deep work).
- Use very small, clear start behaviours instead of vague discipline tasks.
- Design environment and time box rules around the true peak avoidance windows like {peak}.
- Link identity work to becoming someone who handles {trigger} with short, focused bursts.

Must include across 21 days:
- At least 5 tasks that define a tiny, concrete starting action (for example open document and write one sentence).
- At least 3 tasks that reduce distractions in the main work location {loc}.
- At least 3 tasks that handle emotional patterns like {emo} before work instead of during.
- At least 2 tasks that rehearse what to do after a bad day without abandoning the plan.
"""

    else:
        cat_block = f"""
Category: Other or unclear

Core strategy:
- The category label is not precise, so lean heavily on the user's actual patterns.
- Design tasks explicitly around the main trigger {trigger}, peak times {peak}, and locations {loc}.
- Use emotional pattern {emo} to time interventions before the urge becomes very strong.
- Apply standard habit-breaking tools: friction, replacement, identity, slip recovery, environment design.

Must include across 21 days:
- At least 5 tasks that directly reference the described triggers, times, or locations.
- At least 3 tasks that practice urge delay plus a named replacement behaviour.
- At least 2 tasks that explicitly connect daily actions to the motivation: {motive}.
"""

    return base_context + "\n" + cat_block


# ---------- 21-Day Plan Node ----------

def _fallback_plan21(quiz_summary: Optional[QuizSummary] = None) -> Plan21D:
    """
    Fallback 21-day plan if the LLM output fails validation.

    Uses QuizSummary if available (canonical_habit_name, main_trigger, motivation_reason),
    but does NOT rely on any old fields like 'habit_name'.
    """
    if quiz_summary:
        habit = (
            quiz_summary.canonical_habit_name
            or quiz_summary.user_habit_raw
            or "your habit"
        )
        trigger = quiz_summary.main_trigger or "your usual triggers"
        motive = quiz_summary.motivation_reason or "your reasons for change"
    else:
        habit = "your habit"
        trigger = "your usual triggers"
        motive = "your reasons for change"

    plan_summary = (
        f"This 21-day plan helps you reduce {habit} with small daily actions, "
        f"focusing on awareness, friction around {trigger}, and identity shifts based on {motive}."
    )

    day_tasks = {
        "day_1":  f"Write down when and why {habit} usually happens. No pressure to change yet.",
        "day_2":  f"Before each urge for {habit}, pause 30 seconds and name what you’re feeling.",
        "day_3":  f"Move one step further from your usual {trigger} location before acting.",
        "day_4":  f"Choose a 5-minute healthy activity to try once when an urge appears.",
        "day_5":  f"Disable one small cue that feeds {habit} (notification, tab, app, or object).",
        "day_6":  f"Set a clear daily cutoff time after which you do not allow {habit}.",
        "day_7":  "Slip-recovery: review this week, note one pattern, and adjust cutoff time if needed.",
        "day_8":  f"Delay {habit} by 5 minutes once today and do your chosen healthy activity first.",
        "day_9":  f"Change your usual {habit} location; do it somewhere less comfortable if you must.",
        "day_10": f"Tell future-you in a note why reducing {habit} matters over the next 3 months.",
        "day_11": f"Reduce one typical {habit} episode by half in time, intensity, or frequency.",
        "day_12": "Plan a simple evening routine that does not include your main trigger source.",
        "day_13": f"Practice one ‘urge surfing’ cycle: breathe, observe, and let one urge pass unacted.",
        "day_14": "Slip-recovery: list three things that went well and one small adjustment for next week.",
        "day_15": f"Define a rule: one specific situation where {habit} is no longer allowed at all.",
        "day_16": f"Replace one full {habit} episode with your healthy alternative, start to finish.",
        "day_17": "Prepare your environment tonight so tomorrow’s first hour is completely trigger-free.",
        "day_18": f"Teach someone (or journal) one insight you’ve learned about your {habit} triggers.",
        "day_19": "Create a 2-sentence identity statement about who you’re becoming without this habit.",
        "day_20": "Plan how you will keep these limits and routines going after Day 21.",
        "day_21": "Review progress, refresh your identity statement, and choose one long-term keystone rule.",
    }

    return Plan21D(plan_summary=plan_summary, day_tasks=day_tasks)

def plan21_node(state: HabitState) -> Dict[str, Any]:
    """
    Generate the 21-day plan using the QuizSummary as context
    + category-specific guidance so different habits feel truly different.
    """
    if not state.quiz_summary:
        return {"plan21": _fallback_plan21(None)}

    quiz_json = state.quiz_summary.model_dump()
    guidance = _category_guidance(state.quiz_summary)

    prompt = PLAN_21D_PROMPT.format(
        quiz_summary_json=json.dumps(quiz_json, ensure_ascii=False),
        category_guidance=guidance,
    )

    # 🔹 Use your JSON LLM helper, NOT MODEL_JSON, NOT _json_llm
    data = _llm_json(prompt, max_tokens=1600, temperature=0.35)

    try:
        # Basic sanitization
        day_tasks = data.get("day_tasks", {}) or {}
        for i in range(1, 21):
            key = f"day_{i}"
            if key not in day_tasks or not isinstance(day_tasks[key], str) or not day_tasks[key].strip():
                day_tasks[key] = _fallback_plan21(state.quiz_summary).day_tasks[key]

        data["day_tasks"] = day_tasks

        if "plan_summary" not in data or not isinstance(data["plan_summary"], str):
            data["plan_summary"] = (
                f"Personalized 21-day behavioural plan to reduce {state.quiz_summary.canonical_habit_name}."
            )

        plan = Plan21D(**data)
    except:
        plan = _fallback_plan21(state.quiz_summary)

    return {"plan21": plan}


def why_day_node(state: HabitState, day_key: str) -> Dict[str, Any]:
    """
    Given the current HabitState and a specific day key ("day_1", "day_2", ...),
    return an elite-level explanation of WHY this task exists.

    This version:
    - uses the same ChatOpenAI stack as the rest of the app
    - does NOT use WHY_DAY_PROMPT.format(...) (no KeyError on placeholders)
    - returns a dict so update_state() can attach explanation to HabitState
    """

    # -------- Basic guards -------- #
    if state.plan21 is None:
        # No plan = nothing to explain
        explanation = (
            "I can't explain this task yet because your 21-day plan hasn't been generated. "
            "Please complete the quiz and generate the plan first."
        )
        return {
            "last_why_day": day_key,
            "last_why_explanation": explanation,
        }

    if state.quiz_summary is None:
        explanation = (
            "I can't explain this task yet because the diagnostic summary is missing. "
            "Try re-running the quiz and plan generation."
        )
        return {
            "last_why_day": day_key,
            "last_why_explanation": explanation,
        }

    day_task = state.plan21.day_tasks.get(day_key)
    if not day_task:
        explanation = f"No task found for {day_key}, so there is nothing to explain."
        return {
            "last_why_day": day_key,
            "last_why_explanation": explanation,
        }

    # -------- Build context for the LLM -------- #
    quiz_summary_json = json.dumps(state.quiz_summary.model_dump(), ensure_ascii=False)
    plan_21d_json = json.dumps(state.plan21.model_dump(), ensure_ascii=False)
    habit_description = state.habit_description or ""

    # IMPORTANT: we DO NOT call WHY_DAY_PROMPT.format(...)
    # We just use it as a base instruction and append context ourselves.
    prompt_parts = [
        WHY_DAY_PROMPT,
        "\n\n---------------- CONTEXT ----------------\n",
        "User habit description:\n",
        habit_description,
        "\n\nQuiz summary JSON:\n",
        quiz_summary_json,
        "\n\nFull 21-day plan JSON:\n",
        plan_21d_json,
        "\n\nFOCUS DAY:\n",
        f"day_key: {day_key}\n",
        f"day_task: {day_task}\n",
        "\nExplain in 3–6 sentences:\n",
        "- Why this specific task exists in the protocol,\n",
        "- Which part of the habit mechanism it targets (trigger, craving, avoidance, reward, identity, environment),\n",
        "- Why it is placed at this point in the 21-day sequence.\n",
        "Be surgical and concrete. No generic motivation talk.\n",
    ]

    user_prompt = "".join(prompt_parts)

    # -------- Call LLM -------- #
    llm = _text_llm(temperature=0.4)

    fallback_explanation = (
        "This task targets a critical part of your habit loop: the moment between urge and escape. "
        "By executing it exactly as written, you train your brain to associate that discomfort with "
        "action and progress instead of avoidance and short-term relief."
    )

    try:
        resp = llm.invoke(user_prompt)
        explanation = (resp.content or "").strip()

        # If model returns something empty or garbage-short, use fallback
        if not explanation or len(explanation.split()) < 5:
            explanation = fallback_explanation

    except Exception as e:
        print(f"[why_day_node] Error while generating explanation for {day_key}: {e}")
        explanation = fallback_explanation

    # Return a dict so update_state() can attach this to HabitState
    return {
        "last_why_day": day_key,
        "last_why_explanation": explanation,
    }


# ---------- Coach Node ----------

def coach_node(state: HabitState) -> Dict[str, Any]:
    """
    Context-aware AI coach that uses:
    - safety (to block out-of-scope / dangerous requests)
    - quiz_summary
    - plan21
    - chat_history
    - last_user_message
    """

    # 1) Hard safety block for medical / illegal / minors / self-harm / violence / etc.
    # With the new SafetyResult, we check `action`, not `status`.
    safety = state.safety
    if safety is not None and getattr(safety, "action", None) == "block_and_escalate":
        reply = (
            "I’m here only for habit and behavior coaching, so I can’t help with medical, legal, "
            "explicit, or illegal requests. If this is about your health, safety, or a serious "
            "situation, please talk to a qualified professional or someone you trust in real life."
        )

        # update chat history even on blocked replies
        new_history = list(state.chat_history or [])
        user_message = state.last_user_message or state.habit_description or ""
        if user_message:
            new_history.append({"role": "user", "content": user_message})
        new_history.append({"role": "assistant", "content": reply})

        return {
            "coach_reply": reply,
            "chat_history": new_history,
        }

    # 2) Normal coaching flow (safe content)
    quiz_json = state.quiz_summary.model_dump() if state.quiz_summary else {}
    plan_json = state.plan21.model_dump() if state.plan21 else {}

    # Format history
    history_lines = []
    for msg in state.chat_history or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)

    user_message = state.last_user_message or state.habit_description or ""

    base_prompt = COACH_PROMPT + "\n\n"
    base_prompt += f"quiz_summary_json:\n{json.dumps(quiz_json, ensure_ascii=False)}\n\n"
    base_prompt += f"plan_21d_json:\n{json.dumps(plan_json, ensure_ascii=False)}\n\n"
    base_prompt += f"history_text:\n{history_text}\n\n"
    base_prompt += f"user_message:\n{user_message}\n"

    llm = _text_llm()
    try:
        reply = llm.invoke(base_prompt).content.strip()
    except Exception:
        reply = "Let’s focus on one small step you can do today that matches your plan."

    # update chat history
    new_history = list(state.chat_history or [])
    if user_message:
        new_history.append({"role": "user", "content": user_message})
    new_history.append({"role": "assistant", "content": reply})

    return {
        "coach_reply": reply,
        "chat_history": new_history,
    }

