"""
Unhabit AI – Production-Grade API

This module exposes a clean FastAPI app that wraps the
core Unhabit AI nodes (safety, quiz, summary, plan, coach, why-day, canonicalization).

File: api_main.py
"""

import logging
import os
import uuid
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from config import get_settings
from schemas import (
    HabitState,
    SafetyResult,
    QuizForm,
    QuizSummary,
    Plan21D,
)
from ai_nodes import (
    canonicalize_habit_node,
    safety_node,
    quiz_form_node,
    quiz_summary_node,
    plan21_node,
    _fallback_plan21,
    coach_node,
    why_day_node,
)

# --------------------------------------------------------------------
# Logging Setup
# --------------------------------------------------------------------

logger = logging.getLogger("unhabit_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------

settings = get_settings()

# --------------------------------------------------------------------
# API Models (Stable Contracts)
# --------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    details: Optional[Any] = None
    request_id: Optional[str] = None


# ---- Onboarding (safety + quiz_form) ----


class OnboardingStartRequest(BaseModel):
    """
    Entry point for a new habit.

    Frontend calls this as soon as the user types their habit.
    This runs:
      - safety_node
      - quiz_form_node

    and returns a HabitState containing:
      - user_id (if provided)
      - habit_description
      - safety
      - quiz_form
    """
    habit_description: str
    user_id: Optional[str] = None


# ---- Canonicalization (optional) ----


class CanonicalizeHabitRequest(BaseModel):
    """
    Optional helper for quickly classifying a raw habit sentence.

    Not required for the main flow (quiz-summary already does canonicalization),
    but useful if the frontend wants to show an instant chip like:
    "Nicotine (ZYN pouches) – severe".
    """
    habit_description: str


class CanonicalizeHabitResponse(BaseModel):
    habit_name: str
    habit_category: str
    severity_guess: int = 0
    confidence: float = 0.0  # numeric 0.0–1.0 for easier UI logic


# ---- Safety ----


class SafetyRequest(BaseModel):
    state: HabitState


# ---- Quiz ----


class QuizFormRequest(BaseModel):
    state: HabitState


class QuizSummaryRequest(BaseModel):
    state: HabitState


# ---- Plan ----


class PlanRequest(BaseModel):
    state: HabitState


class FallbackPlanRequest(BaseModel):
    state: HabitState


# ---- Coach ----


class CoachRequest(BaseModel):
    """
    - state.habit_description: required
    - state.quiz_summary: strongly recommended
    - state.plan21: improves coach quality
    - state.last_user_message: MUST be set by caller
    - state.chat_history: pass previous history to keep context
    """
    state: HabitState


class CoachResponse(BaseModel):
    coach_reply: str
    chat_history: Any


# ---- Why-Day ----


class WhyDayRequest(BaseModel):
    """
    Caller must include:
    - state.habit_description
    - state.quiz_summary
    - state.plan21

    day_number is 1–21 (we internally map to "day_1", "day_2", etc.)
    """
    state: HabitState
    day_number: int  # 1–21


class WhyDayResponse(BaseModel):
    day_number: int
    explanation: str


# --------------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------------

app = FastAPI(
    title="Unhabit AI – Habit Coach API",
    description=(
        "API for Unhabit: safety, quiz, quiz summary, 21-day plan, "
        "daily why-explanations, and coaching."
    ),
    version="1.0.0",
    contact={
        "name": "Unhabit AI",
        "url": "https://example.com",  # update if needed
    },
)

# CORS – allow all for now; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://unhabit.app"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------


@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    """Attach a request ID to each request and log basic info."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    try:
        response = await call_next(request)
    except Exception as exc:  # global safety net
        logger.exception(f"[{request_id}] Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                details="Unexpected error",
                request_id=request_id,
            ).model_dump(),
        )

    response.headers["X-Request-ID"] = request_id
    return response


# --------------------------------------------------------------------
# Exception Handlers
# --------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", None)
    logger.warning(f"[{request_id}] HTTPException {exc.status_code}: {exc.detail}")

    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        content = {**detail, "request_id": request_id}
    else:
        content = ErrorResponse(
            error=str(detail),
            details=None,
            request_id=request_id,
        ).model_dump()

    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    request_id = getattr(request.state, "request_id", None)
    logger.warning(f"[{request_id}] ValidationError: {exc.errors()}")

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation error",
            details=exc.errors(),
            request_id=request_id,
        ).model_dump(),
    )


# --------------------------------------------------------------------
# Utility Helpers
# --------------------------------------------------------------------


def require_openai_key():
    """Raise a clean error if OPENAI_API_KEY is not configured."""
    if not getattr(settings, "openai_api_key", None):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "OPENAI_API_KEY not configured",
                "details": "Set OPENAI_API_KEY in .env before running Unhabit AI.",
            },
        )


def _apply_node(state: HabitState, node_fn) -> HabitState:
    """
    Helper that mimics LangGraph behaviour:

    - Takes the current HabitState
    - Calls a node (which returns a partial dict)
    - Returns a NEW HabitState with those fields updated
    """
    result = node_fn(state)
    if not isinstance(result, dict):
        raise RuntimeError(f"Node {node_fn.__name__} must return a dict, got {type(result)}")

    return state.model_copy(update=result)


# --------------------------------------------------------------------
# Basic & Health
# --------------------------------------------------------------------


@app.get("/", tags=["meta"])
def root():
    return {
        "message": "Unhabit AI API is running.",
        "docs_url": "/docs",
        "environment": settings.env,
        "debug": settings.debug,
    }


@app.get("/health", tags=["meta"])
def health_check():
    return {
        "status": "ok",
        "openai_key_configured": bool(getattr(settings, "openai_api_key", None)),
        "environment": settings.env,
        "debug": settings.debug,
    }


# --------------------------------------------------------------------
# Onboarding (Safety + Quiz Form)
# --------------------------------------------------------------------


@app.post(
    "/onboarding/start",
    response_model=HabitState,
    tags=["onboarding"],
    summary="Run safety + quiz_form for a new habit",
)
def onboarding_start(req: OnboardingStartRequest):
    """
    This endpoint mirrors the first part of your LangGraph:

      safety_node -> quiz_form_node

    Frontend flow:
    - User types habit_description
    - Call /onboarding/start
    - You get back HabitState with:
        - user_id
        - habit_description
        - safety
        - quiz_form
    - Then you show the MCQ quiz to the user.
    """
    require_openai_key()

    try:
        # Build initial state
        state = HabitState(
            user_id=req.user_id,
            habit_description=req.habit_description,
        )

        # 1) Safety
        state = _apply_node(state, safety_node)

        # 2) Quiz form
        state = _apply_node(state, quiz_form_node)

        return state

    except Exception as e:
        logger.exception(f"onboarding_start failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "onboarding_start failed", "details": str(e)},
        )


# --------------------------------------------------------------------
# Canonicalization (Optional Helper)
# --------------------------------------------------------------------


@app.post(
    "/canonicalize-habit",
    response_model=CanonicalizeHabitResponse,
    tags=["canonicalization"],
    summary="Canonicalize raw habit text (optional helper)",
)
def canonicalize_habit(req: CanonicalizeHabitRequest):
    """
    Optional endpoint.

    The main flow does canonicalization inside /quiz-summary.
    Use this only if the frontend wants an *instant* classification
    as soon as the user types their habit.
    """
    require_openai_key()

    state = HabitState(habit_description=req.habit_description)

    try:
        node_result = canonicalize_habit_node(state) or {}
    except Exception as e:
        logger.exception(f"canonicalize_habit_node crashed: {e}")
        node_result = {}

    # Map node_result → API response
    habit_name = (
        node_result.get("habit_name")
        or node_result.get("canonical_habit_name")
        or req.habit_description
    )

    habit_category = node_result.get("habit_category", "unknown")

    # canonical_confidence is often "low" / "medium" / "high"
    conf_raw = (
        node_result.get("confidence")
        or node_result.get("canonical_confidence")
        or 0.0
    )

    if isinstance(conf_raw, str):
        conf_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        confidence = conf_map.get(conf_raw.lower(), 0.0)
    else:
        try:
            confidence = float(conf_raw)
        except (TypeError, ValueError):
            confidence = 0.0

    # No explicit severity in your node yet → keep 0 for now
    severity_guess = node_result.get("severity_guess", 0)
    try:
        severity_guess = int(severity_guess)
    except (TypeError, ValueError):
        severity_guess = 0

    return CanonicalizeHabitResponse(
        habit_name=habit_name,
        habit_category=habit_category,
        severity_guess=severity_guess,
        confidence=confidence,
    )


# --------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------


@app.post(
    "/safety",
    response_model=SafetyResult,
    tags=["safety"],
    summary="Classify habit / text for safety",
)
def run_safety(req: SafetyRequest):
    """
    Wraps ai_nodes.safety_node(state) which returns {"safety": SafetyResult}.
    """
    require_openai_key()

    try:
        result = safety_node(req.state)
        safety = result.get("safety")
        if isinstance(safety, SafetyResult):
            return safety
        raise RuntimeError("safety_node did not return SafetyResult under 'safety'")
    except Exception as e:
        logger.exception(f"safety_node failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "safety_node failed", "details": str(e)},
        )


# --------------------------------------------------------------------
# Quiz
# --------------------------------------------------------------------


@app.post(
    "/quiz-form",
    response_model=QuizForm,
    tags=["quiz"],
    summary="Generate diagnostic quiz (MCQ)",
)
def generate_quiz(req: QuizFormRequest):
    """
    Wraps ai_nodes.quiz_form_node(state) which returns {"quiz_form": QuizForm}.

    Usually you won't call this directly if you use /onboarding/start,
    but it is available if the backend wants more granular control.
    """
    require_openai_key()

    try:
        result = quiz_form_node(req.state)
        quiz_form = result.get("quiz_form")
        if isinstance(quiz_form, QuizForm):
            return quiz_form
        raise RuntimeError("quiz_form_node did not return QuizForm under 'quiz_form'")
    except Exception as e:
        logger.exception(f"quiz_form_node failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "quiz_form_node failed", "details": str(e)},
        )


@app.post(
    "/quiz-summary",
    response_model=QuizSummary,
    tags=["quiz"],
    summary="Summarize quiz into mechanistic profile (canonicalization happens here)",
)
def summarize_quiz(req: QuizSummaryRequest):
    """
    This is the BRAIN of the system.

    It wraps ai_nodes.quiz_summary_node(state) which returns {"quiz_summary": QuizSummary}.

    The QuizSummary includes:
      - user_habit_raw
      - canonical_habit_name
      - habit_category
      - severity_level
      - category_confidence
      - core_loop, primary_payoff, avoidance_target, etc.

    Flow:
    - After user answers the MCQ quiz, frontend sends:
        - state.habit_description
        - state.quiz_form
        - state.user_quiz_answers
    - This endpoint returns the full QuizSummary, which is then used by /plan-21d and /coach.
    """
    require_openai_key()

    try:
        result = quiz_summary_node(req.state)
        summary = result.get("quiz_summary")
        if isinstance(summary, QuizSummary):
            return summary
        raise RuntimeError("quiz_summary_node did not return QuizSummary under 'quiz_summary'")
    except Exception as e:
        logger.exception(f"quiz_summary_node failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "quiz_summary_node failed", "details": str(e)},
        )


# --------------------------------------------------------------------
# 21-Day Plan
# --------------------------------------------------------------------


@app.post(
    "/plan-21d",
    response_model=Plan21D,
    tags=["plan"],
    summary="Generate main 21-day reduction plan",
)
def generate_plan(req: PlanRequest):
    """
    Wraps ai_nodes.plan21_node(state) which returns {"plan21": Plan21D}.

    Required:
    - state.quiz_summary

    If quiz_summary is missing or the LLM fails, the node falls back internally
    to a deterministic _fallback_plan21.
    """
    require_openai_key()

    try:
        result = plan21_node(req.state)
        plan = result.get("plan21")
        if isinstance(plan, Plan21D):
            return plan
        raise RuntimeError("plan21_node did not return Plan21D under 'plan21'")
    except Exception as e:
        logger.exception(f"plan21_node failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "plan21_node failed", "details": str(e)},
        )


@app.post(
    "/plan-21d-fallback",
    response_model=Plan21D,
    tags=["plan"],
    summary="Generate fallback 21-day plan (no LLM)",
)
def generate_fallback_plan(req: FallbackPlanRequest):
    """
    Directly calls the internal _fallback_plan21 helper for deterministic plans.

    Useful for:
    - Testing
    - Very strict cost control environments
    - When you want a guaranteed-safe default plan
    """
    try:
        plan = _fallback_plan21(req.state.quiz_summary)
        return plan
    except Exception as e:
        logger.exception(f"_fallback_plan21 failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "fallback_plan21 failed", "details": str(e)},
        )


# --------------------------------------------------------------------
# Coach
# --------------------------------------------------------------------


@app.post(
    "/coach",
    response_model=CoachResponse,
    tags=["coach"],
    summary="Context-aware daily coach",
)
def coach(req: CoachRequest):
    """
    Wraps ai_nodes.coach_node(state) which returns:
    {
        "coach_reply": str,
        "chat_history": [...]
    }

    Caller responsibilities:
    - Pass quiz_summary + plan21 for maximum context.
    - Set state.last_user_message to the latest user input.
    - Pass state.chat_history from previous turns to maintain conversation.
    """
    require_openai_key()

    try:
        result = coach_node(req.state)
        reply = result.get("coach_reply", "")
        history = result.get("chat_history", [])

        if not isinstance(reply, str):
            reply = str(reply)

        return CoachResponse(coach_reply=reply, chat_history=history)

    except Exception as e:
        logger.exception(f"coach_node failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "coach_node failed", "details": str(e)},
        )


# --------------------------------------------------------------------
# Why This Day?
# --------------------------------------------------------------------


@app.post(
    "/why-day",
    response_model=WhyDayResponse,
    tags=["plan"],
    summary="Explain why a specific day/task exists",
)
def why_day(req: WhyDayRequest):
    """
    Explain why a specific day in the plan exists.

    Caller must:
    - Include plan21 and quiz_summary inside HabitState.
    - Send day_number (1–21).

    We convert to a day_key like "day_1" and delegate to ai_nodes.why_day_node.
    """
    require_openai_key()

    if req.day_number < 1 or req.day_number > 21:
        raise HTTPException(
            status_code=400,
            detail={"error": "day_number must be between 1 and 21"},
        )

    day_key = f"day_{req.day_number}"

    try:
        explanation = why_day_node(req.state, day_key)

        if not isinstance(explanation, str):
            explanation = str(explanation)

        return WhyDayResponse(day_number=req.day_number, explanation=explanation)

    except Exception as e:
        logger.exception(f"why_day_node failed for {day_key}: {e}")
        # Still return a generic explanation, not a 500, to keep UX smooth
        fallback = (
            "This day is placed here to target a critical part of your habit loop: "
            "the moment between urge and escape. Doing this task exactly as written "
            "weakens the old pattern and strengthens the identity of someone who executes."
        )
        return WhyDayResponse(day_number=req.day_number, explanation=fallback)


# --------------------------------------------------------------------
# Local dev runner
# --------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
