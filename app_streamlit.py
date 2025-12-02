import os
import json
import requests
import streamlit as st

from schemas import HabitState, QuizForm, QuizSummary, Plan21D

# --------------------- API Client --------------------- #

API_BASE = os.getenv("UNHABIT_API_BASE", "http://localhost:8000")


def call_api(path: str, payload: dict) -> dict:
    """
    Helper to call the Unhabit FastAPI.

    - path: e.g. "/onboarding/start"
    - payload: dict that will be sent as JSON

    Raises RuntimeError with details if the API responds with 4xx/5xx.
    """
    url = f"{API_BASE}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=60)
    except Exception as e:
        raise RuntimeError(f"API {path} request failed: {e}")

    if resp.status_code >= 400:
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        raise RuntimeError(f"API {path} failed: {resp.status_code} – {data}")

    return resp.json()


# --------------------- Streamlit setup --------------------- #

st.set_page_config(
    page_title="Unhabit AI – Habit Coach",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Unhabit AI – 21-Day Habit Coach")
st.caption("AI-powered habit reduction with personalized quiz, 21-day plan, and a context-aware coach.")


# --------------------- Session State helpers --------------------- #

def init_state():
    if "habit_state" not in st.session_state:
        st.session_state.habit_state = HabitState()
    if "quiz_answers_cache" not in st.session_state:
        # {question_id: option_id} for MCQs
        st.session_state.quiz_answers_cache = {}


init_state()


def update_state(partial: dict):
    """
    Apply partial updates to the HabitState object in session.
    (We still use this for local fields like last_why_day, etc.)
    """
    state: HabitState = st.session_state.habit_state
    for key, value in partial.items():
        setattr(state, key, value)
    st.session_state.habit_state = state


def reset_app():
    st.session_state.clear()
    init_state()


# --------------------- UI Sections --------------------- #

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Reset all", use_container_width=True):
        reset_app()
        st.experimental_rerun()

    st.markdown("### Debug info")
    state: HabitState = st.session_state.habit_state
    st.json(
        {
            "safety": state.safety.model_dump() if state.safety else None,
            "has_quiz_form": state.quiz_form is not None,
            "has_quiz_summary": state.quiz_summary is not None,
            "has_plan21": state.plan21 is not None,
            "chat_messages": len(state.chat_history),
        },
        expanded=False,
    )

# Main layout: 3 columns
col_left, col_mid, col_right = st.columns([1.2, 1.5, 1.5])


# ----------------------------------------------------
# STEP 1: Habit description + Safety + Quiz generation
# ----------------------------------------------------
with col_left:
    st.subheader("1️⃣ Describe your habit")

    state: HabitState = st.session_state.habit_state

    habit_text = st.text_area(
        "What habit do you want to reduce?",
        value=state.habit_description or "",
        placeholder="Example: I'm addicted to Zyn pouches and use them all day.",
        height=130,
        key="habit_input",
    )

    generate_quiz_clicked = st.button("Generate quiz questions", type="primary")

    if generate_quiz_clicked:
        if not habit_text.strip():
            st.warning("Please describe your habit first.")
        else:
            # Update habit description in state
            state.habit_description = habit_text.strip()

            try:
                # 🔥 Call API: safety + quiz_form
                data = call_api(
                    "/onboarding/start",
                    {
                        "habit_description": state.habit_description,
                        "user_id": state.user_id,
                    },
                )

                # API returns a full HabitState
                st.session_state.habit_state = HabitState(**data)
                state = st.session_state.habit_state

                # Hard-stop if safety blocks
                if state.safety and state.safety.action == "block_and_escalate":
                    st.error(
                        "❌ I’m here only for habit and behavior coaching, so I can’t help with medical, "
                        "illegal, explicit, or harmful requests. If this is about your health, safety, or a "
                        "serious situation, please reach out to a trusted person or a local professional."
                    )
                    st.stop()

                st.success("✅ Quiz generated. Scroll to step 2 to answer the questions.")

            except RuntimeError as e:
                st.error(f"Failed to generate quiz via API: {e}")

    # Show safety status if available
    if state.safety:
        if state.safety.action == "allow":
            st.success(
                f"Safety status: OK ✅  \n"
                f"Risk classification: {state.safety.risk}"
            )
        else:
            st.error(
                f"Safety action: {state.safety.action}  \n"
                f"Risk classification: {state.safety.risk}  \n"
                f"Message: {state.safety.message}"
            )


# ----------------------------------------------------
# STEP 2: Show quiz + collect answers + generate plan
# ----------------------------------------------------
with col_mid:
    st.subheader("2️⃣ Answer your personalized quiz")

    state: HabitState = st.session_state.habit_state
    quiz_form = state.quiz_form

    if quiz_form is None:
        st.info("Generate the quiz first from step 1 to see questions here.")
    else:
        st.markdown(f"**AI's understanding of your habit:** `{quiz_form.habit_name_guess}`")
        st.markdown("---")

        # Ensure we have a local cache dict
        if "quiz_answers_cache" not in st.session_state:
            st.session_state.quiz_answers_cache = {}

        answers_cache = st.session_state.quiz_answers_cache

        # Display MCQ questions
        for q in quiz_form.questions:
            st.markdown(f"**{q.question}**")
            if q.helper_text:
                st.caption(q.helper_text)

            # Build options list
            option_labels = [opt.label for opt in q.options]
            option_ids = [opt.id for opt in q.options]

            # Determine preselected option (from cache or state.user_quiz_answers)
            preselected_index = 0
            selected_option_id = answers_cache.get(q.id)

            if selected_option_id and selected_option_id in option_ids:
                preselected_index = option_ids.index(selected_option_id)
            elif isinstance(state.user_quiz_answers, dict):
                existing = state.user_quiz_answers.get(q.id)
                if existing and existing in option_ids:
                    preselected_index = option_ids.index(existing)

            # Radio for MCQ selection
            choice_index = st.radio(
                "Choose one:",
                options=list(range(len(option_labels))),
                format_func=lambda i: option_labels[i],
                index=preselected_index,
                key=f"quiz_answer_{q.id}",
            )

            # Store selected option id in cache
            answers_cache[q.id] = option_ids[choice_index]

            st.markdown("---")

        # Button to generate 21-day plan
        if st.button("Generate my 21-day plan", type="primary", key="generate_plan_btn"):
            # Structured dict: {question_id: option_id}
            answers_dict = dict(st.session_state.quiz_answers_cache)

            # Store directly as dict in HabitState (matches schema)
            state = st.session_state.habit_state
            state.user_quiz_answers = answers_dict

            try:
                # 1) Summarize quiz via API
                summary_json = call_api(
                    "/quiz-summary",
                    {
                        "state": state.model_dump(),
                    },
                )
                state.quiz_summary = QuizSummary(**summary_json)

                # 2) Generate plan via API
                plan_json = call_api(
                    "/plan-21d",
                    {
                        "state": state.model_dump(),
                    },
                )
                state.plan21 = Plan21D(**plan_json)

                # 3) Reset WHY-info for new plan (local)
                state.last_why_day = None
                state.last_why_explanation = None

                # 4) Generate first coach reply via API
                state.last_user_message = None
                coach_json = call_api(
                    "/coach",
                    {
                        "state": state.model_dump(),
                    },
                )
                state.coach_reply = coach_json.get("coach_reply", "")
                state.chat_history = coach_json.get("chat_history", [])

                st.session_state.habit_state = state

                state = st.session_state.habit_state
                if state.safety and state.safety.action == "allow":
                    st.success(
                        f"Plan generated and safety still OK ✅  \n"
                        f"Risk classification: {state.safety.risk}"
                    )
                else:
                    st.warning("Plan generated, but safety indicates this may need review.")

            except RuntimeError as e:
                st.error(f"Failed to generate plan via API: {e}")


# ----------------------------------------------------
# STEP 3: Show plan + coach chat + WHY engine
# ----------------------------------------------------
with col_right:
    st.subheader("3️⃣ Your 21-day plan & AI coach")

    state: HabitState = st.session_state.habit_state
    plan = state.plan21

    if plan is None:
        st.info("Complete the quiz and generate your plan in step 2 to see it here.")
    else:
        # Show plan summary
        st.markdown("#### 📋 Plan summary")
        st.write(plan.plan_summary)

        st.markdown("#### 📅 Daily tasks")

        # Iterate days in order
        for day_key in sorted(plan.day_tasks.keys(), key=lambda x: int(x.split("_")[1])):
            state = st.session_state.habit_state
            task_text = state.plan21.day_tasks[day_key]

            with st.container():
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(
                        f"**{day_key.replace('_', ' ').title()}**: {task_text}"
                    )
                with cols[1]:
                    if st.button("Why?", key=f"why_btn_{day_key}"):
                        try:
                            day_num = int(day_key.split("_")[1])
                            why_json = call_api(
                                "/why-day",
                                {
                                    "state": state.model_dump(),
                                    "day_number": day_num,
                                },
                            )
                            explanation = why_json.get("explanation", "")

                            # Save locally in state for display
                            state.last_why_day = day_key
                            state.last_why_explanation = explanation
                            st.session_state.habit_state = state

                            st.rerun()  # refresh so caption shows under correct day

                        except RuntimeError as e:
                            st.error(f"Why this task? API failed: {e}")
                        except Exception as e:
                            st.error(f"Why this task? Unexpected error: {e}")

                # After the button, show explanation if this is the last-asked day
                state = st.session_state.habit_state
                if (
                    getattr(state, "last_why_day", None) == day_key
                    and getattr(state, "last_why_explanation", None)
                ):
                    st.caption(f"**Why this task?** {state.last_why_explanation}")

            st.markdown("---")

        st.markdown("#### 🧑‍🏫 AI Coach")

        # Show chat history
        if state.chat_history:
            for msg in state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Coach:** {msg['content']}")
        elif state.coach_reply:
            # First message from coach if history empty
            st.markdown(f"**Coach:** {state.coach_reply}")

        st.markdown("---")

        # Chat input
        user_msg = st.text_input(
            "Ask your coach something about your habit, your plan, or a slip:",
            key="coach_input",
            placeholder="Example: I slipped on day 3. What should I do now?",
        )

        if st.button("Send to coach", key="send_to_coach_btn"):
            if not user_msg.strip():
                st.warning("Please type a message for the coach.")
            else:
                state = st.session_state.habit_state
                state.last_user_message = user_msg.strip()

                try:
                    coach_json = call_api(
                        "/coach",
                        {
                            "state": state.model_dump(),
                        },
                    )

                    state.coach_reply = coach_json.get("coach_reply", "")
                    state.chat_history = coach_json.get("chat_history", [])
                    st.session_state.habit_state = state

                    state = st.session_state.habit_state
                    if state.safety and state.safety.action == "allow":
                        st.success(
                            f"Safety status: OK ✅  \n"
                            f"Risk classification: {state.safety.risk}"
                        )
                    else:
                        st.warning(
                            "Coach replied, but latest safety check suggests this may be sensitive content."
                        )

                    st.rerun()  # refresh to show updated chat

                except RuntimeError as e:
                    st.error(f"Coach API call failed: {e}")
