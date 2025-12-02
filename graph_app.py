from langgraph.graph import StateGraph, END
from schemas import HabitState
from ai_nodes import (
    safety_node,
    quiz_form_node,
    quiz_summary_node,
    plan21_node,
    coach_node,
)


def build_onboarding_graph():
    """
    Full onboarding flow:

    1) safety      – classify habit description.
    2) quiz_form   – AI generates tailored questions.
    3) (frontend asks questions & fills user_quiz_answers)
    4) quiz_summary– compress description + quiz + answers.
    5) plan21      – generate personalized 21-day plan.
    6) coach       – first coach message.
    """
    graph = StateGraph(HabitState)

    graph.add_node("safety", safety_node)
    graph.add_node("quiz_form", quiz_form_node)
    graph.add_node("quiz_summary", quiz_summary_node)
    graph.add_node("plan21", plan21_node)
    graph.add_node("coach", coach_node)

    graph.set_entry_point("safety")

    graph.add_edge("safety", "quiz_form")
    graph.add_edge("quiz_form", "quiz_summary")
    graph.add_edge("quiz_summary", "plan21")
    graph.add_edge("plan21", "coach")
    graph.add_edge("coach", END)

    return graph.compile()
