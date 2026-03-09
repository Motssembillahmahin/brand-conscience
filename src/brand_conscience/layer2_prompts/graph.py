"""LangGraph subgraph for Layer 2 — Prompt Engineering."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)


class PromptsState(TypedDict, total=False):
    """State for the prompts subgraph."""

    strategic_decision: dict
    context: dict
    action_type: str
    built_prompts: list[str]
    scored_prompts: list[dict]
    passing_prompts: list[str]


def build_prompts(state: PromptsState) -> dict[str, Any]:
    """Build prompts from templates and strategic context."""
    from brand_conscience.layer2_prompts.prompt_builder import PromptBuilder

    builder = PromptBuilder()
    prompts = builder.build(
        strategic_decision=state.get("strategic_decision", {}),
        context=state.get("context", {}),
        action_type=state.get("action_type", "launch"),
    )
    return {"built_prompts": prompts}


def score_prompts(state: PromptsState) -> dict[str, Any]:
    """Score and filter prompts through the scoring gate."""
    from brand_conscience.layer2_prompts.scoring_gate import ScoringGate

    gate = ScoringGate()
    results = gate.filter(state.get("built_prompts", []))

    scored = [{"prompt": r.prompt, "score": r.score, "passed": r.passed} for r in results]
    passing = [r.prompt for r in results if r.passed]

    return {"scored_prompts": scored, "passing_prompts": passing}


def build_prompts_graph() -> StateGraph:
    """Build the Layer 2 prompts subgraph."""
    graph = StateGraph(PromptsState)

    graph.add_node("build_prompts", build_prompts)
    graph.add_node("score_prompts", score_prompts)

    graph.set_entry_point("build_prompts")
    graph.add_edge("build_prompts", "score_prompts")
    graph.add_edge("score_prompts", END)

    return graph
