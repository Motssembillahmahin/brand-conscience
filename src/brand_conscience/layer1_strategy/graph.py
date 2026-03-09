"""LangGraph subgraph for Layer 1 — Strategy."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)


class StrategyState(TypedDict, total=False):
    """State for the strategy subgraph."""

    moment_profile: dict
    active_campaigns: int
    total_spend: float
    budget_remaining: float
    strategic_decision: dict


def make_strategic_decision(state: StrategyState) -> dict[str, Any]:
    """Run the strategic RL agent."""
    from brand_conscience.common.types import ActionType
    from brand_conscience.layer0_awareness.signals import MomentProfile
    from brand_conscience.layer1_strategy.strategic_agent import StrategicAgent

    profile_dict = state.get("moment_profile", {})

    profile = MomentProfile(
        id=profile_dict.get("id", ""),
        urgency_score=profile_dict.get("urgency_score", 0.0),
        recommended_action=ActionType(profile_dict.get("recommended_action", "hold")),
        affected_categories=profile_dict.get("affected_categories", []),
        affected_audiences=profile_dict.get("affected_audiences", []),
        context_summary=profile_dict.get("context_summary", ""),
    )

    agent = StrategicAgent()
    decision = agent.decide(
        moment_profile=profile,
        active_campaigns=state.get("active_campaigns", 0),
        total_spend=state.get("total_spend", 0.0),
        budget_remaining=state.get("budget_remaining", 0.0),
    )

    return {"strategic_decision": decision.to_dict()}


def build_strategy_graph() -> StateGraph:
    """Build the Layer 1 strategy subgraph."""
    graph = StateGraph(StrategyState)

    graph.add_node("strategic_decision", make_strategic_decision)
    graph.set_entry_point("strategic_decision")
    graph.add_edge("strategic_decision", END)

    return graph
