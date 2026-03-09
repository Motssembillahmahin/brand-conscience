"""LangGraph subgraph for Layer 0 — Awareness."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from brand_conscience.common.logging import get_logger
from brand_conscience.layer0_awareness.business_monitor import BusinessMonitor
from brand_conscience.layer0_awareness.creative_monitor import CreativeMonitor
from brand_conscience.layer0_awareness.cultural_monitor import CulturalMonitor
from brand_conscience.layer0_awareness.moment_profile import MomentProfileBuilder

logger = get_logger(__name__)


class AwarenessState(TypedDict, total=False):
    """State for the awareness subgraph."""

    business_signals: list[dict]
    cultural_signals: list[dict]
    creative_signals: list[dict]
    moment_profile: dict


def collect_business(state: AwarenessState) -> dict[str, Any]:
    monitor = BusinessMonitor()
    signals = monitor.collect_signals()
    return {"business_signals": signals}


def collect_cultural(state: AwarenessState) -> dict[str, Any]:
    monitor = CulturalMonitor()
    signals = monitor.collect_signals()
    return {"cultural_signals": signals}


def collect_creative(state: AwarenessState) -> dict[str, Any]:
    monitor = CreativeMonitor()
    signals = monitor.collect_signals()
    return {"creative_signals": signals}


def build_profile(state: AwarenessState) -> dict[str, Any]:
    builder = MomentProfileBuilder()
    profile = builder.build(
        business_signals=state.get("business_signals", []),
        cultural_signals=state.get("cultural_signals", []),
        creative_signals=state.get("creative_signals", []),
    )
    return {"moment_profile": profile.to_dict()}


def build_awareness_graph() -> StateGraph:
    """Build the Layer 0 awareness subgraph.

    Flow: collect signals in parallel → build moment profile.
    """
    graph = StateGraph(AwarenessState)

    graph.add_node("collect_business", collect_business)
    graph.add_node("collect_cultural", collect_cultural)
    graph.add_node("collect_creative", collect_creative)
    graph.add_node("build_profile", build_profile)

    graph.set_entry_point("collect_business")

    # All collectors feed into profile builder
    graph.add_edge("collect_business", "collect_cultural")
    graph.add_edge("collect_cultural", "collect_creative")
    graph.add_edge("collect_creative", "build_profile")
    graph.add_edge("build_profile", END)

    return graph
