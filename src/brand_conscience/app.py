"""Master LangGraph composition — orchestrates all 6 layers."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)


class PipelineState(TypedDict, total=False):
    """Full pipeline state threading through all layers."""

    # Layer 0
    business_signals: list[dict]
    cultural_signals: list[dict]
    creative_signals: list[dict]
    moment_profile: dict

    # Layer 1
    active_campaigns: int
    total_spend: float
    budget_remaining: float
    strategic_decision: dict

    # Layer 2
    context: dict
    built_prompts: list[str]
    scored_prompts: list[dict]
    passing_prompts: list[str]

    # Layer 3
    generated_paths: list[str]
    evaluation_results: list[dict]
    approved_creative_ids: list[str]

    # Layer 4
    campaign_id: str
    ab_test_groups: list[str]
    deployment_status: str

    # Layer 5
    collected_metrics: list[dict]
    rewards: list[dict]

    # Control
    should_proceed: bool


def run_awareness(state: PipelineState) -> dict[str, Any]:
    """Layer 0: Collect signals and build moment profile."""
    from brand_conscience.layer0_awareness.business_monitor import BusinessMonitor
    from brand_conscience.layer0_awareness.creative_monitor import CreativeMonitor
    from brand_conscience.layer0_awareness.cultural_monitor import CulturalMonitor
    from brand_conscience.layer0_awareness.moment_profile import MomentProfileBuilder

    business = BusinessMonitor().collect_signals()
    cultural = CulturalMonitor().collect_signals()
    creative = CreativeMonitor().collect_signals()

    builder = MomentProfileBuilder()
    profile = builder.build(business, cultural, creative)

    should_proceed = profile.urgency_score > 0.3
    logger.info(
        "pipeline_awareness_complete",
        urgency=profile.urgency_score,
        action=profile.recommended_action.value,
        proceeding=should_proceed,
    )

    return {
        "moment_profile": profile.to_dict(),
        "should_proceed": should_proceed,
    }


def check_urgency(state: PipelineState) -> str:
    """Conditional edge: proceed only if urgency warrants action."""
    if state.get("should_proceed", False):
        return "safety_gate"
    return END


def safety_gate(state: PipelineState) -> dict[str, Any]:
    """Deterministic safety gate — pause only campaigns matched to unsafe signals."""
    from brand_conscience.common.notifications import SlackNotifier
    from brand_conscience.db.queries import get_live_campaigns
    from brand_conscience.layer4_deployment.campaign_manager import CampaignManager
    from brand_conscience.models.safety.impact_matcher import SafetyImpactMatcher

    profile = state.get("moment_profile", {})
    cultural_signals = profile.get("cultural_signals", [])
    unsafe_signals = [s for s in cultural_signals if not s.get("is_safe", True)]

    if not unsafe_signals:
        logger.info("safety_gate_clear", reason="no_unsafe_signals")
        return {"should_proceed": True}

    live_campaigns = get_live_campaigns()
    if not live_campaigns:
        logger.info("safety_gate_clear", reason="no_live_campaigns")
        return {"should_proceed": True}

    matcher = SafetyImpactMatcher()
    result = matcher.evaluate(unsafe_signals, live_campaigns)

    # Pause only matched campaigns
    manager = CampaignManager()
    notifier = SlackNotifier()
    signal_topics = [s.get("topic", "") for s in unsafe_signals]

    for campaign_id in result.campaigns_to_pause:
        match = result.match_details[campaign_id]
        try:
            manager.pause(campaign_id)
            notifier.send_safety_pause(
                campaign_id=campaign_id,
                campaign_name=match.campaign_name,
                reasons=match.match_reasons,
                signal_topics=signal_topics,
            )
            logger.info(
                "safety_gate_paused_campaign",
                campaign_id=campaign_id,
                reasons=match.match_reasons,
            )
        except Exception as exc:
            logger.warning(
                "safety_gate_pause_failed",
                campaign_id=campaign_id,
                error=str(exc),
            )

    # Continue pipeline if there are still safe campaigns or new campaigns needed
    has_safe = len(result.campaigns_safe) > 0
    logger.info(
        "safety_gate_complete",
        paused=len(result.campaigns_to_pause),
        safe=len(result.campaigns_safe),
        proceeding=has_safe,
    )
    return {"should_proceed": has_safe}


def check_after_safety(state: PipelineState) -> str:
    """Conditional edge after safety gate: proceed only if safe campaigns remain."""
    if state.get("should_proceed", False):
        return "run_strategy"
    return END


def run_strategy(state: PipelineState) -> dict[str, Any]:
    """Layer 1: Make strategic audience and budget decisions."""
    from brand_conscience.layer1_strategy.tasks import run_strategic_cycle

    decision = run_strategic_cycle(state.get("moment_profile", {}))
    return {"strategic_decision": decision}


def run_prompts(state: PipelineState) -> dict[str, Any]:
    """Layer 2: Build and score prompts."""
    from brand_conscience.layer2_prompts.tasks import run_prompt_cycle

    result = run_prompt_cycle(
        strategic_decision=state.get("strategic_decision", {}),
        context=state.get("context", {}),
    )
    return {
        "passing_prompts": result["passing_prompts"],
        "scored_prompts": result["prompts"],
    }


def run_creative(state: PipelineState) -> dict[str, Any]:
    """Layer 3: Generate and evaluate creatives."""
    from brand_conscience.layer3_creative.tasks import run_creative_cycle

    result = run_creative_cycle(
        passing_prompts=state.get("passing_prompts", []),
    )
    return {"approved_creative_ids": result["creative_ids"]}


def run_deployment(state: PipelineState) -> dict[str, Any]:
    """Layer 4: Deploy campaign to Meta."""
    from brand_conscience.layer4_deployment.tasks import run_deployment_cycle

    creative_ids = state.get("approved_creative_ids", [])
    if not creative_ids:
        logger.info("pipeline_no_approved_creatives")
        return {"deployment_status": "skipped"}

    result = run_deployment_cycle(
        strategic_decision=state.get("strategic_decision", {}),
        creative_ids=creative_ids,
    )
    return {
        "campaign_id": result["campaign_id"],
        "deployment_status": result["status"],
        "ab_test_groups": result["ab_groups"],
    }


def build_pipeline_graph() -> StateGraph:
    """Build the master pipeline graph composing all layers."""
    graph = StateGraph(PipelineState)

    graph.add_node("run_awareness", run_awareness)
    graph.add_node("safety_gate", safety_gate)
    graph.add_node("run_strategy", run_strategy)
    graph.add_node("run_prompts", run_prompts)
    graph.add_node("run_creative", run_creative)
    graph.add_node("run_deployment", run_deployment)

    graph.set_entry_point("run_awareness")
    graph.add_conditional_edges("run_awareness", check_urgency)
    graph.add_conditional_edges("safety_gate", check_after_safety)
    graph.add_edge("run_strategy", "run_prompts")
    graph.add_edge("run_prompts", "run_creative")
    graph.add_edge("run_creative", "run_deployment")
    graph.add_edge("run_deployment", END)

    return graph


def create_app() -> Any:
    """Create and compile the master pipeline graph."""
    from brand_conscience.common.checkpointing import get_checkpoint_saver

    graph = build_pipeline_graph()
    checkpointer = get_checkpoint_saver()

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("pipeline_graph_compiled")
    return compiled
