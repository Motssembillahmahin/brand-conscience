"""Prompt builder — fills templates with strategic context."""

from __future__ import annotations

from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.layer2_prompts.templates import PromptTemplate, TemplateRegistry

logger = get_logger(__name__)


class PromptBuilder:
    """Build prompts by filling templates with context from strategic decisions."""

    def __init__(self, registry: TemplateRegistry | None = None) -> None:
        self._registry = registry or TemplateRegistry()

    @traced(name="build_prompts", tags=["layer2", "prompt"])
    def build(
        self,
        strategic_decision: dict,
        context: dict,
        action_type: str = "launch",
        max_prompts: int = 5,
    ) -> list[str]:
        """Build prompts from templates and strategic context.

        Args:
            strategic_decision: Dict from Layer 1 strategic agent.
            context: Additional context (product info, brand details).
            action_type: Recommended action type from MomentProfile.
            max_prompts: Maximum number of prompts to generate.

        Returns:
            List of filled prompt strings.
        """
        templates = self._registry.select_for_action(action_type)[:max_prompts]

        # Build context dict for template filling
        fill_context = {
            "audience_description": strategic_decision.get("audience_segment", "general"),
            "product_name": context.get("product_name", "our product"),
            "brand_name": context.get("brand_name", "our brand"),
            "brand_tone": context.get("brand_tone", "professional and modern"),
            "brand_values": context.get("brand_values", "quality and innovation"),
            "key_message": context.get("key_message", "discover the difference"),
            "style": context.get("style", "modern, clean, professional"),
            "setting": context.get("setting", "minimalist studio"),
            "scenario": context.get("scenario", "everyday use"),
            "mood": context.get("mood", "confident and aspirational"),
            "offer_details": context.get("offer_details", "limited time offer"),
            "urgency_message": context.get("urgency_message", "act now"),
            "narrative": context.get("narrative", "brand story"),
            "season": context.get("season", "current"),
            "theme": context.get("theme", "celebration"),
            "cultural_context": context.get("cultural_context", "contemporary"),
        }

        prompts = []
        for template in templates:
            try:
                prompt = self._fill_template(template, fill_context)
                prompts.append(prompt)
            except KeyError as e:
                logger.warning(
                    "template_fill_failed",
                    template=template.name,
                    missing_key=str(e),
                )

        logger.info(
            "prompts_built",
            count=len(prompts),
            templates_used=[t.name for t in templates[:len(prompts)]],
        )
        return prompts

    def _fill_template(self, template: PromptTemplate, context: dict) -> str:
        """Fill a template with context values, using defaults for missing keys."""
        try:
            return template.template.format(**context)
        except KeyError:
            # Fill what we can, leave unfilled placeholders as-is
            result = template.template
            for key, value in context.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result
