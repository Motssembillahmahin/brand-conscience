"""Prompt template registry."""

from __future__ import annotations

from dataclasses import dataclass, field

from brand_conscience.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """A prompt template with placeholders for context filling."""

    name: str
    template: str
    category: str = "general"
    tags: list[str] = field(default_factory=list)


# Built-in templates
DEFAULT_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        name="product_hero",
        template=(
            "Create a {style} advertisement image for {product_name}. "
            "Target audience: {audience_description}. "
            "Key message: {key_message}. "
            "Brand tone: {brand_tone}. "
            "Setting: {setting}."
        ),
        category="product",
        tags=["hero", "product"],
    ),
    PromptTemplate(
        name="lifestyle",
        template=(
            "Create a lifestyle advertisement showing {scenario}. "
            "Product: {product_name} naturally integrated into the scene. "
            "Target demographic: {audience_description}. "
            "Mood: {mood}. Style: {style}."
        ),
        category="lifestyle",
        tags=["lifestyle", "aspirational"],
    ),
    PromptTemplate(
        name="promotional",
        template=(
            "Design a promotional advertisement for {product_name}. "
            "Offer: {offer_details}. "
            "Urgency: {urgency_message}. "
            "Target: {audience_description}. "
            "Visual style: {style}, eye-catching and bold."
        ),
        category="promo",
        tags=["promo", "sale", "urgent"],
    ),
    PromptTemplate(
        name="brand_awareness",
        template=(
            "Create a brand awareness advertisement for {brand_name}. "
            "Brand values: {brand_values}. "
            "Visual narrative: {narrative}. "
            "Style: {style}. "
            "Audience: {audience_description}."
        ),
        category="brand",
        tags=["awareness", "brand"],
    ),
    PromptTemplate(
        name="seasonal",
        template=(
            "Design a {season} seasonal advertisement for {product_name}. "
            "Theme: {theme}. "
            "Cultural context: {cultural_context}. "
            "Target: {audience_description}. "
            "Style: {style}."
        ),
        category="seasonal",
        tags=["seasonal", "timely"],
    ),
]


class TemplateRegistry:
    """Registry of prompt templates."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {
            t.name: t for t in DEFAULT_TEMPLATES
        }

    def get(self, name: str) -> PromptTemplate | None:
        return self._templates.get(name)

    def list_templates(self, category: str | None = None) -> list[PromptTemplate]:
        templates = list(self._templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    def register(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template
        logger.info("template_registered", name=template.name)

    def select_for_action(self, action_type: str) -> list[PromptTemplate]:
        """Select appropriate templates based on recommended action."""
        if action_type == "launch":
            return self.list_templates()
        elif action_type == "refresh":
            return [t for t in self._templates.values() if "lifestyle" in t.tags or "hero" in t.tags]
        elif action_type == "adjust":
            return [t for t in self._templates.values() if "promo" in t.tags]
        return self.list_templates()
