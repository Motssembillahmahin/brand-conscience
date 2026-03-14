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
# Formula: Subject + Setting + Style + Lighting + Composition + Details + Quality
DEFAULT_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        name="product_hero",
        template=(
            "{product_name} as the hero subject, {key_message}. "
            "Setting: {setting}. "
            "Style: {style}. "
            "Lighting: {lighting}. "
            "Camera: {composition}. "
            "Details: {details}, brand tone is {brand_tone}. "
            "{quality}."
        ),
        category="product",
        tags=["hero", "product"],
    ),
    PromptTemplate(
        name="lifestyle",
        template=(
            "{scenario} with {product_name} naturally integrated into the scene. "
            "Setting: {setting}. "
            "Style: {style}. "
            "Lighting: {lighting}. "
            "Camera: {composition}. "
            "Details: {details}, mood is {mood}. "
            "{quality}."
        ),
        category="lifestyle",
        tags=["lifestyle", "aspirational"],
    ),
    PromptTemplate(
        name="promotional",
        template=(
            "Promotional advertisement for {product_name}, {offer_details}. "
            "Setting: {setting}, {urgency_message}. "
            "Style: {style}, eye-catching and bold. "
            "Lighting: {lighting}. "
            "Camera: {composition}. "
            "Details: {details}. "
            "{quality}."
        ),
        category="promo",
        tags=["promo", "sale", "urgent"],
    ),
    PromptTemplate(
        name="brand_awareness",
        template=(
            "Brand awareness image for {brand_name}, conveying {brand_values}. "
            "Setting: {setting}, visual narrative of {narrative}. "
            "Style: {style}. "
            "Lighting: {lighting}. "
            "Camera: {composition}. "
            "Details: {details}. "
            "{quality}."
        ),
        category="brand",
        tags=["awareness", "brand"],
    ),
    PromptTemplate(
        name="seasonal",
        template=(
            "{season} seasonal advertisement for {product_name}, theme: {theme}. "
            "Setting: {setting}, cultural context: {cultural_context}. "
            "Style: {style}. "
            "Lighting: {lighting}. "
            "Camera: {composition}. "
            "Details: {details}. "
            "{quality}."
        ),
        category="seasonal",
        tags=["seasonal", "timely"],
    ),
]


class TemplateRegistry:
    """Registry of prompt templates."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {t.name: t for t in DEFAULT_TEMPLATES}

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
            return [
                t for t in self._templates.values() if "lifestyle" in t.tags or "hero" in t.tags
            ]
        elif action_type == "adjust":
            return [t for t in self._templates.values() if "promo" in t.tags]
        return self.list_templates()
