"""Tests for Gemini client."""

from __future__ import annotations

from brand_conscience.layer3_creative.gemini_client import GeminiClient


def test_gemini_client_init():
    client = GeminiClient(api_key="test-key")
    assert client._api_key == "test-key"
