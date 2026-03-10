"""Tests for cultural monitor."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx

from brand_conscience.layer0_awareness.cultural_monitor import CulturalMonitor

SAMPLE_TRENDS_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:ht="https://trends.google.com/trending/rss">
  <channel>
    <item>
      <title>Test Topic</title>
      <ht:approx_traffic>500,000+</ht:approx_traffic>
    </item>
    <item>
      <title>Another Topic</title>
      <ht:approx_traffic>100,000+</ht:approx_traffic>
    </item>
  </channel>
</rss>"""


def test_collect_signals_returns_list():
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.text = SAMPLE_TRENDS_RSS
    mock_response.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_response

    monitor = CulturalMonitor(http_client=mock_client)
    signals = monitor.collect_signals()
    assert isinstance(signals, list)
    assert len(signals) == 2


def test_collect_signals_handles_http_error():
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.get.side_effect = httpx.ConnectError("connection failed")

    monitor = CulturalMonitor(http_client=mock_client)
    signals = monitor.collect_signals()
    assert isinstance(signals, list)
    assert len(signals) == 0


def test_parse_trends_rss_extracts_topics():
    mock_client = MagicMock(spec=httpx.Client)
    monitor = CulturalMonitor(http_client=mock_client)
    signals = monitor._parse_trends_rss(SAMPLE_TRENDS_RSS)
    assert len(signals) == 2
    assert signals[0].topic == "Test Topic"
    assert signals[0].source == "google_trends"
    assert signals[0].velocity == 0.5  # 500000 / 1_000_000


def test_screen_for_safety_flags_unsafe():
    mock_client = MagicMock(spec=httpx.Client)
    mock_safety = MagicMock()
    mock_safety.check.return_value = (False, [{"category": "violence"}])

    monitor = CulturalMonitor(brand_safety=mock_safety, http_client=mock_client)
    signals = monitor._parse_trends_rss(SAMPLE_TRENDS_RSS)
    screened = monitor._screen_for_safety(signals)

    assert all(not s.is_safe for s in screened)
