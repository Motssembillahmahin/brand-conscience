"""Cultural monitor — social trends, sentiment, brand safety (1-hour cadence)."""

from __future__ import annotations

import contextlib

import httpx

from brand_conscience.common.config import get_settings
from brand_conscience.common.logging import get_logger
from brand_conscience.common.tracing import traced
from brand_conscience.layer0_awareness.signals import CulturalSignal
from brand_conscience.models.safety.brand_safety import BrandSafetyClassifier

logger = get_logger(__name__)

# Trending topic sources (Google Trends RSS, NewsAPI, etc.)
_GOOGLE_TRENDS_RSS = "https://trends.google.com/trending/rss?geo=US"


class CulturalMonitor:
    """Collect cultural and social trend signals.

    Fetches trending topics from Google Trends RSS and news APIs,
    then passes every signal through the brand safety classifier.
    """

    def __init__(
        self,
        brand_safety: BrandSafetyClassifier | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._safety = brand_safety or BrandSafetyClassifier()
        self._http = http_client or httpx.Client(timeout=15.0)
        self._settings = get_settings()

    @traced(name="cultural_monitor_collect", tags=["layer0", "cultural"])
    def collect_signals(self) -> list[CulturalSignal]:
        """Collect cultural signals with brand safety screening.

        Returns:
            List of CulturalSignal instances with safety status.
        """
        raw_signals = self._fetch_raw_signals()
        screened = self._screen_for_safety(raw_signals)

        logger.info(
            "cultural_signals_collected",
            total=len(raw_signals),
            safe=sum(1 for s in screened if s.is_safe),
            flagged=sum(1 for s in screened if not s.is_safe),
        )
        return screened

    def _fetch_raw_signals(self) -> list[CulturalSignal]:
        """Fetch trending topics from Google Trends RSS feed.

        Parses the RSS XML for trending search items and converts
        each into a CulturalSignal with estimated velocity and relevance.
        """
        signals: list[CulturalSignal] = []

        try:
            resp = self._http.get(_GOOGLE_TRENDS_RSS)
            resp.raise_for_status()
            signals.extend(self._parse_trends_rss(resp.text))
        except httpx.HTTPError as exc:
            logger.warning("google_trends_fetch_failed", error=str(exc))

        return signals

    def _parse_trends_rss(self, xml_text: str) -> list[CulturalSignal]:
        """Parse Google Trends RSS XML into CulturalSignal instances."""
        import xml.etree.ElementTree as ET

        signals: list[CulturalSignal] = []
        try:
            root = ET.fromstring(xml_text)
            for item in root.iter("item"):
                title_el = item.find("title")
                traffic_el = item.find("{https://trends.google.com/trending/rss}approx_traffic")
                if title_el is None or title_el.text is None:
                    continue

                topic = title_el.text.strip()
                # Parse approximate traffic (e.g., "200,000+")
                velocity = 0.0
                if traffic_el is not None and traffic_el.text:
                    traffic_str = traffic_el.text.replace(",", "").replace("+", "")
                    with contextlib.suppress(ValueError):
                        velocity = float(traffic_str) / 1_000_000  # normalize

                signals.append(
                    CulturalSignal(
                        source="google_trends",
                        topic=topic,
                        sentiment=0.0,  # neutral until analyzed
                        velocity=velocity,
                        relevance=0.5,  # default; can be refined with brand matching
                    )
                )
        except ET.ParseError:
            logger.warning("trends_rss_parse_error")

        return signals

    def _screen_for_safety(self, signals: list[CulturalSignal]) -> list[CulturalSignal]:
        """Run brand safety classifier on each signal."""
        for signal in signals:
            if signal.topic:
                is_safe, flags = self._safety.check(signal.topic)
                signal.is_safe = is_safe
                signal.safety_flags = flags
        return signals
