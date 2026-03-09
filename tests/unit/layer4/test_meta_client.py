"""Tests for Meta client."""

from __future__ import annotations

from brand_conscience.layer4_deployment.meta_client import MetaClient


def test_meta_client_init():
    client = MetaClient(access_token="test-token", ad_account_id="12345")
    assert client._token == "test-token"
    assert client._account_id == "12345"
