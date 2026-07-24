from __future__ import annotations

from app.api import _authorized_api_token
from app.config import Settings


def test_api_token_setting_defaults_to_disabled():
    assert Settings(_env_file=None).acacite_api_token is None


def test_authorized_api_token_accepts_bearer_header():
    assert _authorized_api_token({"Authorization": "Bearer expected"}, "expected")


def test_authorized_api_token_accepts_custom_header():
    assert _authorized_api_token({"X-AcaCite-Token": "expected"}, "expected")


def test_authorized_api_token_rejects_missing_or_wrong_value():
    assert not _authorized_api_token({}, "expected")
    assert not _authorized_api_token({"Authorization": "Bearer wrong"}, "expected")
