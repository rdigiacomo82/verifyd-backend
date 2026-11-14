"""Isolated tests for VeriFYD's occupation-based resource library.

These tests deliberately stub auth and database imports within a fixture only.
They never connect to production and do not alter imports for other test modules.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

MODULE_PATH = Path(__file__).resolve().parents[1] / "trust_voice_therapy_library.py"


@pytest.fixture
def lib(monkeypatch):
    database = types.ModuleType("database")
    database.get_db = lambda: None
    messages = types.ModuleType("trust_voice_messages")
    messages._identity_from_bearer = lambda _: {
        "id": "adult-fixture", "status": "active"
    }
    monkeypatch.setitem(sys.modules, "database", database)
    monkeypatch.setitem(sys.modules, "trust_voice_messages", messages)
    spec = importlib.util.spec_from_file_location("therapy_library_isolated_fixture", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module


def test_disabled_by_default(lib, monkeypatch):
    monkeypatch.delenv("VERIFYD_CIRCLE_THERAPY_LIBRARY_ENABLED", raising=False)
    assert lib.health()["enabled"] is False
    with pytest.raises(HTTPException) as failure:
        lib.profile("Bearer test-session")
    assert failure.value.status_code == 503


def test_no_credential_requirements(lib, monkeypatch):
    monkeypatch.setenv("VERIFYD_CIRCLE_THERAPY_LIBRARY_ENABLED", "1")
    assert "license_number" not in lib.ProfileIn.model_fields
    assert "licensing_board" not in lib.ProfileIn.model_fields
    assert "npi" not in lib.ProfileIn.model_fields
    assert "Speech & Language" in lib.TOPICS
    request = lib.ProfileIn(occupation="Speech & Language",
                            adult_attestation=False, general_materials_only=True)
    with pytest.raises(HTTPException) as failure:
        lib.save_profile(request, "Bearer test-session")
    assert failure.value.status_code == 422


def test_private_details_blocked(lib, monkeypatch):
    monkeypatch.setenv("VERIFYD_CIRCLE_THERAPY_LIBRARY_ENABLED", "1")
    request = lib.ResourceIn(
        title="Sample resource",
        instructions="Patient ID: 919101 and care plan for a named client.",
        topic="Fine Motor", age_band="All ages", difficulty="Introductory",
        format="Activity", rights_and_privacy_confirmed=True,
    )
    with pytest.raises(HTTPException) as failure:
        lib.submit(request, "Bearer test-session")
    assert failure.value.status_code == 422


def test_resource_taxonomy(lib):
    result = lib.taxonomy()
    assert "Physical Therapy" in result["topics"]
    assert "AAC & Communication Boards" in result["topics"]["Speech & Language"]
    assert "Adults (18+)" in result["age_bands"]
    assert "Fine Motor" not in result["topics"]["Speech & Language"]


def test_review_requires_server_allowlist(lib, monkeypatch):
    monkeypatch.delenv("VERIFYD_CIRCLE_THERAPY_LIBRARY_MODERATOR_IDS", raising=False)
    with pytest.raises(HTTPException) as failure:
        lib._moderator("adult-fixture")
    assert failure.value.status_code == 403
    assert lib.health()["private_practice_unlocked"] is False


def test_only_approved_resources_public(lib):
    import inspect
    source = inspect.getsource(lib.catalog)
    assert "WHERE status='approved'" in source
    output = lib._clean({"author_id": "private", "moderated_by": "secret",
                         "resource_id": "sample", "title": "Demo"})
    assert "author_id" not in output
    assert "moderated_by" not in output
    assert output["author_verification"].startswith("Occupation self-reported")


def test_no_clinical_feature_flag_overlap(lib):
    import inspect
    source = inspect.getsource(lib)
    assert "VERIFYD_CIRCLE_THERAPY_PRACTICE_ENABLED" not in source
    assert "VERIFYD_CIRCLE_THERAPY_REGISTRATION_ENABLED" not in source
