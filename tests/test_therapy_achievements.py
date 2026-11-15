"""Contract checks for isolated Therapy & Adaptive Practice rank awards.

The real database must be exercised in staging before enabling this feature.
These tests use stub imports and never open production resources.
"""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path
import pytest
from fastapi import HTTPException

MODULE = Path(__file__).resolve().parents[1] / 'trust_voice_therapy_achievements.py'

@pytest.fixture
def awards(monkeypatch):
    db = types.ModuleType('database')
    db.get_db = lambda: None
    msg = types.ModuleType('trust_voice_messages')
    msg._identity_from_bearer = lambda _: {'id': 'test-only', 'status': 'active'}
    rank = types.ModuleType('trust_voice_ranks')
    rank._schema = lambda: None
    rank.enabled = lambda: True
    monkeypatch.setitem(sys.modules, 'database', db)
    monkeypatch.setitem(sys.modules, 'trust_voice_messages', msg)
    monkeypatch.setitem(sys.modules, 'trust_voice_ranks', rank)
    monkeypatch.setenv('VERIFYD_CIRCLE_THERAPY_RANK_ENABLED', '1')
    spec = importlib.util.spec_from_file_location('_therapy_awards_test_fixture', MODULE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module

def test_points_defined_only_on_server(awards):
    catalog = awards.catalog()
    assert catalog['enabled'] and catalog['once_per_activity_and_level']
    assert catalog['mode'] == 'participation_not_clinical_grade'
    assert len(catalog['activities']) == 9
    assert awards.ACTIVITIES['story']['points']['challenge'] == 50
    assert 'points' not in awards.StartSession.model_fields
    assert 'points' not in awards.CompleteStep.model_fields

def test_complete_award_needs_full_step_receipts(awards):
    good = awards._awards_for(['story', 'story'], ['completed', 'completed'], 2, 'challenge')
    assert len(good) == 1 and good[0]['points'] == 50
    assert not awards._awards_for(['story', 'story'], ['completed', 'skipped'], 2, 'challenge')
    with pytest.raises(ValueError):
        awards._awards_for(['story', 'story'], ['completed'], 2, 'gentle')

def test_never_include_private_therapy_content(awards):
    assert awards.StartSession.model_fields.keys() == {'activity_ids', 'level', 'rounds'}
    assert awards.CompleteStep.model_fields.keys() == {'step_index', 'activity_id', 'action', 'step_token', 'target_index'}
    assert awards.GAZE_TARGETS == (0, 3, 6, 8, 2, 7, 4, 1, 5, 0, 8, 6)

def test_rank_disabled_until_explicit_flag(awards, monkeypatch):
    monkeypatch.delenv('VERIFYD_CIRCLE_THERAPY_RANK_ENABLED')
    assert awards.catalog()['enabled'] is False
    with pytest.raises(HTTPException) as error:
        awards._require()
    assert error.value.status_code == 404

def test_invalid_guided_levels_not_advertised(awards):
    assert 'challenge' not in awards.ACTIVITIES['conversation-practice-v1']['points']
    assert set(awards.ACTIVITIES['gaze-target-v1']['points']) == {'gentle'}
