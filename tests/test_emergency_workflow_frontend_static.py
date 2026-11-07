from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LAB=ROOT/'experiments/emergency_workflow_lab/frontend'


def test_lab_is_dev_and_local_only():
    s=(LAB/'TrustVoiceEmergencyWorkflowLab.jsx').read_text()
    assert 'import.meta.env.DEV' in s
    assert "['localhost', '127.0.0.1']" in s
    assert 'VITE_VERIFYD_EMERGENCY_WORKFLOW_LAB_ENABLED' in s


def test_ui_disclaims_non_delivery():
    s=(LAB/'TrustVoiceEmergencyWorkflowLab.jsx').read_text()
    assert 'NOT 911' in s
    assert 'NO EMERGENCY SERVICES ARE CONTACTED' in s
    assert 'SIMULATED REPLY — NOT 911' in s
    assert 'Do not enter real names, addresses or telephone numbers' in s


def test_ui_does_not_invoke_emergency_or_circle_distress():
    s=(LAB/'TrustVoiceEmergencyWorkflowLab.jsx').read_text()
    for v in ['tel:911','sms:911','CreateSIPParticipant','/trust-voice/safety/distress','rapidsos.com','/calls/911']:
        assert v not in s


def test_ui_requires_active_token():
    s=(LAB/'TrustVoiceEmergencyWorkflowLab.jsx').read_text()
    assert "if (!sessionToken)" in s
    assert 'Authorization: `Bearer ${session.current}`' in s


def test_patch_does_not_change_current_911_or_distress():
    s=(LAB/'TrustVoiceBeta_minimal.patch').read_text()
    added='\n'.join(x[1:] for x in s.splitlines() if x.startswith('+') and not x.startswith('+++'))
    removed='\n'.join(x[1:] for x in s.splitlines() if x.startswith('-') and not x.startswith('---'))
    assert 'TrustVoiceEmergencyWorkflowLab' in added
    assert 'emergencyWorkflowLabEnabled' in added
    assert 'requestEmergency911' not in removed
    assert 'TrustVoiceCircleSafetyOverlay' not in removed
    assert 'placeCall' not in removed


def test_horizons_has_no_lab_on_hosted_site():
    s=(LAB/'TrustVoiceEmergencyWorkflowLab.jsx').read_text()
    assert 'localhost' in s and '127.0.0.1' in s
    assert 'enabled()' in s