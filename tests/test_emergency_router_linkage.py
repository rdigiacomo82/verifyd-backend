"""Repository-level static gates to catch dangerous accidental coupling."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_main_import_and_mount_are_additive():
    main=(ROOT/'main.py').read_text()
    assert 'from trust_voice_emergency_enrollment import router as emergency_enrollment_router' in main
    assert 'app.include_router(emergency_enrollment_router)' in main
    ast.parse(main)

def test_provider_911_dial_not_in_staging_router():
    src=(ROOT/'trust_voice_emergency_enrollment.py').read_text()
    ast.parse(src)
    for forbidden in ('CreateSIPParticipant','sip:911','tel:911','TWILIO_AUTH_TOKEN','TELNYX_API_KEY','requests.post(', 'httpx.post(', 'websocket.send('):
        assert forbidden not in src

def test_readiness_is_always_false():
    import importlib.util,sys
    spec=importlib.util.spec_from_file_location('test_readiness_e911',ROOT/'trust_voice_emergency_enrollment.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    for name in ('can_place_911_call','provider_connected','mobile_sms_verified','address_provider_validated','callback_number_provisioned','emergency_routing_validated'):
        assert module._not_e911()[name] is False