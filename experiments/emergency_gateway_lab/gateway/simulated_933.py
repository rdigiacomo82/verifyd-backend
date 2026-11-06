"""933 E911 readback *SIMULATOR*, NOT a carrier connection.

933 is a service provided by participating emergency-service carriers. This
module has no phone number, SIP trunk, external route, location or audio. It
produces synthetic diagnostics for UI/contract tests only.
"""
from __future__ import annotations

from typing import Literal

Scenario = Literal['provisioned', 'missing_callback', 'missing_address', 'unregistered']
FAKE_ENDPOINT = 'VFVID-SYNTHETIC-ENDPOINT'
FAKE_CALLBACK = 'SIMULATED-CALLBACK-NOT-A-TELEPHONE-NUMBER'
FAKE_LOCATION = 'SIMULATED-LAB-ADDRESS-NOT-DISPATCHABLE'


def simulate_933_readback(scenario: Scenario = 'provisioned') -> dict:
    if scenario not in ('provisioned', 'missing_callback', 'missing_address', 'unregistered'):
        raise ValueError('Invalid synthetic scenario')
    outcomes = {
        'provisioned': ('simulated_ready', 'Both synthetic callback and synthetic location present'),
        'missing_callback': ('simulated_failure', 'Synthetic callback information missing'),
        'missing_address': ('simulated_failure', 'Synthetic emergency address information missing'),
        'unregistered': ('simulated_failure', 'Synthetic endpoint is not registered'),
    }
    status, explanation = outcomes[scenario]
    return {
        'test_number_label': '933',
        'mode': 'OFFLINE_SIMULATED_933',
        'scenario': scenario,
        'status': status,
        'readback': {
            'endpoint': FAKE_ENDPOINT if scenario != 'unregistered' else 'UNREGISTERED-TEST-ENDPOINT',
            'callback': FAKE_CALLBACK if scenario in ('provisioned', 'missing_address') else None,
            'address': FAKE_LOCATION if scenario in ('provisioned', 'missing_callback') else None,
        },
        'explanation': explanation,
        'carrier_connected': False,
        'real_933_call_placed': False,
        'real_911_call_placed': False,
        'dispatch_center_contacted': False,
        'real_emergency_service_validated': False,
        'audio_connected': False,
        'warning': 'Synthetic lab response only. This does NOT establish E911 readiness or validate emergency routing.',
    }