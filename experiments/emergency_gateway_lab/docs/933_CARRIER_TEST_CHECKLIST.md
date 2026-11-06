# VeriFYD Circle — 933 E911 carrier test plan

**This document is a checklist, not evidence of an operational E911 connection.**

## 1. Distinguish the two tests

- **Current lab:** `simulate-933` returns invented placeholders with explicit `carrier_connected: false`. No SIP INVITE, RTP, carrier trunk or emergency service is used.
- **Future live 933:** An authorized emergency-calling provider must offer 933 readback and accept traffic from our provisioned test endpoint. The provider—not our lab—delivers the test response.

## 2. Before any real 933 call

1. Obtain an authorized emergency calling interconnection/trunk and confirm **in writing** that the provider supports `933` for the test endpoint, including any restrictions or charges.
2. Verify test caller/callback identifier, valid US/Canada registered emergency civic address, and active E911 service. For Telnyx, the outbound caller ID needs an E911-enabled number in `+E.164` format. For Bandwidth Dynamic Location Routing, correct endpoint identity and supported location mechanism are required.
3. Configure a restricted outbound dial plan that allows **only 933 on the designated test trunk**; explicitly deny real 911, other short codes, general PSTN, and arbitrary SIP URI destinations. This is a **future nonproduction component**, not included in lab v0.2.
4. Use a SIP/RTP-capable test host and test-only credentials. Configure correct SIP identity headers and device/location mapping.
5. Verify that no public test control is accessible without strong authentication and audit logging; do not store unnecessary personal location data.
6. Confirm a documented window and call rate with the provider. Bandwidth advises limiting 933 calls to **one per minute**.

## 3. Live 933 acceptance checklist (future only)

- [ ] Test SIP INVITE reaches authorized emergency-provider SBC and completes correctly.
- [ ] Receive and hear the provider's automated 933 readback over bidirectional audio.
- [ ] Verify readback callback number exactly matches the provisioned test callback.
- [ ] Verify readback civic address exactly or suitably matches registered endpoint location.
- [ ] For a moving/nomadic device, verify approved dynamic-location method and mapping for that endpoint.
- [ ] Verify failure cases: missing location, missing callback, invalid identity, expired authorization, unsupported device, SIP/RTP failure.
- [ ] Verify no route to 911 or unrelated number exists through the test-only dial plan.
- [ ] Verify outbound logs are redacted and accessible only to authorized staff.

A successful 933 readback is an **interconnection/location smoke test**, **not** proof of correct PSAP dispatch, resilient emergency call handling, or compliance with all 911 rules. Consult emergency-service partners before public release.

## Provider references

- Telnyx: https://support.telnyx.com/en/articles/1130709-how-do-i-test-e911-service
- Bandwidth: https://www.bandwidth.com/support/en/articles/12823286-the-933-service
- Bandwidth DLR: https://www.bandwidth.com/support/en/articles/12822952-e911-dynamic-location-routing-integration-and-testing-guide