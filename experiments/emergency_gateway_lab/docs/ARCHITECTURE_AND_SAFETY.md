# VeriFYD-owned emergency gateway — future architecture and release gates

## Existing Circle infrastructure

- Browser and PWA: Horizon frontend at `vfvid.com/voice`.
- Control-plane: Render FastAPI and authenticated Trust Voice WebSocket.
- Person-to-person audio/video: WebRTC peer-to-peer transport.
- Group communications: LiveKit room integration.
- Location sharing and Circle Safety: separate opt-in location controls and Trust Circle distress.

## Phase 0 (this branch): isolated lab signaling

Browser/PWA (later) -> **test-only WebSocket** -> lab session manager -> UDP SIP `OPTIONS` -> **loopback mock SIP responder**.

**No INVITE, no RTP, no PSTN, no public SIP, no PSAP, no live coordinates, no actual emergency call.** All test endpoints return explicit labels (`real_emergency_call: false`, `media_connected: false`).

## Phase 1 (future, still test-only)

Existing Circle browser/LiveKit room -> separate LiveKit SIP bridge -> private SIP test PBX -> simulated extension. This adds live audio verification without public telephone routing or a 911 dialable UI. Host SIP/RTP on a UDP-capable machine separate from the Render HTTP web service. Gate all traffic at a firewall. Keep all SIP/PBX routes from allowing public calling.

## Phase 2 (requires external agreements)

Provider-operated authorized emergency interconnection or directly negotiated capability access, per region. Operational requirements to work through with qualified telephony and public-safety specialists:

- Routing to the correct PSAP or designated authority, confirmed location and callback handling; treatment of mobile users with stale location.
- ANI/pANI and dispatchable location provisioning / registered-location updates, applicable ALI or NG911 interfaces.
- Testing coordination with emergency authorities; do not place unscheduled test 911 calls.
- Accessibility for users who cannot speak, dispatch workflows, reliable failover, monitoring, 24/7 emergency availability, incident logs, security/privacy controls.
- Applicable FCC/state/local emergency-calling obligations and clear user disclosures for power, data, and device limitations.

**Until complete:** do not advertise or expose this as E911, do not attach any public trunk, do not permit `911`, `112`, or other emergency digit destinations to pass to SIP. The independent Trust Circle distress system continues as-is; it is not emergency dispatch.

## Cost boundary

Lab phase can run locally with no external telephony charges. Later: SIP-capable compute/networking and emergency interconnect charges, independent of Render WebSocket and application compute. Do not add billable Render services or telephony accounts by default.