"""Explainable, conservative voicemail risk labels. No claim of authenticated speech.

This deliberately reuses VeriFYD Mail Shield's content-risk lexicon while
keeping audio provenance independent of transcript-based scam evidence.
"""
from __future__ import annotations

import re
from typing import Any

from mail_shield_scoring import _content_penalty

SEVERE_PATTERNS = {
    "Payment requested through gift cards": r"\bgift\s*cards?\b|\bsteam\s*cards?\b",
    "Request for a password or verification code": r"\b(password|passcode|one[- ]time code|otp|verification code)\b",
    "Request to install remote-access software": r"\b(anydesk|teamviewer|remote desktop|screen sharing app)\b",
    "Pressure to send cryptocurrency": r"\b(bitcoin|crypto(?:currency)?|wallet address)\b",
    "Threat of arrest or legal consequences": r"\b(warrant for your arrest|you will be arrested|police will arrest|lawsuit today)\b",
}
REVIEW_PATTERNS = {
    "Urgent callback or immediate action requested": r"\b(call back immediately|immediate action|act now|urgent response|last chance)\b",
    "Banking or payment information discussed": r"\b(bank account|routing number|wire transfer|payment details|credit card)\b",
    "Identity information requested": r"\b(social security number|social insurance number|date of birth|confirm your identity)\b",
    "Unsolicited link or website instructions": r"\b(visit (?:our|the|this) website|click (?:the|this) link|open this link)\b",
}
MAX_TRANSCRIPT = 12000


def _matches(transcript: str, patterns: dict[str, str]) -> list[str]:
    return [reason for reason, pattern in patterns.items() if re.search(pattern, transcript, re.I)]


def review_voicemail_content(transcript: str | None, audio: dict[str, Any] | None) -> dict[str, Any]:
    """Risk labels are triage, not identity authentication or a guarantee of safety."""
    transcript = (transcript or "").strip()[:MAX_TRANSCRIPT]
    audio = audio if isinstance(audio, dict) else {}
    available = audio.get("available") is True and audio.get("has_audio", True) is not False
    try:
        score = max(0, min(100, round(float(audio["audio_ai_score"])))) if available else None
    except (TypeError, ValueError, KeyError):
        score = None
    confidence = str(audio.get("confidence") or "low").lower()
    sound_evidence = [
        str(x)[:190] for x in audio.get("evidence", [])[:4]
        if isinstance(x, str)
    ] if available else []

    severe = _matches(transcript, SEVERE_PATTERNS)
    review = _matches(transcript, REVIEW_PATTERNS)
    mail_penalty, _, _ = _content_penalty("", transcript) if transcript else (0, "LOW", [])
    reason_codes: list[str] = []
    reasons: list[str] = []

    if severe:
        reason_codes.append("scam_language")
        reasons.extend(severe[:3])
    if review or mail_penalty >= 15:
        reason_codes.append("needs_verification")
        reasons.extend(review[:3] or ["Payment, account or urgency language warrants independent verification"])

    ai_suspected = (
        score is not None and score >= 75
        and confidence not in {"low", "no_audio", "unavailable", "unknown"}
    )
    if ai_suspected:
        reason_codes.append("possible_synthetic_voice")
        reasons.append("Audio analysis found possible synthetic-voice indicators; this is not proof of AI generation.")

    # Fraud risk is based on suspicious *behavior* in the message, not simply
    # the caller's voice. AI alone should never be a red fraud verdict.
    if len(severe) >= 2 or (severe and (review or mail_penalty >= 15)):
        level = "red"
        title = "Potential scam or harmful request"
    elif severe or review or mail_penalty >= 15 or ai_suspected:
        level = "blue"
        title = "Review before responding"
    elif not transcript:
        level = "blue"
        title = "Unable to assess spoken message"
        reasons.append("Transcript was unavailable; no claim about the message's safety can be made.")
    else:
        level = "green"
        title = "No obvious warning signs found"
        reasons.append("No obvious scam language detected. This does not verify the caller or guarantee safety.")

    # Detection failure is not a negative result, and a short/simple transcript
    # does not permit a confident green label.
    if level == "green" and len(transcript.split()) < 5:
        level = "blue"
        title = "Not enough speech to assess"
        reasons = ["Very short transcription; treat as unverified."]

    return {
        "level": level,
        "title": title,
        "explanations": list(dict.fromkeys(reasons))[:5],
        "risk_codes": list(dict.fromkeys(reason_codes)),
        "synthetic_voice": {
            "assessment": "possible" if ai_suspected else ("not_established" if available else "unavailable"),
            "heuristic_score": score,
            "confidence": confidence if available else "unavailable",
            "indicators": sound_evidence,
            "disclaimer": "AI-voice detection is probabilistic and can be wrong.",
        },
        "scam_language": {
            "assessment": "flagged" if severe or review or mail_penalty >= 15 else (
                "not_assessed" if not transcript else "no_obvious_indicators"
            ),
        },
        "disclaimer": "This advisory is not proof of caller identity, AI generation, fraud, or safety.",
    }
