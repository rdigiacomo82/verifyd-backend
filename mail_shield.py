import logging

from fastapi import APIRouter

from mail_shield_models import MailShieldScanRequest, MailShieldScanResponse, MailShieldResult
from mail_shield_scoring import score_mail_shield
from mail_shield_header_tools import extract_domain_from_email

log = logging.getLogger("verifyd.mail_shield")

router = APIRouter(prefix="/mail-shield", tags=["Mail Shield"])


@router.post("/scan", response_model=MailShieldScanResponse)
async def scan_mail_shield(payload: MailShieldScanRequest):
    """
    Lightweight VeriFYD Mail Shield MVP.

    This endpoint is intentionally isolated from the existing video/photo/audio/document
    authenticity pipeline. It performs metadata/header/link/content/attachment-name
    risk scoring only. It does not download attachments, create certificates, write
    to Trust Desk, or call existing AI detection unless a later deep-scan phase adds
    that explicitly.
    """
    result = score_mail_shield(payload)

    # Do not log email body or sensitive content.
    sender_domain = extract_domain_from_email(payload.from_email or "")
    log.info(
        "Mail Shield scan: email_id=%s sender_domain=%s color=%s score=%s",
        payload.email_id or "",
        sender_domain,
        result.get("color"),
        result.get("trust_score"),
    )

    return MailShieldScanResponse(
        status="ok",
        mail_shield=MailShieldResult(**result),
    )