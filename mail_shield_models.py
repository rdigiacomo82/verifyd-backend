from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class MailShieldAttachment(BaseModel):
    filename: str = ""
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None


class MailShieldScanRequest(BaseModel):
    email_id: Optional[str] = None
    from_email: Optional[str] = None
    from_name: Optional[str] = None
    reply_to: Optional[str] = None
    subject: Optional[str] = None
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    headers: Dict[str, Any] = Field(default_factory=dict)
    links: List[str] = Field(default_factory=list)
    attachments: List[MailShieldAttachment] = Field(default_factory=list)
    deep_scan: bool = False


class MailShieldResult(BaseModel):
    trust_score: int
    risk_level: str
    color: str
    label: str
    recommended_action: str
    signals: Dict[str, str]
    reasons: List[str]
    verify_further_available: bool = True


class MailShieldScanResponse(BaseModel):
    status: str = "ok"
    mail_shield: MailShieldResult