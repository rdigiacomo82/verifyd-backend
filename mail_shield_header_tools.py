import re


_EMAIL_RE = re.compile(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", re.IGNORECASE)


def extract_domain_from_email(email: str) -> str:
    if not email:
        return ""
    match = _EMAIL_RE.search(email)
    raw = match.group(1) if match else email
    if "@" not in raw:
        return ""
    domain = raw.rsplit("@", 1)[-1].strip().lower()
    if domain.startswith("[") and domain.endswith("]"):
        return ""
    return domain.strip(".")


def normalize_headers(headers: dict | None) -> dict:
    if not headers:
        return {}
    return {str(k).strip().lower(): str(v or "") for k, v in headers.items()}


def detect_authentication_risk(headers: dict | None) -> tuple[int, str, list[str]]:
    h = normalize_headers(headers)
    auth = h.get("authentication-results", "")
    received_spf = h.get("received-spf", "")
    combined = f"{auth} {received_spf}".lower()

    penalty = 0
    reasons = []

    if not auth and not received_spf:
        penalty += 8
        reasons.append("Email authentication headers were not available")
        return penalty, "REVIEW", reasons

    if "spf=fail" in combined or "spf fail" in combined:
        penalty += 20
        reasons.append("SPF authentication failed")
    if "dkim=fail" in combined or "dkim fail" in combined:
        penalty += 20
        reasons.append("DKIM authentication failed")
    if "dmarc=fail" in combined or "dmarc fail" in combined:
        penalty += 25
        reasons.append("DMARC authentication failed")

    if penalty >= 25:
        return penalty, "HIGH", reasons
    if penalty > 0:
        return penalty, "REVIEW", reasons
    return penalty, "LOW", reasons


def looks_like_brand_spoof(from_name: str, from_domain: str) -> bool:
    name = (from_name or "").lower()
    domain = (from_domain or "").lower()

    brands = {
        "microsoft": ["microsoft.com", "office.com", "outlook.com"],
        "apple": ["apple.com"],
        "google": ["google.com"],
        "paypal": ["paypal.com"],
        "docusign": ["docusign.com"],
        "adobe": ["adobe.com"],
        "amazon": ["amazon.com"],
        "chase": ["chase.com"],
        "bank of america": ["bankofamerica.com"],
        "wells fargo": ["wellsfargo.com"],
        "state farm": ["statefarm.com"],
        "allstate": ["allstate.com"],
    }

    for brand, allowed_domains in brands.items():
        if brand in name and not any(domain == d or domain.endswith("." + d) for d in allowed_domains):
            return True
    return False


def has_lookalike_domain(domain: str) -> bool:
    if not domain:
        return False
    d = domain.lower()
    suspicious_tokens = ["0", "1", "rn", "vv", "secure-", "-secure", "login-", "-login", "verify-", "-verify"]
    known_typos = ["micros0ft", "paypa1", "g00gle", "arnazon", "docuslgn", "app1e"]
    return any(t in d for t in known_typos) or any(t in d for t in suspicious_tokens)