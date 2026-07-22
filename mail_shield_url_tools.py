import html
import re
from urllib.parse import urlparse


SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "rebrand.ly", "ow.ly",
    "is.gd", "buff.ly", "cutt.ly", "shorturl.at", "lnkd.in", "bitly.com"
}

SUSPICIOUS_TLDS = {
    "zip", "mov", "click", "top", "xyz", "icu", "cam", "country",
    "stream", "gq", "tk", "ml", "cf"
}


_URL_RE = re.compile(r"https?://[^\s<>'\"()]+", re.IGNORECASE)


def normalize_domain(value: str) -> str:
    if not value:
        return ""
    value = value.strip().lower()
    if "://" not in value:
        value = "http://" + value
    try:
        host = urlparse(value).netloc.lower()
    except Exception:
        return ""
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host.strip(".")


def extract_urls(body_text: str = "", body_html: str = "") -> list[str]:
    text = body_text or ""
    if body_html:
        text += "\n" + html.unescape(re.sub(r"<[^>]+>", " ", body_html))
        # Also catch href="..."
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', body_html, flags=re.IGNORECASE)
        text += "\n" + "\n".join(hrefs)
    found = _URL_RE.findall(text)
    cleaned = []
    seen = set()
    for url in found:
        url = url.rstrip(".,;:!?)\"]}'")
        if url not in seen:
            seen.add(url)
            cleaned.append(url)
    return cleaned


def is_shortened_url(url: str) -> bool:
    return normalize_domain(url) in SHORTENER_DOMAINS


def is_punycode_or_suspicious_tld(url: str) -> bool:
    host = normalize_domain(url)
    if not host:
        return False
    if "xn--" in host:
        return True
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    return tld in SUSPICIOUS_TLDS


def domain_distance_hint(sender_domain: str, url_domain: str) -> bool:
    if not sender_domain or not url_domain:
        return False
    if sender_domain == url_domain or url_domain.endswith("." + sender_domain):
        return False
    sender_root = sender_domain.split(".")[-2] if "." in sender_domain else sender_domain
    url_root = url_domain.split(".")[-2] if "." in url_domain else url_domain
    return sender_root not in url_root and url_root not in sender_root