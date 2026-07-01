# ============================================================
#  VeriFYD — c2pa_checker.py
#
#  Phase 3A: VeriFYD Content Credentials Plus Report
#
#  Safe design:
#    - This module is informational by default.
#    - It does not make the final authenticity/tamper decision.
#    - It tries to read C2PA / Content Credentials through available tools.
#    - If no tool/manifest is available, it returns a clear user-safe report.
#
#  Public helpers:
#    analyze_content_credentials(path, filename="", media_type="") -> dict
#    check_c2pa(path) -> tuple[int, str, dict]   # compatibility for detection.py
# ============================================================

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Tuple

log = logging.getLogger("verifyd.c2pa")


AI_GENERATOR_TERMS = (
    "openai", "sora", "dall-e", "dalle", "chatgpt",
    "midjourney", "stable diffusion", "stability ai", "runway",
    "pika", "kling", "ideogram", "leonardo", "firefly",
    "adobe firefly", "gemini", "imagen", "canva", "copilot",
)

CAMERA_TERMS = (
    "canon", "nikon", "sony", "fujifilm", "fuji", "leica",
    "panasonic", "olympus", "om system", "gopro", "dji",
    "apple", "iphone", "samsung", "google pixel", "pixel",
)


def _blank_report(
    *,
    status: str = "NO_CONTENT_CREDENTIALS_FOUND",
    filename: str = "",
    media_type: str = "",
    tool_available: bool = False,
    tool_used: str = "",
    technical_note: str = "",
) -> Dict[str, Any]:
    interpretation = (
        "No C2PA / Content Credentials provenance data was found. "
        "This does not automatically mean the file is fake; it means VeriFYD could not confirm "
        "origin or edit history from embedded provenance data. VeriFYD still analyzes the file "
        "using forensic, AI, metadata, hash, and file-structure signals."
    )
    if status == "CONTENT_CREDENTIALS_UNAVAILABLE":
        interpretation = (
            "Content Credentials inspection was not available in this runtime. "
            "This does not affect VeriFYD's forensic, AI, metadata, hash, and file-structure analysis."
        )
    return {
        "title": "VeriFYD Content Credentials Plus Report",
        "status": status,
        "manifest_found": False,
        "manifest_valid": "unknown",
        "claim_generator": "",
        "reported_actions": [],
        "ingredients": [],
        "interpretation": interpretation,
        "risk_level": "informational",
        "filename": filename or "",
        "media_type": media_type or "",
        "tool_available": bool(tool_available),
        "tool_used": tool_used or "",
        "technical_notes": [technical_note] if technical_note else [],
    }


def _safe_text(value: Any, limit: int = 300) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:limit]


def _walk_values(obj: Any, keys: tuple[str, ...], limit: int = 30) -> List[str]:
    """Collect short string values for any dict keys matching the supplied names."""
    found: List[str] = []
    lowered = {k.lower() for k in keys}

    def walk(x: Any) -> None:
        if len(found) >= limit:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                if kl in lowered:
                    if isinstance(v, (str, int, float, bool)):
                        t = _safe_text(v)
                        if t and t not in found:
                            found.append(t)
                    elif isinstance(v, list):
                        for item in v[:10]:
                            if isinstance(item, (str, int, float, bool)):
                                t = _safe_text(item)
                                if t and t not in found:
                                    found.append(t)
                walk(v)
        elif isinstance(x, list):
            for item in x[:80]:
                walk(item)

    walk(obj)
    return found[:limit]


def _walk_action_names(obj: Any, limit: int = 20) -> List[str]:
    actions: List[str] = []

    def add(value: Any) -> None:
        text = _safe_text(value, 120)
        if text and text not in actions:
            actions.append(text)

    def walk(x: Any) -> None:
        if len(actions) >= limit:
            return
        if isinstance(x, dict):
            # C2PA action assertions often include fields like action, label, name, actionName.
            for key in ("action", "label", "name", "actionName"):
                if key in x:
                    add(x.get(key))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for item in x[:100]:
                walk(item)

    walk(obj)
    return actions[:limit]


def _walk_ingredients(obj: Any, limit: int = 20) -> List[str]:
    ingredients: List[str] = []

    def add(value: Any) -> None:
        text = _safe_text(value, 160)
        if text and text not in ingredients:
            ingredients.append(text)

    def walk(x: Any) -> None:
        if len(ingredients) >= limit:
            return
        if isinstance(x, dict):
            key_blob = " ".join(str(k).lower() for k in x.keys())
            if "ingredient" in key_blob or "thumbnail" in key_blob or "relationship" in key_blob:
                for key in ("title", "filename", "format", "documentID", "instanceID", "relationship"):
                    if key in x:
                        add(x.get(key))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for item in x[:100]:
                walk(item)

    walk(obj)
    return ingredients[:limit]


def _extract_claim_generator(parsed: Any) -> str:
    candidates = _walk_values(
        parsed,
        (
            "claim_generator", "claimGenerator", "generator", "softwareAgent",
            "software_agent", "name", "issuer", "producer", "creator",
        ),
        limit=40,
    )
    # Prefer values that look like software/tool names.
    for c in candidates:
        low = c.lower()
        if any(term in low for term in AI_GENERATOR_TERMS + CAMERA_TERMS + ("adobe", "photoshop", "lightroom", "content credentials")):
            return c
    return candidates[0] if candidates else ""


def _classify_generator(generator: str, actions: List[str]) -> tuple[str, str]:
    blob = " ".join([generator or ""] + list(actions or [])).lower()
    if any(term in blob for term in AI_GENERATOR_TERMS):
        return "AI_OR_GENERATIVE_TOOL_CLAIMED", "medium"
    if any(term in blob for term in CAMERA_TERMS):
        return "CAMERA_OR_DEVICE_PROVENANCE_CLAIMED", "low"
    if "adobe" in blob or "photoshop" in blob or "lightroom" in blob:
        return "EDITING_SOFTWARE_PROVENANCE_CLAIMED", "informational"
    return "CONTENT_CREDENTIALS_FOUND", "informational"


def _interpret(status: str, generator: str, actions: List[str], ingredients: List[str], valid: str) -> str:
    if status == "AI_OR_GENERATIVE_TOOL_CLAIMED":
        return (
            "Content Credentials were found and appear to reference an AI or generative-media workflow. "
            "VeriFYD treats this as provenance information and still cross-checks it against forensic, "
            "AI, metadata, hash, and file-structure signals."
        )
    if status == "CAMERA_OR_DEVICE_PROVENANCE_CLAIMED":
        return (
            "Content Credentials were found and appear to reference camera or device provenance. "
            "This can support authenticity, but VeriFYD still cross-checks the file itself because "
            "provenance data is only one verification signal."
        )
    if status == "EDITING_SOFTWARE_PROVENANCE_CLAIMED":
        return (
            "Content Credentials were found and appear to reference editing or export software. "
            "This may indicate a normal editing/export workflow rather than an untouched camera-original."
        )
    if valid == "invalid":
        return (
            "Content Credentials were found, but VeriFYD could not validate them cleanly. "
            "Treat the provenance claim as questionable and rely on the broader VeriFYD forensic analysis."
        )
    return (
        "Content Credentials were found. VeriFYD reports the embedded provenance claims and cross-checks "
        "them against forensic, AI, metadata, hash, and file-structure signals."
    )


def _report_from_parsed_manifest(parsed: Any, *, filename: str = "", media_type: str = "", tool_used: str = "") -> Dict[str, Any]:
    generator = _extract_claim_generator(parsed)
    actions = _walk_action_names(parsed)
    ingredients = _walk_ingredients(parsed)

    raw_text = json.dumps(parsed, ensure_ascii=False, default=str)[:300000].lower()
    manifest_valid = "valid"
    if any(x in raw_text for x in ("invalid signature", "signature invalid", "validation error", "claim signature invalid")):
        manifest_valid = "invalid"
    elif any(x in raw_text for x in ("validation_status", "validationStatus", "signature")):
        manifest_valid = "valid"

    status, risk = _classify_generator(generator, actions)
    if manifest_valid == "invalid":
        status = "CONTENT_CREDENTIALS_INVALID"
        risk = "medium"

    technical_notes = []
    if tool_used:
        technical_notes.append(f"Inspected with {tool_used}.")
    if generator:
        technical_notes.append(f"Claim generator: {generator}.")
    if actions:
        technical_notes.append(f"Reported action count: {len(actions)}.")
    if ingredients:
        technical_notes.append(f"Ingredient/source asset count: {len(ingredients)}.")

    return {
        "title": "VeriFYD Content Credentials Plus Report",
        "status": status,
        "manifest_found": True,
        "manifest_valid": manifest_valid,
        "claim_generator": generator,
        "reported_actions": actions[:20],
        "ingredients": ingredients[:20],
        "interpretation": _interpret(status, generator, actions, ingredients, manifest_valid),
        "risk_level": risk,
        "filename": filename or "",
        "media_type": media_type or "",
        "tool_available": True,
        "tool_used": tool_used or "content_credentials_reader",
        "technical_notes": technical_notes,
    }


def _try_c2patool(path: str) -> tuple[bool, Dict[str, Any], str]:
    exe = shutil.which("c2patool")
    if not exe:
        return False, {}, "c2patool_not_found"

    commands = [
        [exe, path, "--json"],
        [exe, "--json", path],
    ]
    last_error = ""
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            if proc.returncode != 0:
                last_error = err[:300] or f"returncode={proc.returncode}"
                continue
            if not output:
                last_error = "empty_output"
                continue
            try:
                return True, json.loads(output), ""
            except Exception as parse_exc:
                # Some builds print explanatory text before JSON; try to isolate it.
                start = output.find("{")
                end = output.rfind("}")
                if start >= 0 and end > start:
                    try:
                        return True, json.loads(output[start:end + 1]), ""
                    except Exception:
                        pass
                last_error = f"json_parse_failed:{parse_exc}"
        except Exception as exc:
            last_error = str(exc)[:300]

    return True, {}, last_error or "c2patool_failed"


def _try_c2pa_python(path: str) -> tuple[bool, Dict[str, Any], str]:
    """
    Best-effort c2pa-python support.

    c2pa-python APIs have changed across versions, so this uses cautious dynamic
    inspection. If the installed package exposes a reader function, we use it;
    otherwise the report gracefully falls back.
    """
    try:
        import c2pa  # type: ignore
    except Exception as exc:
        return False, {}, f"c2pa_python_not_importable:{exc}"

    candidates = []
    for name in ("read_file", "read", "Reader", "C2paReader"):
        if hasattr(c2pa, name):
            candidates.append((name, getattr(c2pa, name)))

    for name, obj in candidates:
        try:
            if name in ("Reader", "C2paReader"):
                reader = obj(path)
                for meth_name in ("json", "to_json", "manifest_store", "read", "get_json"):
                    if hasattr(reader, meth_name):
                        value = getattr(reader, meth_name)()
                        if isinstance(value, str):
                            return True, json.loads(value), ""
                        if isinstance(value, dict):
                            return True, value, ""
            else:
                value = obj(path)
                if isinstance(value, str):
                    return True, json.loads(value), ""
                if isinstance(value, dict):
                    return True, value, ""
        except Exception as exc:
            log.debug("c2pa-python candidate %s failed: %s", name, exc)
            continue

    return True, {}, "c2pa_python_no_supported_reader_api"


def analyze_content_credentials(path: str, filename: str = "", media_type: str = "") -> Dict[str, Any]:
    """Return a VeriFYD Content Credentials Plus Report for one file."""
    if not path or not os.path.exists(path):
        return _blank_report(
            status="CONTENT_CREDENTIALS_UNAVAILABLE",
            filename=filename,
            media_type=media_type,
            technical_note="File path was not available for Content Credentials inspection.",
        )

    # Try CLI first because c2patool produces the most complete manifest JSON when present.
    cli_available, cli_json, cli_error = _try_c2patool(path)
    if cli_json:
        return _report_from_parsed_manifest(cli_json, filename=filename, media_type=media_type, tool_used="c2patool")

    py_available, py_json, py_error = _try_c2pa_python(path)
    if py_json:
        return _report_from_parsed_manifest(py_json, filename=filename, media_type=media_type, tool_used="c2pa-python")

    tool_available = bool(cli_available or py_available)
    technical_note = ""
    # If a tool ran and reported no manifest, keep the user-facing status as no credentials found.
    if tool_available:
        joined = " | ".join(x for x in (cli_error, py_error) if x)
        if any(x in joined.lower() for x in ("no manifest", "manifest not found", "no claim", "not found", "unsupported")):
            technical_note = "Content Credentials tool ran, but no manifest was found."
        else:
            technical_note = f"Content Credentials inspection did not return a readable manifest. {joined[:240]}".strip()
    else:
        technical_note = "No Content Credentials inspection tool was available in this runtime."

    # Keep this as NO_CONTENT_CREDENTIALS_FOUND for the clean MVP card. The technical note
    # gives debugging detail without alarming customers.
    return _blank_report(
        status="NO_CONTENT_CREDENTIALS_FOUND",
        filename=filename,
        media_type=media_type,
        tool_available=tool_available,
        tool_used="c2patool/c2pa-python" if tool_available else "",
        technical_note=technical_note,
    )


def check_c2pa(path: str) -> Tuple[int, str, Dict[str, Any]]:
    """
    Compatibility hook for detection.py.

    For Phase 3A, Content Credentials are reported as provenance context but do
    not independently override the VeriFYD authenticity decision. A future phase
    can safely adjust this if you want C2PA claims to affect scoring.
    """
    report = analyze_content_credentials(path)
    status = str(report.get("status", "NO_CONTENT_CREDENTIALS_FOUND"))
    # Intentionally neutral in Phase 3A.
    return 0, status, report
