"""Privacy and anonymization guardrails for dental imaging."""
from __future__ import annotations
import re
from typing import Any

# Patterns for potential patient identifying information (PII/PHI)
PHI_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(mrn|id|patient)[\s:#_-]*([a-zA-Z0-9]{4,16})\b", re.IGNORECASE),  # Medical Record Numbers
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),  # Dates of birth / exam dates
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
    re.compile(r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b"),  # Phone
]

def sanitize_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitizes text and dictionary values to ensure no patient identifying data is sent to external APIs."""
    sanitized: dict[str, Any] = {}
    for k, v in data.items():
        # Disallow raw pixel buffers or bytes from being transmitted
        if k in ("image_bytes", "raw_pixels", "buffer", "array") or isinstance(v, (bytes, bytearray)):
            continue
        if isinstance(v, str):
            clean_str = v
            for pattern in PHI_PATTERNS:
                clean_str = pattern.sub("[REDACTED_PHI]", clean_str)
            sanitized[k] = clean_str
        elif isinstance(v, dict):
            sanitized[k] = sanitize_metadata(v)
        elif isinstance(v, list):
            sanitized[k] = [
                sanitize_metadata(item) if isinstance(item, dict)
                else (pattern_clean(item) if isinstance(item, str) else item)
                for item in v
            ]
        else:
            sanitized[k] = v
    return sanitized

def pattern_clean(val: str) -> str:
    res = val
    for pattern in PHI_PATTERNS:
        res = pattern.sub("[REDACTED_PHI]", res)
    return res

def check_image_transmission_allowed() -> bool:
    """Never upload raw patient dental radiographs to external LLM APIs unless explicitly authorized."""
    return False
