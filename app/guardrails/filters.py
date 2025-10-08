# app/guardrails/filters.py
import re

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), # xxx-xx-xxxx # Social Security Number
    re.compile(r"\b\d{3}-\d{3}-\d{4}\b"), # xxx-xxx-xxxx # Phone Number
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), # email address
]
PROFANITY_WORDS = {"dagnabbit","darn","heck","shoot","sugarplums","frickin"}
PROFANITY_PATTERNS = [re.compile(rf"\b{w}\b", re.IGNORECASE) for w in PROFANITY_WORDS]

def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()

def redact_pii(text: str) -> str:
    for p in PII_PATTERNS:
        text = p.sub("REDACTED", text)
    return text

def contains_profanity(text: str) -> bool:
    return any(p.search(text) for p in PROFANITY_PATTERNS)
