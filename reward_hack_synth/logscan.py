"""S5 — surface the generator's own rationalization instead of discarding it.

The pilot's one real data defect was diagnosed after the fact from a log we had
already collected and never read. The generating model noticed the weakness and
talked itself past it:

    "Despite potential corruption, it seems the selected row may still be identified"
    "The ratio of non-ASCII seems manageable at about 23%"

Two markers, findable by regex, sitting in a file on disk. This is not a hard
gate — it is far too noisy for that — but an item whose author argued itself
into keeping it is exactly the item a reviewer should look at first.
"""
from __future__ import annotations

import re

RATIONALIZATION = re.compile(
    r"[^.\n]*\b("
    r"manageable|seems? (?:fine|ok|okay|acceptable|reasonable)|close enough|"
    r"good enough|probably (?:fine|ok|okay|acceptable)|still (?:be )?"
    r"(?:identified|readable|legible|recoverable|usable)|may still|might still|"
    r"despite (?:the |potential |some )?|somewhat (?:weak|marginal|borderline)|"
    r"a bit (?:weak|thin|marginal)|not ideal but|on the fence|borderline|"
    r"arguably (?:fine|ok|enough)|leaning toward keeping|decided to keep"
    r")\b[^.\n]*", re.I)


def scan(text: str, *, limit: int = 12) -> list[str]:
    """Rationalization-shaped sentences in a generation log."""
    seen: list[str] = []
    for m in RATIONALIZATION.finditer(text or ""):
        s = " ".join(m.group(0).split())
        if len(s) > 12 and s not in seen:
            seen.append(s)
        if len(seen) >= limit:
            break
    return seen


def scan_shard(log: dict) -> list[str]:
    blob = " ".join(str(log.get(k, "") or "") for k in ("stdout_tail", "stderr_tail"))
    return scan(blob)
