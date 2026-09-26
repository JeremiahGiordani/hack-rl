"""Real-benchmark substrate for the hack-layer datasets.

Nothing here is synthetic: questions and their labels come from the source
benchmarks unmodified, per the dataset spec ("synthetic generation should build
the *hack layer*, not invent facts that need to be independently correct").

Pulled over the HF datasets-server REST API rather than the ``datasets``
library — same data, no multi-hundred-MB dependency in the container.
"""
from __future__ import annotations

import json
import random
import re
import urllib.parse
import urllib.request
from typing import Any

DS_SERVER = "https://datasets-server.huggingface.co/rows"

# benchmark key -> (hf dataset, config, split, answer format)
SOURCES: dict[str, tuple[str, str, str, str]] = {
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "default", "test", "mc"),
    "gsm8k": ("openai/gsm8k", "main", "test", "numeric"),
    "trivia_qa": ("mandarjoshi/trivia_qa", "rc.nocontext", "validation", "short_answer"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "test", "mc"),
    "race": ("ehovy/race", "high", "test", "mc_passage"),
    "truthful_qa": ("truthfulqa/truthful_qa", "generation", "validation", "misconception"),
}

LETTERS = "ABCDEFGHIJ"


def _get_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    url = (
        f"{DS_SERVER}?dataset={urllib.parse.quote(dataset)}"
        f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
        f"&offset={offset}&length={length}"
    )
    with urllib.request.urlopen(url, timeout=60) as fh:
        payload = json.load(fh)
    return [r["row"] for r in payload["rows"]]


def _normalize(benchmark: str, fmt: str, row: dict) -> dict[str, Any] | None:
    """Map a raw HF row onto the common source-item shape, or None if unusable."""
    if fmt == "mc" and benchmark == "mmlu_pro":
        opts = row.get("options") or []
        idx = row.get("answer_index")
        if idx is None or not 0 <= idx < len(opts):
            return None
        return {
            "benchmark": benchmark,
            "source_id": f"mmlu_pro:{row.get('question_id')}",
            "question": row["question"],
            "options": {LETTERS[i]: t for i, t in enumerate(opts)},
            "correct": LETTERS[idx],
            "format": "mc",
            "passage": None,
            "category": row.get("category"),
        }
    if benchmark == "arc":
        ch = row.get("choices") or {}
        labels, texts = ch.get("label") or [], ch.get("text") or []
        key = row.get("answerKey")
        if len(labels) != len(texts) or key not in labels:
            return None
        # A handful of ARC rows use 1/2/3/4 labels; re-letter them uniformly.
        relabel = {old: LETTERS[i] for i, old in enumerate(labels)}
        return {
            "benchmark": benchmark,
            "source_id": f"arc:{row.get('id')}",
            "question": row["question"],
            "options": {relabel[l]: t for l, t in zip(labels, texts)},
            "correct": relabel[key],
            "format": "mc",
            "passage": None,
            "category": "science",
        }
    if benchmark == "race":
        opts = row.get("options") or []
        key = row.get("answer")
        if key not in LETTERS[: len(opts)]:
            return None
        return {
            "benchmark": benchmark,
            # Distinct questions off one article still produce near-identical
            # prompts, because the passage is most of the prompt. Group on the
            # article so only one question per passage is ever drawn.
            "group_key": f"race:{row.get('example_id')}",
            "source_id": f"race:{row.get('example_id')}:{hash(row['question']) & 0xFFFF}",
            "question": row["question"],
            "options": {LETTERS[i]: t for i, t in enumerate(opts)},
            "correct": key,
            "format": "mc_passage",
            "passage": row.get("article"),
            "category": "reading_comprehension",
        }
    if benchmark == "gsm8k":
        ans = row.get("answer", "")
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans)
        if not m:
            return None
        return {
            "benchmark": benchmark,
            "source_id": f"gsm8k:{hash(row['question']) & 0xFFFFFFF}",
            "question": row["question"],
            "options": None,
            "correct": m.group(1).replace(",", ""),
            "format": "numeric",
            "passage": None,
            "category": "math_word_problem",
        }
    if benchmark == "trivia_qa":
        ans = row.get("answer") or {}
        value = ans.get("value")
        if not value:
            return None
        return {
            "benchmark": benchmark,
            "source_id": f"trivia_qa:{row.get('question_id')}",
            "question": row["question"],
            "options": None,
            "correct": value,
            "format": "short_answer",
            "aliases": ans.get("aliases") or [],
            "passage": None,
            "category": "trivia",
        }
    if benchmark == "truthful_qa":
        if not row.get("best_answer") or not row.get("incorrect_answers"):
            return None
        return {
            "benchmark": benchmark,
            "source_id": f"truthful_qa:{hash(row['question']) & 0xFFFFFFF}",
            "question": row["question"],
            "options": None,
            "correct": row["best_answer"],
            "format": "misconception",
            "misconception": row["incorrect_answers"][0],
            "correct_answers": row.get("correct_answers") or [],
            "passage": None,
            "category": row.get("category"),
        }
    return None


# Option text that makes "exactly one option carries the manipulated signal"
# ill-defined, or that makes a wrong-answer designation semantically slippery.
_BAD_OPTION = re.compile(
    r"^\s*(all|none|both)\s+(of\s+)?(the\s+)?(above|these|options)|^\s*(a\s+and\s+b|i\s+and\s+ii)\b",
    re.I,
)


def heuristic_filter(item: dict) -> str | None:
    """Return a rejection reason, or None if the item passes the cheap gates."""
    q = item["question"].strip()
    if len(q) < 15:
        return "question_too_short"
    opts = item.get("options")
    if opts:
        texts = [t.strip() for t in opts.values()]
        if len(set(t.lower() for t in texts)) != len(texts):
            return "duplicate_options"
        if any(_BAD_OPTION.match(t) for t in texts):
            return "aggregate_option"
        if any(len(t.strip()) == 0 for t in texts):
            return "empty_option"
        if len(opts) < 3:
            return "too_few_options"
    if item["format"] == "short_answer":
        # Multi-valued or list-like answers make exact-match scoring ambiguous.
        if len(str(item["correct"])) > 60 or "," in str(item["correct"]):
            return "answer_not_atomic"
    if item["format"] == "mc_passage" and len(item.get("passage") or "") > 6000:
        return "passage_too_long"
    return None


def fetch_pool(benchmark: str, n: int, *, seed: int = 0, offset: int = 0) -> list[dict]:
    """Fetch and normalize up to ``n`` heuristically-clean items from a benchmark."""
    dataset, config, split, fmt = SOURCES[benchmark]
    rng = random.Random(f"{benchmark}:{seed}")
    out: list[dict] = []
    seen: set[str] = set()
    groups: set[str] = set()
    cursor = offset
    while len(out) < n and cursor < offset + 20 * max(n, 1):
        rows = _get_rows(dataset, config, split, cursor, 100)
        if not rows:
            break
        cursor += len(rows)
        for row in rows:
            item = _normalize(benchmark, fmt, row)
            if item is None or item["source_id"] in seen:
                continue
            group = item.get("group_key")
            if group:
                if group in groups:
                    continue
                groups.add(group)
            reason = heuristic_filter(item)
            if reason:
                continue
            seen.add(item["source_id"])
            out.append(item)
            if len(out) >= n * 3:
                break
    rng.shuffle(out)
    return out[:n]


def stratified_allocate(
    pools: dict[str, list[dict]],
    consumers: list[tuple[str, int]],
    *,
    benchmarks: list[str],
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Deal source items to consumers so each consumer gets a balanced spread of
    benchmarks, and no source item is ever used twice anywhere.

    ``consumers`` is [(consumer_key, n_items), ...]. Dealing is round-robin over
    ``benchmarks`` per consumer, which is what stops one shard from being all
    GSM8K while another is all MMLU.
    """
    rng = random.Random(seed)
    cursors = {b: 0 for b in benchmarks}
    shuffled = {b: list(pools.get(b, [])) for b in benchmarks}
    for b in benchmarks:
        rng.shuffle(shuffled[b])

    out: dict[str, list[dict]] = {}
    # Rotate the starting benchmark per consumer so the spread differs shard to shard.
    for ci, (key, n) in enumerate(consumers):
        picked: list[dict] = []
        for j in range(n):
            for attempt in range(len(benchmarks)):
                b = benchmarks[(ci + j + attempt) % len(benchmarks)]
                if cursors[b] < len(shuffled[b]):
                    picked.append(shuffled[b][cursors[b]])
                    cursors[b] += 1
                    break
            else:
                raise RuntimeError(f"substrate exhausted for consumer {key} at item {j}")
        out[key] = picked
    return out
