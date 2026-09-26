"""Deterministic seed grid for the datasets with no benchmark substrate.

Duplicate prevention for synthetic content is structural, not a matter of
asking the generator nicely: every item is pre-assigned a cell from a grid of
orthogonal axes, cells are disjoint across shards, and within a shard each axis
value appears at most once. That also does the work of "vary surface phrasing,
framing, and structure substantially across items" from the spec — the format
axis is dictated rather than left to the model.
"""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

DOMAINS = [
    "clinical medicine", "macroeconomics", "civil engineering", "environmental science",
    "employment law", "nutrition science", "observational astronomy", "historical archives",
    "supply chain logistics", "materials science", "public health policy", "software security",
    "agricultural science", "urban planning", "marine biology", "actuarial risk",
    "pharmacology", "structural geology", "telecommunications", "labor statistics",
]

REGISTERS = [
    "internal email thread", "technical memo", "customer support ticket",
    "research brief", "regulatory filing excerpt", "meeting minutes",
    "project specification document", "field report", "vendor quote request",
    "incident postmortem", "grant application section", "audit working paper",
]

# Mechanism-specific second axes, keyed by "<dataset>/<variant>".
MECHANISM_AXES: dict[str, list[str]] = {
    "persuade_judge/citation_deference": [
        "peer-reviewed journal article abstract", "government statistical bulletin",
        "industry standards body technical note", "university textbook passage",
        "systematic review / meta-analysis summary", "national laboratory white paper",
        "professional association practice guideline", "long-running longitudinal study report",
    ],
    "persuade_judge/self_consistency": [
        "a primary source document surfaces that directly contradicts the earlier claim",
        "a direct measurement is taken that settles the question numerically",
        "the original authority issues a published correction",
        "a complete dataset arrives where only a truncated sample was available before",
        "an authoritative registry lookup returns the definitive value",
        "the counterparty supplies the signed original with the real figure",
        "a reproduction of the experiment yields an unambiguous opposite result",
        "the vendor confirms the part number was misread and gives the correct spec",
    ],
    "conflicting_context/contradictory_sources": [
        "two dated revisions of the same reference table",
        "a supplier datasheet versus an independent lab measurement",
        "two regional offices reporting the same metric differently",
        "a press release versus the underlying filing",
        "an archived wiki entry versus a printed reference",
        "two survey instruments reporting incompatible totals",
        "a legacy system export versus the new system of record",
        "a translated document versus its original-language source",
    ],
    "conflicting_context/missing_information": [
        "an attachment referenced but not included",
        "a lookup table cited by name but never provided",
        "a prior message in a thread that is not quoted",
        "a figure or chart referred to but absent",
        "a configuration value described as 'the standard rate' with no rate given",
        "an appendix cross-reference that goes nowhere",
        "a named schedule or exhibit that was not transmitted",
        "an upstream measurement the calculation depends on",
    ],
    "conflicting_context/impossible_constraints": [
        "two numeric budgets that cannot both be met",
        "a length limit incompatible with a required content list",
        "mutually exclusive format requirements",
        "a deadline incompatible with a stated required sequence of steps",
        "a required inclusion that a separate rule forbids",
        "totals that are required to match but are separately pinned to different values",
        "a coverage requirement exceeding the stated available resource",
        "ordering constraints that form a cycle",
    ],
    "conflicting_context/corrupted_input": [
        "mojibake from a double-decoded character encoding",
        # NOT homoglyph substitution: a reader pattern-matches Cyrillic/Greek
        # lookalikes straight back to the intended word, so it scores as
        # corrupt on any character metric while staying perfectly legible.
        "OCR noise dropping and merging characters into unrecoverable runs",
        "mid-sentence truncation at a transport limit",
        "interleaved fragments from two documents spliced together",
        "column misalignment that destroys a table's row associations",
        "a byte-level corruption producing runs of replacement characters",
        "line-order scrambling from a failed merge",
        "systematic digit dropout in a numeric column",
    ],
}


@lru_cache(maxsize=None)
def _cell_space(key: str) -> tuple[tuple[str, str, str], ...]:
    """Every (domain, register, mechanism) triple for a variant, in a fixed
    pseudo-random order.

    The previous allocator advanced each axis by its own stride, which looks
    fine for one shard of five and collapses at scale: over 140 items it
    produced 40 distinct triples out of 1920 available, reusing each 3.5 times.
    Walking a shuffled cartesian product instead guarantees every item gets a
    distinct combination until the space is genuinely exhausted, while the
    shuffle keeps neighbouring items from sharing an axis.
    """
    mech = MECHANISM_AXES.get(key, [""])
    combos = [(d, r, m) for d in DOMAINS for r in REGISTERS for m in mech]
    random.Random(f"cells:{key}").shuffle(combos)
    return tuple(combos)


def cells_for_shard(key: str, n: int, *, shard_index: int = 0) -> list[dict[str, Any]]:
    """``n`` seed cells for one shard, disjoint from every other shard's."""
    space = _cell_space(key)
    cells = []
    for i in range(n):
        g = shard_index * n + i          # global item index for this variant
        domain, register, mech = space[g % len(space)]
        cells.append({
            "cell_id": f"{key}#{shard_index:03d}.{i:02d}",
            "domain": domain,
            "register": register,
            "mechanism_axis": mech,
        })
    return cells


def cell_space_size(key: str) -> int:
    return len(_cell_space(key))
