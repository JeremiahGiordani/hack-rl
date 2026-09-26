"""Shard planning, rejection feedback, and assembly.

All determinism lives here: which source items go where, which seed cells go
where, and what counts as an acceptable item. The generation containers get
none of this — they get a spec and an assignment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import seeds, specs, substrate

# Benchmarks used for the rigged-rubric shards — deliberately spanning
# different question *formats*, not just topics.
D1_BENCHMARKS = ["mmlu_pro", "gsm8k", "trivia_qa", "arc", "race"]
# Persuade-the-grader needs short, independently checkable ground truth.
PG_BENCHMARKS = ["mmlu_pro", "gsm8k", "trivia_qa"]

ABBR = {
    "rigged_rubric": "d1", "persuade_judge": "d2", "conflicting_context": "d3",
}


def _vabbr(variant: str) -> str:
    return "".join(w[0] for w in variant.split("_"))


def _ids(dataset: str, variant: str, n: int) -> list[str]:
    return [f"{ABBR[dataset]}_{_vabbr(variant)}_{i:04d}" for i in range(1, n + 1)]


LEDGER = "used_source_ids.json"


def load_used_ids(out_dir: Path) -> set[str]:
    """Source ids consumed by earlier invocations writing to this output dir.

    Disjointness was only ever enforced within a single run, so regenerating
    one shard in a separate invocation re-planned from a fresh substrate and
    could hand it a question another shard already used. That is exactly how
    trivia_qa:tc_217 ended up in both d1_sgb_0005 and d2_pg_0003.
    """
    p = out_dir / LEDGER
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()))
    except (json.JSONDecodeError, TypeError):
        return set()


def record_used_ids(out_dir: Path, items_jsonl: str) -> None:
    used = load_used_ids(out_dir)
    for line in items_jsonl.splitlines():
        if not line.strip():
            continue
        try:
            sid = (json.loads(line).get("source") or {}).get("source_id")
        except json.JSONDecodeError:
            continue
        if sid:
            used.add(sid)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LEDGER).write_text(json.dumps(sorted(used), indent=2))


def _split(total: int, parts: int) -> list[int]:
    """Distribute `total` items over `parts` variants as evenly as possible."""
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def plan_shards(verified: dict[str, list[dict]], n_items: int = 5,
                *, seed: int = 0, exclude_source_ids: set[str] | None = None,
                target_per_dataset: int | None = None,
                datasets: list[str] | None = None,
                shard_offset: int = 0) -> list[dict[str, Any]]:
    """Build the shard list.

    With `target_per_dataset` set, each dataset's target is split evenly across
    its variants and each variant is cut into shards of at most `n_items`. A
    shard stays small on purpose: it is one Codex call, and within-shard
    diversity is a gate, so growing the shard makes the generator's job harder
    rather than the run cheaper.
    """
    exclude = exclude_source_ids or set()
    want = datasets or list(specs.VARIANTS)
    shards: list[dict[str, Any]] = []

    d1_variants = specs.VARIANTS["rigged_rubric"]
    d2_variants = specs.VARIANTS["persuade_judge"]

    # How many items each benchmark-backed consumer needs.
    d1_total = target_per_dataset or (n_items * len(d1_variants))
    d1_counts = _split(d1_total, len(d1_variants))
    d2_counts = dict(zip(d2_variants,
                         _split(target_per_dataset or (n_items * len(d2_variants)),
                                len(d2_variants))))
    pg_need = d2_counts["persuade_grader"]

    # Reserve persuade-the-grader's slice of each shared pool before dealing D1,
    # so the two allocations cannot collide.
    per_bench_hold = -(-pg_need // len(PG_BENCHMARKS)) + 2
    d1_pools, pg_pools = {}, {}
    for b in D1_BENCHMARKS:
        pool = [i for i in verified.get(b, []) if i.get("source_id") not in exclude]
        hold = per_bench_hold if b in PG_BENCHMARKS else 0
        pg_pools[b] = pool[:hold]
        d1_pools[b] = pool[hold:]

    # Only draw substrate for datasets actually being generated: allocating for
    # dataset 1 while generating only dataset 3 fails on an unrelated pool.
    d1_alloc = substrate.stratified_allocate(
        d1_pools, list(zip(d1_variants, d1_counts)),
        benchmarks=D1_BENCHMARKS, seed=seed) if "rigged_rubric" in want else {}
    pg_alloc = substrate.stratified_allocate(
        pg_pools, [("persuade_grader", pg_need)],
        benchmarks=PG_BENCHMARKS, seed=seed + 1) if "persuade_judge" in want else {}

    tq_need = d2_counts["opinion_matching"]
    tq = [i for i in verified.get("truthful_qa", []) if i.get("source_id") not in exclude][:tq_need]
    if "persuade_judge" in want and len(tq) < tq_need:
        raise RuntimeError(f"truthful_qa pool too small: {len(tq)} < {tq_need}")

    def emit(dataset: str, variant: str, assignment: list[dict], **extra) -> None:
        # shard_offset continues an existing run's numbering, so a top-up draws
        # fresh seed cells instead of colliding with cells already generated.
        for k, chunk in enumerate(_chunk(assignment, n_items)):
            shards.append({"dataset": dataset, "variant": variant,
                           "n_items": len(chunk), "shard_index": shard_offset + k,
                           "assignment": chunk, **extra})

    def ids(dataset: str, variant: str, count: int) -> list[str]:
        start = shard_offset * n_items + 1
        return [f"{ABBR[dataset]}_{_vabbr(variant)}_{i:04d}"
                for i in range(start, start + count)]

    # --- dataset 1: every variant benchmark-backed -------------------------
    if "rigged_rubric" in want:
        for v, srcs in zip(d1_variants, (d1_alloc[v] for v in d1_variants)):
            emit("rigged_rubric", v,
                 [{"id": i, "source": s}
                  for i, s in zip(ids("rigged_rubric", v, len(srcs)), srcs)])

    # --- dataset 2: two benchmark-backed, two seed-grid ---------------------
    if "persuade_judge" in want:
        emit("persuade_judge", "opinion_matching",
             [{"id": i, "source": s}
              for i, s in zip(ids("persuade_judge", "opinion_matching", len(tq)), tq)])
        pg = pg_alloc["persuade_grader"]
        emit("persuade_judge", "persuade_grader",
             [{"id": i, "source": s}
              for i, s in zip(ids("persuade_judge", "persuade_grader", len(pg)), pg)])
        for v in ("citation_deference", "self_consistency"):
            count = d2_counts[v]
            cells = _cells("persuade_judge", v, count, n_items, shard_offset)
            emit("persuade_judge", v,
                 [{"id": i, "seed_cell": c}
                  for i, c in zip(ids("persuade_judge", v, count), cells)])

    # --- dataset 3: entirely seed-grid --------------------------------------
    if "conflicting_context" in want:
        counts = dict(zip(specs.VARIANTS["conflicting_context"],
                          _split(target_per_dataset or (n_items * 4), 4)))
        for v, count in counts.items():
            cells = _cells("conflicting_context", v, count, n_items, shard_offset)
            emit("conflicting_context", v,
                 [{"id": i, "seed_cell": c}
                  for i, c in zip(ids("conflicting_context", v, count), cells)])

    return shards


def _cells(dataset: str, variant: str, count: int, n_items: int,
           shard_offset: int = 0) -> list[dict]:
    """Seed cells for a whole variant, spread across as many shards as needed."""
    key = f"{dataset}/{variant}"
    space = seeds.cell_space_size(key)
    if count > space:
        raise RuntimeError(
            f"{key}: asked for {count} items but only {space} distinct seed cells "
            f"exist; widen the axes in seeds.py before generating this many")
    out: list[dict] = []
    for k in range(-(-count // n_items)):
        out += seeds.cells_for_shard(key, n_items, shard_index=shard_offset + k)
    return out[:count]


def render_rejections(problems: list[str], *, n_found: int, n_expected: int) -> str:
    """Feedback handed back to a shard for its next round.

    Deliberately one-sided: it says what is wrong and never says what is
    right. Nothing here may read as confirmation that the rest is acceptable.
    """
    lines = [
        "# Rejected",
        "",
        f"This shard was rejected. It produced {n_found} item(s); {n_expected} are required.",
        "",
        "The problems found are listed below. This list is what was caught — it is",
        "not a certificate that everything else is fine. Re-read SPEC.md and go",
        "back over every item, including ones not named here.",
        "",
    ]
    lines += [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, dataset: str, variant: str, *,
                  items_jsonl: str | None = None,
                  log: dict | None = None, shard_index: int = 0) -> None:
    """Persist one shard.

    JSONL shards go to their own file under `shards/`. Writing every shard of a
    variant to a single `items.jsonl` meant each one overwrote the last — at
    five items per variant that is invisible, at 110 shards per variant it
    discards almost everything. `consolidate()` merges them afterwards.
    """
    d = out_dir / dataset / variant
    d.mkdir(parents=True, exist_ok=True)
    if items_jsonl is not None:
        (d / "shards").mkdir(exist_ok=True)
        (d / "shards" / f"shard_{shard_index:03d}.jsonl").write_text(items_jsonl)
    if log is not None:
        name = "generation_log.json" if shard_index == 0 else f"generation_log_{shard_index:03d}.json"
        (d / name).write_text(json.dumps(log, indent=2))


def consolidate(out_dir: Path) -> dict[str, int]:
    """Merge each variant's per-shard files into one items.jsonl."""
    counts: dict[str, int] = {}
    for shards_dir in sorted(out_dir.glob("*/*/shards")):
        variant_dir = shards_dir.parent
        lines: list[str] = []
        for f in sorted(shards_dir.glob("shard_*.jsonl")):
            lines += [l for l in f.read_text().splitlines() if l.strip()]
        if lines:
            (variant_dir / "items.jsonl").write_text("\n".join(lines) + "\n")
            counts[f"{variant_dir.parent.name}/{variant_dir.name}"] = len(lines)
    return counts


def cross_shard_duplicates(out_dir: Path, *, threshold: float = 0.6) -> dict[str, list[str]]:
    """Near-duplicate prompts across a whole dataset, not just within a shard.

    The per-shard check compares five items against each other. At 550 items
    spread over 110 shards, two shards can produce near-identical prompts and
    nothing sees it, because no shard ever compares itself to another.
    """
    import re as _re
    from itertools import combinations

    def toks(t: str) -> set[str]:
        return set(_re.findall(r"[a-z0-9]+", (t or "").lower()))

    findings: dict[str, list[str]] = {}
    for ds_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        items: list[tuple[str, str, set[str]]] = []
        for j in sorted(ds_dir.rglob("items.jsonl")):
            for line in j.read_text().splitlines():
                if not line.strip():
                    continue
                it = json.loads(line)
                items.append((it.get("id", "?"), it.get("variant", "?"),
                              toks(it.get("prompt", ""))))
        hits = []
        for (ia, va, a), (ib, vb, b) in combinations(items, 2):
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac > threshold:
                hits.append(f"{va}/{ia} ~ {vb}/{ib} (jaccard {jac:.2f})")
        if hits:
            findings[ds_dir.name] = hits
    return findings


def write_raw(out_dir: Path, shard: dict, items_jsonl: str) -> None:
    """Checkpoint a shard's generated output before it is validated."""
    if not items_jsonl:
        return
    d = out_dir / "_raw" / shard["dataset"] / shard["variant"]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"shard_{shard.get('shard_index', 0):03d}.jsonl").write_text(items_jsonl)


def write_rejected(out_dir: Path, rec: dict) -> None:
    """Park a shard that never passed, out of the consolidation path."""
    d = out_dir / "_rejected" / rec["dataset"] / rec["variant"]
    d.mkdir(parents=True, exist_ok=True)
    idx = rec.get("shard_index", 0)
    (d / f"shard_{idx:03d}.jsonl").write_text(rec.get("items_jsonl", ""))
    (d / f"shard_{idx:03d}.problems.json").write_text(
        json.dumps({"round": rec.get("round"), "problems": rec.get("problems", [])}, indent=2))


def shortfall(out_dir: Path, target_per_dataset: int) -> dict[str, int]:
    """How many items each dataset is short of target after rejections."""
    counts: dict[str, int] = {}
    for ds_dir in sorted(p for p in out_dir.iterdir()
                         if p.is_dir() and not p.name.startswith("_")):
        n = 0
        for j in ds_dir.rglob("items.jsonl"):
            n += len([l for l in j.read_text().splitlines() if l.strip()])
        counts[ds_dir.name] = target_per_dataset - n
    return counts
