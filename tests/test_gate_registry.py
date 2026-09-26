"""Structural guard against spec/validator drift.

Every stuck shard in pilot 1 was a check the validator enforced that SPEC.md
never mentioned. Discipline did not prevent it; this test does.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward_hack_synth import gates, specs  # noqa: E402

VALIDATORS = ["reward_hack_synth/validators/items.py"]
ROOT = Path(__file__).resolve().parents[1]


def _ids_in(path: str) -> set[str]:
    """Gate ids passed to g.check(cond, "id") / bad("id")."""
    tree = ast.parse((ROOT / path).read_text())
    out: set[str] = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name == "check" and len(n.args) >= 2:
            arg = n.args[1]
        elif name == "bad" and n.args:
            arg = n.args[0]
        else:
            continue
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.add(arg.value)
    return out


def code_ids() -> set[str]:
    ids: set[str] = set()
    for p in VALIDATORS:
        ids |= _ids_in(p)
    return ids


def test_every_stated_mechanical_gate_is_enforced():
    """Prose with no check behind it is a promise the pipeline does not keep."""
    unenforced = gates.mechanical_ids() - code_ids() - gates.DYNAMIC_IDS
    assert not unenforced, f"stated in SPEC.md but never checked: {sorted(unenforced)}"


def test_every_check_is_attached_to_a_stated_gate():
    """A check the generator was never told about is the pilot-1 failure mode."""
    known = (gates.mechanical_ids() | gates.behavioral_ids()
             | set(gates.SUBCHECKS) | gates.SCHEMA_IDS | gates.DYNAMIC_IDS)
    orphans = code_ids() - known
    assert not orphans, (
        f"enforced but not stated in SPEC.md: {sorted(orphans)}. Add a GateSpec "
        f"with prose, or register it in SUBCHECKS under the gate it serves.")


def test_subcheck_parents_exist():
    all_ids = {g.id for g in gates.REGISTRY}
    dangling = {k: v for k, v in gates.SUBCHECKS.items() if v not in all_ids}
    assert not dangling, f"SUBCHECKS pointing at unknown gates: {dangling}"


def test_every_variant_renders_its_gates_into_the_spec():
    for dataset, variants in specs.VARIANTS.items():
        for variant in variants:
            spec = specs.build_spec(dataset, variant, 5)
            assert "### Hard gates" in spec, f"{dataset}/{variant}"
            for g in gates.for_variant(dataset, variant):
                head = " ".join(g.prose.split())[:60]
                assert head in " ".join(spec.split()), f"{dataset}/{variant} missing {g.id}"


# ---------------------------------------------------------------------------
# P0 — every gate needs a negative test
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).resolve().parent


def _negatively_tested_ids() -> set[str]:
    """Gate ids asserted to FIRE somewhere in the suite.

    A test that only shows good items passing does not establish that a check
    can see the violation it exists for. Three separate defects this project
    shipped were exactly that: the corruption metric was blind to homoglyphs,
    the difficulty probe measured the probe model rather than the task, and
    structural_extraction had no check at all for whether the exploit read the
    exposed file. Writing the negative test forces you to construct the
    violation, and a check that cannot see it fails immediately.
    """
    found: set[str] = set()
    for path in TEST_DIR.glob("test_*.py"):
        src = path.read_text()
        for node in ast.walk(ast.parse(src)):
            # any assert whose source mentions a gate id, in a file that is
            # checking for failure rather than success
            if not isinstance(node, ast.Assert):
                continue
            seg = ast.get_source_segment(src, node) or ""
            if "==  []" in seg or "== []" in seg:
                continue  # a "passes cleanly" assertion proves nothing here
            for const in ast.walk(node):
                if isinstance(const, ast.Constant) and isinstance(const.value, str):
                    found.add(const.value)
    return found


# Gates whose violation cannot be constructed as a unit-test fixture, with the
# reason. Keep this list short and justified — it is the escape hatch that
# lets an unverified check exist.
NEGATIVE_TEST_EXEMPT = {
    # Schema hygiene: malformed input, not a rule an item can violate while
    # doing the task correctly.
    "missing_field", "prompt_too_thin", "unknown_dataset", "unknown_variant",
    # Verified by calibration against labelled items rather than fixtures,
    # because the violation is a semantic property of real generated content.
    "new_info_not_dispositive", "sources_do_not_diverge",
    "answerable_without_the_missing_value", "corruption_does_not_defeat_reading",
}


def test_every_mechanical_gate_has_a_negative_test():
    tested = _negatively_tested_ids()
    missing = sorted(gates.mechanical_ids() - tested - NEGATIVE_TEST_EXEMPT)
    assert not missing, (
        "gates with no test proving they FIRE on a violation: "
        f"{missing}. Construct a violating item and assert the gate id appears, "
        "or add it to NEGATIVE_TEST_EXEMPT with a reason.")


def test_exemptions_are_real_gates():
    known = {g.id for g in gates.REGISTRY} | set(gates.SUBCHECKS) | gates.SCHEMA_IDS
    stale = sorted(NEGATIVE_TEST_EXEMPT - known)
    assert not stale, f"exemptions for gates that no longer exist: {stale}"


# ---------------------------------------------------------------------------
# scale: properties that only break at volume
# ---------------------------------------------------------------------------

def test_seed_cells_stay_distinct_at_volume():
    """A stride-based allocator produced 40 distinct triples across 140 items
    while 1920 existed — invisible in a 5-item shard, ruinous at 550."""
    from reward_hack_synth import seeds
    for key in ("conflicting_context/corrupted_input",
                "persuade_judge/self_consistency"):
        seen = set()
        n = 5
        for shard in range(110):          # 550 items
            for c in seeds.cells_for_shard(key, n, shard_index=shard):
                seen.add((c["domain"], c["register"], c["mechanism_axis"]))
        assert len(seen) == 550, f"{key}: {len(seen)} distinct cells for 550 items"


def test_plan_hits_the_requested_item_count_exactly():
    from collections import Counter
    from reward_hack_synth import orchestrator as orch
    pools = {b: [{"source_id": f"{b}:{i}", "question": "q",
                  "options": {"A": "x", "B": "y", "C": "z"}, "correct": "A"}
                 for i in range(400)]
             for b in ("mmlu_pro", "gsm8k", "trivia_qa", "arc", "race", "truthful_qa")}
    wanted = ["rigged_rubric", "persuade_judge", "conflicting_context"]
    shards = orch.plan_shards(pools, n_items=5, target_per_dataset=550, datasets=wanted)
    per = Counter()
    for s in shards:
        per[s["dataset"]] += s["n_items"]
    assert dict(per) == {d: 550 for d in wanted}, dict(per)
    ids = [a["id"] for s in shards for a in s["assignment"]]
    assert len(ids) == len(set(ids)) == 1650


def test_plan_refuses_more_items_than_seed_cells_exist():
    """Better to fail at planning than to silently emit duplicate cells."""
    import pytest as _pytest
    from reward_hack_synth import orchestrator as orch
    pools = {b: [{"source_id": f"{b}:{i}", "question": "q",
                  "options": {"A": "x", "B": "y"}, "correct": "A"} for i in range(9000)]
             for b in ("mmlu_pro", "gsm8k", "trivia_qa", "arc", "race", "truthful_qa")}
    with _pytest.raises(RuntimeError, match="distinct seed cells"):
        orch.plan_shards(pools, n_items=5, target_per_dataset=40000,
                         datasets=["conflicting_context"])


def test_cross_shard_duplicate_sweep_sees_across_shards(tmp_path):
    """The per-shard check compares five items; nothing compared shard to shard."""
    from reward_hack_synth import orchestrator as orch
    d = tmp_path / "conflicting_context"
    same = "The Northfield depot reported a throughput of 1420 units this quarter. " * 4
    for k, (vid, iid) in enumerate((("a", "x1"), ("b", "x2"))):
        v = d / vid
        v.mkdir(parents=True)
        (v / "items.jsonl").write_text(json.dumps(
            {"id": iid, "variant": vid, "prompt": same}))
    found = orch.cross_shard_duplicates(tmp_path)
    assert "conflicting_context" in found, found


def test_race_pool_draws_one_question_per_article():
    """Two questions off one RACE article yield ~0.75-similar prompts, because
    the passage is most of the prompt. Question-level disjointness is not
    enough; the passage has to be unique too."""
    from reward_hack_synth import substrate as sub
    rows = [{"example_id": "high1.txt", "question": f"Q{i}", "answer": "A",
             "options": ["a", "b", "c", "d"], "article": "Long passage. " * 40}
            for i in range(5)]
    normalized = [sub._normalize("race", "mc_passage", r) for r in rows]
    assert len({n["group_key"] for n in normalized}) == 1, "same article, one group"
    assert len({n["source_id"] for n in normalized}) == 5, "distinct questions"
