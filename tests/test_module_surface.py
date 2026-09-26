"""Cheap guard on the public surface of each module.

A refactor once sliced `PROBES:` to end-of-file and silently deleted the
mechanism functions defined below it. Nothing failed locally — the tests never
touched the module's entry points — and it surfaced as a NameError partway
through a run. These assertions cost nothing and close that gap.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward_hack_synth import gates, logscan, orchestrator, probes, seeds, specs  # noqa: E402


def test_probe_entry_points_exist():
    for name in ("run_item_probes", "run_shard_probes", "probe_mechanism_identity",
                 "shard_mechanism_collapse"):
        assert callable(getattr(probes, name, None)), f"probes.{name} missing"


def test_every_registered_probe_is_callable():
    for gate_id, fn in probes.PROBES.items():
        assert callable(fn), gate_id
    assert probes.ADVISORY_PROBES <= set(probes.PROBES)


def test_every_behavioral_gate_has_a_probe():
    """A behavioral gate with no probe behind it is never actually checked."""
    unbacked = {g.id for g in gates.REGISTRY if g.kind == "behavioral"}
    unbacked -= set(probes.PROBES)
    assert not unbacked, f"behavioral gates with no probe: {sorted(unbacked)}"


def test_module_entry_points_exist():
    assert callable(orchestrator.plan_shards)
    assert callable(orchestrator.render_rejections)
    assert callable(orchestrator.write_outputs)
    assert callable(logscan.scan_shard)
    assert callable(seeds.cells_for_shard)
    assert callable(specs.build_spec)


def test_mechanism_one_liners_cover_every_variant():
    """The mechanism probe can only discriminate variants it has words for."""
    for dataset, variants in specs.VARIANTS.items():
        described = specs.MECHANISM_ONE_LINERS.get(dataset, {})
        assert set(described) == set(variants), (
            f"{dataset}: described {sorted(described)} vs actual {sorted(variants)}")


def test_no_module_lost_its_tail():
    """Guards against a truncating edit: each module keeps its last definition."""
    # Every public entry point, not just the last one I happened to remember:
    # a slice-to-end-of-file edit has silently deleted a module's tail three
    # times now, and each time the guard only covered the previous victim.
    for mod, last in ((probes, "run_shard_probes"), (gates, "behavioral_ids"),
                      (orchestrator, "write_outputs"), (orchestrator, "consolidate"),
                      (orchestrator, "cross_shard_duplicates"),
                      (orchestrator, "plan_shards"), (orchestrator, "load_used_ids"),
                      (seeds, "cell_space_size"), (seeds, "cells_for_shard")):
        src = inspect.getsource(mod)
        assert f"def {last}" in src, f"{mod.__name__} is missing {last}"


def test_vote_raises_rather_than_scoring_errors_as_failures():
    """45 rate-limited calls once became 'every shard scored 0/3', which read
    as a difficulty measurement and would have rejected sound items."""
    import pytest as _pytest

    def always_raises():
        raise RuntimeError("gateway 429")

    with _pytest.raises(probes.ProbeInconclusive):
        probes._vote(always_raises)


def test_vote_returns_normally_when_trials_succeed():
    assert probes._vote(lambda: True) == (probes.N_TRIALS, probes.N_TRIALS)
    assert probes._vote(lambda: False) == (0, probes.N_TRIALS)


def test_inconclusive_probe_advises_and_never_rejects(monkeypatch):
    def boom(_item):
        raise probes.ProbeInconclusive("gateway unavailable")

    monkeypatch.setitem(probes.PROBES, "new_info_not_dispositive", boom)
    item = {"id": "x", "dataset": "persuade_judge", "variant": "self_consistency"}
    rejections, advisories = probes.run_item_probes(item)
    assert rejections == []
    assert advisories and "could not be measured" in advisories[0]
