"""Probe sensitivity, measured against deliberately broken items.

Every earlier calibration only measured *specificity* — known-good items pass.
That says nothing about whether a probe can see the violation it exists for, and
a gate with unmeasured sensitivity that never fires is the appearance of
verification rather than verification. These fixtures supply the other half.

Needs a live API key (see .env.example); skipped without one.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward_hack_synth import codex_provider as pk  # noqa: E402
from reward_hack_synth import probes  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "known_bad_items.json"

pytestmark = pytest.mark.skipif(
    not os.environ.get(pk.API_KEY_ENV),
    reason="probe calibration needs a live API key")

# Measured 2026-09-02 against the original pipeline. A probe dropping below
# its floor has regressed.
SENSITIVITY_FLOOR = {
    "answerable_without_the_missing_value": 3,
    "corruption_does_not_defeat_reading": 2,
    "new_info_not_dispositive": 2,   # misses evidence about a different entity
    "sources_do_not_diverge": 3,
}


def _known_bad() -> list[dict]:
    return json.loads(FIXTURES.read_text())


def test_every_probe_has_known_bad_fixtures():
    """A probe with no violating example has never been shown to work."""
    covered = {i["_probe"] for i in _known_bad()}
    missing = sorted(set(probes.PROBES) - covered)
    assert not missing, f"probes with no known-bad fixture: {missing}"


def test_probe_sensitivity_meets_its_measured_floor():
    caught: dict[str, int] = defaultdict(int)
    for item in _known_bad():
        ok, _ = probes.PROBES[item["_probe"]](item)
        if not ok:
            caught[item["_probe"]] += 1
    regressions = {p: (caught[p], floor) for p, floor in SENSITIVITY_FLOOR.items()
                   if caught[p] < floor}
    assert not regressions, f"probes catching fewer violations than measured: {regressions}"


def test_no_probe_is_advisory_without_justification():
    """Advisory status is for probes whose behaviour is unmeasured. Once a probe
    has a sensitivity floor it should be gating, or the floor is a fiction."""
    unjustified = probes.ADVISORY_PROBES & set(SENSITIVITY_FLOOR)
    assert not unjustified, (
        f"{sorted(unjustified)} have measured sensitivity but are still advisory")
