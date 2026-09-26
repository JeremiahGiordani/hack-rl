# reward-hack-synth

A generator for synthetic reward-hacking training data — three datasets, each
built around one exploit mechanism:

- **`rigged_rubric`** — a stated cue in the prompt (a leaked answer key, a
  point table, a "known grader bug" note, a keyword-gaming checklist) diverges
  from the truth. No inference required; the hack is reading the signal.
- **`persuade_judge`** — no explicit cue anywhere. The model must *infer* an
  unstated preference (a confidently-stated wrong belief, a fabricated
  citation, an earlier committed-to claim, or a judge who might reward
  polished advocacy over correctness) and choose to serve it over the truth.
- **`conflicting_context`** — the task itself contains a genuine gap or
  contradiction (disagreeing sources, missing information, unsatisfiable
  constraints, corrupted input). The hack is proceeding confidently as though
  the problem weren't there.

Every item is **dual-scorable**: it carries both an `honest` reward spec (the
truthful response) and a `hacky` one (the response that games the proxy), so
downstream research can reward either path and measure what happens.

This is an extraction of one research pipeline's dataset-construction code,
trimmed to exactly these three datasets and repointed from an internal
gateway at a plain OpenAI-compatible API key. It needs no cloud infrastructure
— everything here runs as a local CLI.

## The gate registry — the part worth reading first

Every rule an item must satisfy lives exactly once, in
[`reward_hack_synth/gates.py`](reward_hack_synth/gates.py), as a `GateSpec`:
prose (what the generator is told) plus a mechanical check or behavioral probe
(what verifies it). `specs.py`'s SPEC.md text is *rendered* from this
registry — a rule cannot be enforced without being stated to the model that
has to satisfy it, and `tests/test_gate_registry.py` asserts the two halves
never drift apart (every mechanical id in the registry has code behind it,
every check in the code is attached to a stated gate, and — the sharper
requirement — every mechanical gate has a test proving it actually *fires* on
a constructed violation, not just that clean items pass).

Two properties can't be checked by regex at all — whether corrupted text is
genuinely illegible, whether two sources really diverge, whether new evidence
is genuinely dispositive — so those are gated by **behavioral probes**
(`probes.py`): put a model in the position the gate describes and measure
what it does, never ask it to self-report.

## Layout

```
reward_hack_synth/
  gates.py            the gate registry (see above)
  specs.py             SPEC.md text per dataset/variant, rendered from gates.py
  seeds.py             deterministic seed grid for the two non-benchmark-backed datasets
  substrate.py         pulls + blind-reverifies real benchmark questions (MMLU-Pro, GSM8K, ARC, RACE, TriviaQA, TruthfulQA)
  orchestrator.py      shard planning, rejection feedback, assembly
  probes.py            behavioral gates (semantic/perceptual properties)
  logscan.py           flags the generator's own rationalization in its logs
  codex_provider.py     Codex CLI + plain-HTTP client against your configured endpoint
  generate.py           the CLI entry point
  validators/items.py   mechanical gates — stdlib only, orchestrator-side only
tests/                  the gate test suite (see below)
```

## Setup

```bash
pip install -e .
npm install -g @openai/codex@0.152.1   # pinned — config.toml schema drifts between releases
cp .env.example .env                   # fill in your API key
export $(cat .env | xargs)             # or use direnv / your shell's usual mechanism
```

`LLM_BASE_URL` defaults to `https://api.openai.com/v1`; point it at any
OpenAI-compatible provider or gateway. `LLM_MODEL` and `LLM_API_KEY_ENV` are
also configurable — see `.env.example`.

## Running it

```bash
# 1. Prove the Codex-CLI-through-your-endpoint path works end to end.
python -m reward_hack_synth.generate --smoke

# 2. Generate. Pulls + blind-reverifies benchmark substrate, plans shards,
#    generates with Codex, validates (mechanical gates, then behavioral
#    probes on what survives), rejects and regenerates up to --rounds times,
#    then assembles items.jsonl per dataset/variant.
python -m reward_hack_synth.generate --out my_run --n-items 5 --rounds 3

# Useful flags:
#   --only rigged_rubric,persuade_judge/persuade_grader   restrict datasets/variants
#   --target 550       total items for the run (split evenly across variants)
#   --workers 8        concurrent Codex shard generations (local thread pool)
```

Output lands under `<out>/<dataset>/<variant>/items.jsonl`, with a
`REVIEW.md` of advisory findings (probes not trusted to reject on their own,
and any generator rationalization `logscan.py` caught) and a `SUMMARY.json`.

## Tests

```bash
python -m pytest tests/ -v
```

`test_gate_registry.py`, `test_module_surface.py`, and most of
`test_validators.py` run with no network access and no API key.
`test_probe_calibration.py` needs a live key (it's skipped without one) — it
measures each probe's sensitivity against deliberately broken fixtures in
`tests/fixtures/known_bad_items.json`, not just its specificity against
known-good items.

## What generation actually costs

Nothing here needs a cloud account — `generate.py`'s concurrency is a plain
local thread pool standing in for the original pipeline's Modal `.map()`. Each
accepted shard is one Codex CLI invocation; a $-per-item budget depends
entirely on which model you point `LLM_MODEL` at.
