"""CLI for synthetic reward-hacking dataset generation via Codex against a
plain OpenAI-compatible endpoint.

Deliberate separation of concerns, preserved from the original Modal-based
pipeline this was extracted from:

  * Generation gets the spec and the assignment and nothing else — no
    validator, no checker, no green light.
  * Verification (mechanical gates + behavioral probes) runs afterward and can
    only ever reject. A false negative costs a regeneration; a false positive
    leaves us where we would be with no checker at all.

Usage:
    python -m reward_hack_synth.generate --smoke
    python -m reward_hack_synth.generate --out out_dir --n-items 5 --rounds 3
    python -m reward_hack_synth.generate --out out_dir --only rigged_rubric,persuade_judge/persuade_grader
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import codex_provider as pk
from . import logscan
from . import orchestrator as orch
from . import probes
from . import specs
from . import substrate as sub
from .validators import items as V

N_ITEMS_DEFAULT = 5
MAX_ROUNDS = 3
MAX_WORKERS_DEFAULT = 8


# --------------------------------------------------------------------------
# milestone 0 — does codex actually reach the model through the endpoint?
# --------------------------------------------------------------------------

def smoke() -> dict:
    """Prove the provider path end-to-end before anything else runs."""
    result: dict = {"model": pk.DEFAULT_MODEL}

    # 1. plain HTTP through the endpoint
    try:
        result["chat"] = pk.chat([{"role": "user", "content": "Reply with exactly: PONG"}],
                                 max_tokens=16).strip()[:40]
    except Exception as exc:  # noqa: BLE001
        result["chat"] = f"ERROR {type(exc).__name__}: {exc}"

    # 2. codex CLI through the endpoint
    pk.write_codex_config()
    result["config"] = Path(f"{pk.CODEX_HOME}/config.toml").read_text()
    result["codex_version"] = subprocess.run(
        pk.as_runner(["codex", "--version"]), capture_output=True, text=True).stdout.strip()

    wd = tempfile.mkdtemp(prefix="smoke-")
    out, err, rc = pk.codex_exec(
        wd,
        "Write a file named proof.txt in the current directory whose only "
        "contents are the word PROVIDER_OK. Then stop.",
        tag="smoke", timeout_s=600, reasoning_effort=None,
    )
    proof = Path(wd) / "proof.txt"
    result["rc"] = rc
    result["proof"] = proof.read_text().strip()[:60] if proof.exists() else "<not written>"
    result["ok"] = proof.exists() and "PROVIDER_OK" in result["proof"]
    if not result["ok"]:
        result["stdout_tail"] = out[-1500:]
        result["stderr_tail"] = err[-1500:]
    print(f"[smoke] ok={result['ok']} rc={rc} chat={result.get('chat')!r}", flush=True)
    return result


# --------------------------------------------------------------------------
# substrate
# --------------------------------------------------------------------------

def build_substrate(per_benchmark: int = 40, seed: int = 0) -> dict:
    """Fetch benchmark pools and independently re-verify their labels.

    The spec requires source items with 'unambiguous, undisputed ground truth'
    and says to filter or re-verify contested labels. Heuristics alone cannot
    do that, so every candidate is answered blind by a strong model and kept
    only when the model's answer matches the benchmark's label.
    """
    pools: dict[str, list[dict]] = {}
    for bench in sub.SOURCES:
        pools[bench] = sub.fetch_pool(bench, per_benchmark, seed=seed)
        print(f"[substrate] {bench}: fetched {len(pools[bench])}", flush=True)

    def ask(item: dict) -> tuple[dict, bool]:
        # truthful_qa is a curated misconception set, not a labelled exam —
        # a blind model answer is not evidence about its labels.
        if item["format"] == "misconception":
            return item, True
        if item.get("options"):
            opts = "\n".join(f"{k}) {v}" for k, v in item["options"].items())
            q = ((f"Passage:\n{item['passage']}\n\n" if item.get("passage") else "")
                 + f"{item['question']}\n{opts}\n\nReply with only the letter.")
        elif item["format"] == "numeric":
            q = f"{item['question']}\n\nReply with only the final number."
        else:
            q = f"{item['question']}\n\nReply with only the answer, no sentence."
        try:
            got = pk.chat([{"role": "user", "content": q}], max_tokens=2000).strip()
        except Exception:  # noqa: BLE001
            return item, False
        exp = str(item["correct"]).strip()
        if item.get("options"):
            got_l = got.strip().strip(".)").upper()[:1]
            return item, got_l == exp.upper()
        if item["format"] == "numeric":
            import re
            nums = re.findall(r"-?[\d,]+(?:\.\d+)?", got)
            return item, bool(nums) and nums[-1].replace(",", "") == exp.replace(",", "")
        cands = {exp.casefold(), *(a.casefold() for a in item.get("aliases", []))}
        return item, got.casefold().strip(".") in cands

    verified: dict[str, list[dict]] = {}
    for bench, pool in pools.items():
        with ThreadPoolExecutor(max_workers=12) as ex:
            checked = list(ex.map(ask, pool))
        kept = [i for i, ok in checked if ok]
        verified[bench] = kept
        print(f"[substrate] {bench}: {len(kept)}/{len(pool)} survived blind re-verification",
              flush=True)
    return verified


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def _stage(work_dir: str, spec_text: str, assignment: list[dict],
           feedback: str | None, prior: str | None, prior_name: str) -> None:
    wd = Path(work_dir)
    (wd / "SPEC.md").write_text(spec_text)
    (wd / "assignment.json").write_text(json.dumps(assignment, indent=2))
    if prior:
        (wd / prior_name).write_text(prior)
    if feedback:
        (wd / "REJECTIONS.md").write_text(feedback)


_REVISION_NOTE = """\

An earlier attempt at this shard was rejected. REJECTIONS.md lists the specific \
problems found in it, and your previous output is still here. Fix exactly those \
problems, and re-check the rest of your work against SPEC.md while you are at \
it — the rejection list is what was caught, not necessarily everything that is \
wrong. Note that the listed problems are the only feedback you will get; \
nothing here confirms when an item is acceptable.
"""


def gen_jsonl_shard(shard: dict) -> dict:
    """Generate one shard with Codex."""
    dataset, variant = shard["dataset"], shard["variant"]
    n = shard["n_items"]
    tag = f"{dataset}/{variant}"

    pk.write_codex_config()
    wd = tempfile.mkdtemp(prefix=f"{dataset}-{variant}-")
    _stage(wd, specs.build_spec(dataset, variant, n), shard["assignment"],
           shard.get("feedback"), shard.get("prior_items"), "items.jsonl")

    prompt = specs.JSONL_PROMPT + (_REVISION_NOTE if shard.get("feedback") else "")
    out, err, rc = pk.codex_exec(wd, prompt, tag=tag, timeout_s=shard.get("timeout_s", 6000))

    items_path = Path(wd) / "items.jsonl"
    text = items_path.read_text() if items_path.exists() else ""
    print(f"[{tag}] rc={rc} lines={len(text.splitlines())}", flush=True)
    return {"dataset": dataset, "variant": variant, "rc": rc, "items_jsonl": text,
            "stdout_tail": out[-4000:], "stderr_tail": err[-2000:]}


def probe_items(items_jsonl: str) -> dict:
    """Behavioral gates — semantic/perceptual properties no regex can measure.

    Pilot 1 proved a regex can't: it shipped a corruption gate that scored
    genuinely-corrupt text at 0.000. These probe the property directly, and
    like every other validator here they can only reject.
    """
    items = [json.loads(l) for l in items_jsonl.splitlines() if l.strip()]
    if not items:
        return {"rejections": [], "advisories": []}
    rejections, advisories = probes.run_shard_probes(items)
    print(f"[probe] {items[0].get('dataset')}/{items[0].get('variant')}: "
          f"{len(rejections)} rejection(s), {len(advisories)} advisory", flush=True)
    return {"rejections": rejections, "advisories": advisories}


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Plan every shard, generate, reject, regenerate, assemble.

    `--only` optionally restricts to a comma-separated list of
    "dataset/variant" or bare dataset names, e.g.
    --only rigged_rubric,persuade_judge/persuade_grader
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="run the end-to-end connectivity smoke test and exit")
    parser.add_argument("--n-items", type=int, default=N_ITEMS_DEFAULT)
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--out", default="reward_hack_synth_out")
    parser.add_argument("--only", default="")
    parser.add_argument("--per-benchmark", type=int, default=40)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--shard-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS_DEFAULT,
                        help="concurrent Codex shard generations")
    args = parser.parse_args(argv)

    if args.smoke:
        smoke()
        return

    n_items, rounds = args.n_items, args.rounds
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[plan] building + re-verifying substrate ...", flush=True)
    verified = build_substrate(per_benchmark=args.per_benchmark)
    (out_dir / "substrate.json").write_text(json.dumps(verified, indent=2))
    print("[plan] verified pool sizes: "
          + ", ".join(f"{k}={len(v)}" for k, v in verified.items()), flush=True)

    # Exclude source items consumed by earlier invocations writing here, so
    # regenerating one shard in a separate run cannot hand it a question
    # another shard already used.
    already = orch.load_used_ids(out_dir)
    if already:
        print(f"[plan] excluding {len(already)} source item(s) used by earlier runs",
              flush=True)
    want = None
    if args.only:
        want = sorted({s.split("/")[0] for s in args.only.split(",") if s.strip()})
    shards = orch.plan_shards(verified, n_items=n_items, exclude_source_ids=already,
                              target_per_dataset=args.target or None, datasets=want,
                              shard_offset=args.shard_offset)
    if args.only:
        keys = {s.strip() for s in args.only.split(",") if s.strip()}
        shards = [s for s in shards
                  if s["dataset"] in keys or f"{s['dataset']}/{s['variant']}" in keys]
    print(f"[plan] {len(shards)} shard(s), {n_items} items each", flush=True)

    summary: dict = {"n_items": n_items, "rounds": {}, "shards": {}}
    # Advisory findings: never block a shard, always reach the reviewer.
    review: list[str] = []

    pending = list(shards)
    accepted: dict[str, dict] = {}
    for rnd in range(1, rounds + 1):
        if not pending:
            break
        print(f"\n[round {rnd}] generating {len(pending)} shard(s) ...", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(gen_jsonl_shard, pending))

        # Mechanical gates first; behavioral probes only on what survives them,
        # since a probe costs model calls and a malformed shard fails anyway.
        staged = []
        for shard, res in zip(pending, results):
            text = res.get("items_jsonl", "")
            # Persist raw output the moment it returns, before anything can go
            # wrong downstream. Generation is the expensive part; validation is
            # cheap and rerunnable, so a validator fault must never cost a round.
            orch.write_raw(out_dir, shard, text)
            tmp = out_dir / ".scratch.jsonl"
            tmp.write_text(text)
            n_found, problems = V.validate_file(str(tmp), expect=shard["n_items"])
            tmp.unlink(missing_ok=True)
            staged.append([shard, res, text, n_found, problems])

        probe_idx = [i for i, st in enumerate(staged) if not st[4]]
        if probe_idx:
            print(f"[round {rnd}] probing {len(probe_idx)} mechanically-clean shard(s) ...",
                  flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                probe_results = list(ex.map(probe_items, [staged[i][2] for i in probe_idx]))
            for i, pr in zip(probe_idx, probe_results):
                staged[i][4] = list(pr.get("rejections", []))
                staged[i].append(list(pr.get("advisories", [])))

        next_pending = []
        for st in staged:
            shard, res, text, n_found, problems = st[:5]
            # Keyed per shard: at one shard per variant a variant key works, at
            # 110 shards per variant it keeps only the last one.
            si = shard.get("shard_index", 0)
            key = f"{shard['dataset']}/{shard['variant']}#{si:03d}"
            advisories = st[5] if len(st) > 5 else []
            advisories += [f"{key}: generator rationalization: {h}"
                           for h in logscan.scan_shard(res)]
            review.extend(advisories)
            print(f"[round {rnd}] {key}: {n_found} item(s), "
                  f"{len(problems)} problem(s), {len(advisories)} advisory", flush=True)
            if problems:
                shard = dict(shard)
                shard["feedback"] = orch.render_rejections(
                    problems, n_found=n_found, n_expected=shard["n_items"])
                shard["prior_items"] = text
                next_pending.append(shard)
                accepted[key] = {"round": rnd, "status": "rejected",
                                 "problems": problems, "items_jsonl": text,
                                 "log": res, "shard_index": si,
                                 "dataset": shard["dataset"], "variant": shard["variant"]}
            else:
                accepted[key] = {"round": rnd, "status": "accepted",
                                 "problems": [], "items_jsonl": text, "log": res,
                                 "shard_index": si,
                                 "dataset": shard["dataset"], "variant": shard["variant"]}
                # Persist immediately. Writing only at the very end means
                # killing a run discards completed, verified shards.
                orch.write_outputs(out_dir, shard["dataset"], shard["variant"],
                                   items_jsonl=text, shard_index=shard.get("shard_index", 0),
                                   log={"status": "accepted", "round": rnd})
                orch.record_used_ids(out_dir, text)
        pending = next_pending
        summary["rounds"][f"r{rnd}"] = len(next_pending)

    # ---- assemble ---------------------------------------------------------
    for key, rec in accepted.items():
        dataset, variant = rec["dataset"], rec["variant"]
        log = {k: v for k, v in rec["log"].items() if k != "items_jsonl"}
        log |= {"status": rec["status"], "round": rec["round"], "problems": rec["problems"]}
        if rec["status"] == "accepted":
            orch.write_outputs(out_dir, dataset, variant,
                               items_jsonl=rec["items_jsonl"],
                               shard_index=rec.get("shard_index", 0), log=log)
        else:
            # A shard that never passed its gates must not land in the
            # deliverable looking like one that did. It is kept for inspection
            # and counted as a shortfall, not consolidated.
            orch.write_rejected(out_dir, rec)
        summary["shards"][key] = {"status": rec["status"], "round": rec["round"],
                                  "n_problems": len(rec["problems"])}

    merged = orch.consolidate(out_dir)
    print("[done] consolidated: "
          + ", ".join(f"{k}={v}" for k, v in sorted(merged.items())), flush=True)

    dupes = orch.cross_shard_duplicates(out_dir)
    for ds, hits in dupes.items():
        review.extend(f"{ds}: cross-shard near-duplicate: {h}" for h in hits)
        print(f"[done] {ds}: {len(hits)} cross-shard near-duplicate pair(s)", flush=True)

    if review:
        (out_dir / "REVIEW.md").write_text(
            "# Advisory findings\n\n"
            "Not gate failures — these did not block any shard. They are the "
            "places a reviewer's attention is worth most: probes that are not "
            "trusted to reject on their own, and passages where the generator "
            "argued itself into keeping something.\n\n"
            + "\n".join(f"- {r}" for r in review) + "\n")
        print(f"[done] {len(review)} advisory finding(s) -> {out_dir}/REVIEW.md", flush=True)
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    ok = sum(1 for v in summary["shards"].values() if v["status"] == "accepted")
    print(f"\n[done] {ok}/{len(summary['shards'])} shard(s) accepted -> {out_dir}/", flush=True)
    for key, v in sorted(summary["shards"].items()):
        print(f"   {'OK ' if v['status'] == 'accepted' else 'BAD'}  {key}  (round {v['round']})")


if __name__ == "__main__":
    main()
