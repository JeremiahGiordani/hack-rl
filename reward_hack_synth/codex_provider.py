"""Thin provider client + the Codex-CLI-over-a-plain-endpoint plumbing.

Two ways we reach a model, both against the same OpenAI-compatible endpoint:
  * ``chat()`` — plain HTTP, used for cheap non-agentic passes (substrate
    re-verification, behavioral probes). No Codex involved.
  * ``codex_exec()`` — the agentic path, used by the generation workers.

Configured entirely through environment variables (see ``.env.example``):
  LLM_BASE_URL      base URL of an OpenAI-compatible chat-completions API
                     (default: https://api.openai.com/v1)
  LLM_MODEL         model name/slug passed to both the plain client and the
                     Codex CLI config (default: gpt-5.1)
  LLM_API_KEY_ENV   name of the environment variable holding the API key
                     (default: OPENAI_API_KEY) — the key itself is never
                     hardcoded, only the name of where to find it
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.request
from pathlib import Path

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-5.1")
API_KEY_ENV = os.environ.get("LLM_API_KEY_ENV", "OPENAI_API_KEY")

MODEL_CONTEXT_WINDOW = 400_000
MODEL_MAX_OUTPUT_TOKENS = 128_000

RUNNER = os.environ.get("CODEX_RUNNER_USER", "")
RUNNER_HOME = f"/home/{RUNNER}" if RUNNER else os.path.expanduser("~")
CODEX_HOME = os.environ.get("CODEX_HOME", f"{RUNNER_HOME}/.codex")

BYPASS_FLAG = "--dangerously-bypass-approvals-and-sandbox"


def _api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is not set — see .env.example")
    return key


def chat(messages: list[dict], *, model: str = DEFAULT_MODEL, max_tokens: int = 512,
         timeout: int = 240, retries: int = 3, reasoning_effort: str | None = None) -> str:
    """One non-streaming chat completion through the configured endpoint."""
    key = _api_key()
    payload = {"model": model, "messages": messages,
               "max_completion_tokens": max_tokens}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    body = json.dumps(payload).encode()
    last = ""
    for _ in range(retries):
        req = urllib.request.Request(
            f"{LLM_BASE_URL}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "User-Agent": "reward-hack-synth/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                resp = json.load(fh)
            return resp["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"chat completion failed after {retries} attempts: {last}")


def write_codex_config(model: str = DEFAULT_MODEL, *, wire_api: str = "responses") -> None:
    """Point the Codex CLI at the configured endpoint.

    Two things matter and are easy to get wrong:
      * if generation runs as a different user (``CODEX_RUNNER_USER`` set),
        switching users switches HOME, so the config has to live in *that*
        user's ``~/.codex`` — not the caller's.
      * a leftover ``auth.json`` makes Codex prefer stored credentials and
        ignore ``env_key`` entirely, so it is removed here.
    """
    Path(CODEX_HOME).mkdir(parents=True, exist_ok=True)
    Path(f"{CODEX_HOME}/auth.json").unlink(missing_ok=True)
    Path(f"{CODEX_HOME}/config.toml").write_text(
        'model_provider = "reward_hack_synth"\n'
        f'model = "{model}"\n'
        # These must stay ABOVE the [model_providers.*] header: anything after
        # a table header belongs to that table, so putting them below silently
        # files them as provider keys and Codex still reports "model metadata
        # not found". Without them it falls back to default limits and makes
        # compaction decisions off the wrong numbers.
        f"model_context_window = {MODEL_CONTEXT_WINDOW}\n"
        f"model_max_output_tokens = {MODEL_MAX_OUTPUT_TOKENS}\n"
        "\n"
        "[model_providers.reward_hack_synth]\n"
        'name = "reward_hack_synth"\n'
        f'base_url = "{LLM_BASE_URL}"\n'
        f'env_key = "{API_KEY_ENV}"\n'
        f'wire_api = "{wire_api}"\n'
    )
    if RUNNER:
        subprocess.run(["chown", "-R", f"{RUNNER}:{RUNNER}", CODEX_HOME], check=False)


def as_runner(argv: list[str]) -> list[str]:
    """Run argv as ``CODEX_RUNNER_USER`` if set, with the API key and
    CODEX_HOME explicitly injected — ``runuser`` does not reliably carry
    either across the user switch. With no runner user configured (the
    common case for a local, single-user checkout) argv runs unchanged."""
    if not RUNNER:
        return argv
    return [
        "runuser", "-u", RUNNER, "--",
        "env",
        f"{API_KEY_ENV}={os.environ.get(API_KEY_ENV, '')}",
        f"CODEX_HOME={CODEX_HOME}",
        f"HOME={RUNNER_HOME}",
        *argv,
    ]


def _stream(argv: list[str], *, tag: str, timeout_s: int) -> tuple[str, str, int]:
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    out: list[str] = []
    err: list[str] = []

    def pump(stream, buf, label):
        try:
            for line in iter(stream.readline, ""):
                buf.append(line)
                print(f"[{tag}/{label}] {line.rstrip()}", flush=True)
        finally:
            stream.close()

    threads = [threading.Thread(target=pump, args=(proc.stdout, out, "out")),
               threading.Thread(target=pump, args=(proc.stderr, err, "err"))]
    for t in threads:
        t.start()
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        rc = 124
        print(f"[{tag}/err] TIMEOUT after {timeout_s}s — killed", flush=True)
    for t in threads:
        t.join()
    return "".join(out), "".join(err), rc


def codex_exec(work_dir: str, prompt: str, *, tag: str, timeout_s: int = 3600,
               reasoning_effort: str | None = "high") -> tuple[str, str, int]:
    """``codex exec`` in ``work_dir``, routed through the configured endpoint.

    The model is not passed on the command line — it comes from the
    ``model =`` line in config.toml, so the model name stays in one place.
    """
    argv = ["codex", "exec", "--cd", work_dir, "--skip-git-repo-check", BYPASS_FLAG]
    if reasoning_effort:
        argv += ["-c", f"model_reasoning_effort={reasoning_effort}"]
    argv.append(prompt)
    return _stream(as_runner(argv), tag=tag, timeout_s=timeout_s)
