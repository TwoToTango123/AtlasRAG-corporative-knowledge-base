"""Wait for RouterAI and resume the authorized evaluation without an LLM supervisor.

Probes are unauthenticated GETs. Stops after completion, a non-network error,
three failed resumes, or two hours waiting for availability. Run one supervisor.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "routerai_full"


def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def complete(task="full"):
    if task=='alias':
        path=OUT/'summary.json'
        return path.exists() and json.loads(path.read_text(encoding='utf-8')).get('complete') is True
    if task=="citation_parts":
        path=OUT/'citation_parts_protocol.json'
        return (path.exists() and json.loads(path.read_text(encoding='utf-8')).get('complete') is True
                and all(len(list((OUT/'score_parts'/group).glob('*.json')))==51 for group in ['cases','reviews']))
    if task in {"validation", "repairs", "citations"}:
        path = OUT / "summary.json"
        return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("complete") is True
    if task == "followup":
        for group in ["single_no_hyde", "single_hyde"]:
            rows = [json.loads(p.read_text(encoding="utf-8")) for p in (OUT / group / "cases").glob("*.json")]
            if sum(r.get("status") == "ok" for r in rows) != 39:
                return False
        return (all(len(list((OUT / "reviews" / group).glob("*.json"))) == 39
                    for group in ["full_120b", "full_deepseek", "single_no_hyde", "single_hyde"])
                and len(list((OUT / "controls").glob("*.json"))) == 6)
    counts = []
    for model in ["openai_gpt-oss-120b", "deepseek_deepseek-v4.1-flash-1017"]:
        rows = [json.loads(p.read_text(encoding="utf-8"))
                for p in (OUT / model / "cases").glob("*.json")]
        counts.append(sum(row.get("status") == "ok" for row in rows))
    return counts == [39, 39]


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["full", "followup", "validation", "repairs", "citations", "citation_parts", "alias"], default="full")
    args = parser.parse_args()
    if args.task == "followup":
        OUT = ROOT / "results" / "routerai_followup"
    if args.task == "validation":
        OUT = ROOT / "results" / "validation"
    if args.task == "repairs":
        OUT = ROOT / "results" / "repair_v2"
    if args.task in {"citations","citation_parts"}:
        OUT = ROOT / "results" / "context_citations"
    if args.task=='alias':
        OUT=ROOT/'results/offline_checks/alias_end_to_end'
    module = {"full": "scripts.run_routerai_eval", "followup": "scripts.run_followup_eval",
              "validation": "scripts.run_validation_eval", "repairs": "scripts.run_repair_eval",
              "citations": "scripts.run_context_citation_eval", "citation_parts":"scripts.replay_citation_parts",
              "alias":"scripts.run_alias_end_to_end"}[args.task]
    deadline = time.monotonic() + 7200
    attempts = 0
    OUT.mkdir(parents=True, exist_ok=True)
    run_log = OUT / "background_run.log"
    log("Waiting for RouterAI; resuming existing cases only; existing RUB budget applies.")
    while time.monotonic() < deadline:
        if complete(args.task):
            log("Completed: all requested cases and reviews saved.")
            return
        try:
            response = httpx.get(
                "https://routerai.ru/api/v1/models/openai/gpt-oss-20b/endpoints", timeout=10)
            ready = response.is_success and "data" in response.json()
        except (httpx.HTTPError, ValueError):
            ready = False
        if not ready:
            log("RouterAI unavailable; next probe in 60 seconds.")
            time.sleep(60)
            continue
        attempts += 1
        log(f"RouterAI available; resume attempt {attempts}/3.")
        start_offset = run_log.stat().st_size if run_log.exists() else 0
        run_started = time.monotonic()
        with run_log.open("ab") as file:
            result = subprocess.run(
                [sys.executable, "-m", module, "--phase", "full"],
                cwd=ROOT, stdout=file, stderr=subprocess.STDOUT,
            )
        # The two-hour guard limits waiting for connectivity, not useful work
        # (or time the computer was suspended while the worker was running).
        deadline += time.monotonic() - run_started
        if result.returncode == 0 and complete(args.task):
            log("Completed: all requested cases and reviews saved; summary.md updated.")
            return
        with run_log.open("rb") as file:
            file.seek(start_offset)
            tail = "\n".join(file.read().decode("utf-8", errors="replace").splitlines()[-12:])
        network_error = any(word in tail for word in [
            "ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError",
            "ReadError", "WriteError", "WriteTimeout", "PoolTimeout",
            "RouterAI upstream completion error",
            "incomplete chunked read", "temporarily rate-limited", "HTTP 502", "HTTP 503", "HTTP 504",
        ])
        if not network_error or attempts >= 3:
            log("Stopped after evaluation error. See background_run.log; existing cases remain saved.")
            return
        log("Network error during resume; retaining reservations and checkpoints.")
        time.sleep(60)
    log("Stopped: two-hour availability wait expired. No further calls will be sent.")


if __name__ == "__main__":
    main()
