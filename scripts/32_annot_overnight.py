#!/usr/bin/env python
"""Overnight build of the annotation pool, throttled for a shared machine and a shared Gemma endpoint.

Three loops run side by side and each is resumable (re-running continues where it stopped):
  fill     grow the pool to --fill-to servable items (newest round, then backfill rounds); reads source DBs read-only.
  encoder  propaganda-technique pre-annotation (CPU, OMP_NUM_THREADS limited), in chunks.
  gemma    claim pre-annotation through the shared Gemma endpoint, in small chunks with a low concurrency (default 4,
           the project limit is 12) and a hard deadline. Items are taken in a hash order, so a partial run still covers
           the pool evenly instead of only the oldest rows.

Throttling: before every chunk the script waits while the machine's 1-minute load average is above --max-load-frac of
the cores (or, for Gemma, while GPU utilisation is above --max-gpu-util). Gemma backs off (halves its workers and sleeps)
when a chunk fails more than 20%. Everything runs under nice(1) when started as in docs/ANNOTATION.md.

  nice -n 10 .venv/bin/python scripts/32_annot_overnight.py --fill-to 60000
Progress: logs/annot_overnight.log (one JSON object per event) and logs/annot_overnight_status.json (latest state).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "8")                       # the encoder must not take every core of a shared box

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fnp.annotate import ingest  # noqa: E402
from fnp.annotate.db import tx  # noqa: E402
from fnp.annotate.preannotate import CLAIM_MODEL_PREFIX, ENC_VERSION  # noqa: E402

LOG = ROOT / "logs/annot_overnight.log"
STATUS = ROOT / "logs/annot_overnight_status.json"
_lock = threading.Lock()
_state: dict = {"started": datetime.now(timezone.utc).isoformat(), "fill": {}, "encoder": {}, "gemma": {}}
STOP = threading.Event()


def log(kind: str, **kw) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, **kw}
    with _lock:
        with open(LOG, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def status(part: str, **kw) -> None:
    with _lock:
        _state[part].update(kw)
        _state["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        STATUS.write_text(json.dumps(_state, indent=1, default=str))


# ------------------------------------------------------------------ throttling
def gpu_util() -> int | None:
    try:
        out = []
        for _ in range(3):
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            out.append(max(int(x) for x in r.stdout.split()))
            time.sleep(1)
        return round(sum(out) / len(out))
    except Exception:
        return None


def wait_calm(who: str, max_load: float, max_gpu: int | None) -> None:
    """Block until the machine is calm enough for this loop to do another chunk."""
    waited = 0
    while not STOP.is_set():
        reason = None
        if os.getloadavg()[0] > max_load:
            reason = f"load {os.getloadavg()[0]:.1f} > {max_load:.0f}"
        elif max_gpu is not None:
            g = gpu_util()
            if g is not None and g > max_gpu:
                reason = f"gpu util {g}% > {max_gpu}%"
        if reason is None:
            return
        if waited % 600 == 0:
            log("throttle", who=who, reason=reason)
        status(who, waiting=reason)
        STOP.wait(60)
        waited += 60
    return


# ------------------------------------------------------------------ counting
def n_pending(model: str) -> int:
    with tx() as cur:
        cur.execute("""SELECT count(*) n FROM item i WHERE i.status IN ('queued','in_progress') AND i.lang_group IN ('en','hi')
                       AND NOT EXISTS (SELECT 1 FROM suggestion s WHERE s.item_id=i.item_id AND s.model_version=%s)
                       AND NOT EXISTS (SELECT 1 FROM suggestion_done d WHERE d.item_id=i.item_id AND d.model_version=%s)""",
                    (model, model))
        return cur.fetchone()["n"]


def n_done(model: str) -> int:
    with tx() as cur:
        cur.execute("SELECT count(*) n FROM suggestion_done WHERE model_version=%s", (model,))
        return cur.fetchone()["n"]


# ------------------------------------------------------------------ loops
def fill_loop(a, fill_done: threading.Event) -> None:
    try:
        mode, rounds, small = "newest", 0, 0
        while not STOP.is_set() and ingest.servable_items() < a.fill_to and rounds < a.max_rounds:
            wait_calm("fill", a.max_load, None)
            res = ingest.run(mode, a.per_source, True, None, None, None, seed=101 + rounds)
            added = sum(r["inserted"] for r in res)
            errs = [r["error"] for r in res if r["error"]]
            rounds += 1
            have = ingest.servable_items()
            log("fill_round", round=rounds, mode=mode, inserted=added, servable=have, errors=errs[:3],
                per_platform={r["platform"]: r["inserted"] for r in res})
            status("fill", servable=have, target=a.fill_to, rounds=rounds, last_added=added)
            mode = "backfill"
            small = small + 1 if added < max(5, 0.02 * a.per_source * len(res)) else 0
            if small >= 3:
                log("fill_exhausted", servable=have, note="three rounds in a row added almost nothing; sources exhausted for the filters")
                break
            STOP.wait(5)
        status("fill", finished=True, servable=ingest.servable_items(), reached_target=ingest.servable_items() >= a.fill_to)
    except Exception as e:                                # never let one loop kill the others silently
        log("fill_error", error=f"{type(e).__name__}: {e}")
        status("fill", error=str(e))
    finally:
        fill_done.set()


def _run_preannotate(extra: list[str], timeout: int) -> dict | None:
    cmd = [sys.executable, "-m", "fnp.annotate.preannotate", "--apply", *extra]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    for line in reversed(r.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    log("preannotate_no_output", cmd=extra, stderr=r.stderr[-400:])
    return None


def encoder_loop(a, fill_done: threading.Event) -> None:
    try:
        done_this_run = 0
        while not STOP.is_set():
            left = n_pending(ENC_VERSION)
            status("encoder", pending=left, done_total=n_done(ENC_VERSION), done_this_run=done_this_run)
            if left == 0:
                if fill_done.is_set():
                    break
                STOP.wait(120)
                continue
            if a.max_encoder_items and done_this_run >= a.max_encoder_items:
                break
            wait_calm("encoder", a.max_load, None)
            n = min(a.encoder_chunk, a.max_encoder_items - done_this_run) if a.max_encoder_items else a.encoder_chunk
            t0 = time.time()
            res = _run_preannotate(["--no-gemma", "--limit", str(n)], 3600)
            got = (res or {}).get("encoder", {}).get("items", 0)
            done_this_run += got
            log("encoder_chunk", items=got, spans=(res or {}).get("encoder", {}).get("spans"), seconds=round(time.time() - t0), pending_before=left)
            if got == 0:
                STOP.wait(60)
        status("encoder", finished=True, pending=n_pending(ENC_VERSION), done_total=n_done(ENC_VERSION))
    except Exception as e:
        log("encoder_error", error=f"{type(e).__name__}: {e}")
        status("encoder", error=str(e))


def gemma_loop(a, fill_done: threading.Event) -> None:
    try:
        from fnp.llm import prompts as P
        model = CLAIM_MODEL_PREFIX + str(P.CLAIMS_VERSION)
        workers, deadline = a.gemma_workers, time.time() + a.gemma_hours * 3600
        done_this_run, backoffs = 0, 0
        while not STOP.is_set():
            left = n_pending(model)
            status("gemma", model=model, pending=left, done_total=n_done(model), done_this_run=done_this_run, workers=workers)
            if time.time() > deadline:
                log("gemma_deadline", done_this_run=done_this_run, pending=left)
                status("gemma", stopped="deadline")
                break
            if left == 0:
                if fill_done.is_set():
                    break
                STOP.wait(120)
                continue
            if a.max_gemma_items and done_this_run >= a.max_gemma_items:
                break
            wait_calm("gemma", a.max_load, a.max_gpu_util)
            n = min(a.gemma_chunk, a.max_gemma_items - done_this_run) if a.max_gemma_items else a.gemma_chunk
            before = n_done(model)
            t0 = time.time()
            try:
                res = _run_preannotate(["--no-encoder", "--limit", str(n), "--workers", str(workers)], 3600)
            except subprocess.TimeoutExpired:
                res = None
            took = time.time() - t0
            asked = (res or {}).get("gemma", {}).get("items", n)
            ok = n_done(model) - before
            done_this_run += ok
            fail = 1 - ok / asked if asked else 0
            log("gemma_chunk", asked=asked, ok=ok, fail_rate=round(fail, 2), seconds=round(took), s_per_item=round(took / max(ok, 1), 1), workers=workers)
            if fail > 0.2:                                        # the shared endpoint is struggling: halve and rest
                backoffs += 1
                workers = max(1, workers // 2)
                pause = min(900, 120 * backoffs)
                log("gemma_backoff", workers=workers, sleep_s=pause)
                STOP.wait(pause)
            else:
                backoffs = 0
                STOP.wait(a.gemma_pause)                          # small gap between chunks: leaves room for other users
        status("gemma", finished=True, pending=n_pending(model), done_total=n_done(model))
    except Exception as e:
        log("gemma_error", error=f"{type(e).__name__}: {e}")
        status("gemma", error=str(e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill-to", type=int, default=60000)
    ap.add_argument("--per-source", type=int, default=250)
    ap.add_argument("--max-rounds", type=int, default=80)
    ap.add_argument("--encoder-chunk", type=int, default=500)
    ap.add_argument("--gemma-chunk", type=int, default=150)
    ap.add_argument("--gemma-workers", type=int, default=4, help="Gemma concurrency (project cap is 12; shared endpoint)")
    ap.add_argument("--gemma-hours", type=float, default=11.0, help="hard stop for the Gemma loop")
    ap.add_argument("--gemma-pause", type=int, default=20, help="seconds between Gemma chunks")
    ap.add_argument("--max-load-frac", type=float, default=0.6, help="pause while 1-min load > this x cores")
    ap.add_argument("--max-gpu-util", type=int, default=85, help="Gemma pauses while GPU utilisation is above this %")
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--no-encoder", action="store_true")
    ap.add_argument("--no-gemma", action="store_true")
    ap.add_argument("--max-encoder-items", type=int, default=0, help="smoke tests: stop after this many items")
    ap.add_argument("--max-gemma-items", type=int, default=0, help="smoke tests: stop after this many items")
    a = ap.parse_args()
    a.max_load = a.max_load_frac * (os.cpu_count() or 8)
    log("start", args=vars(a), servable=ingest.servable_items())
    fill_done = threading.Event()
    if a.no_fill:
        fill_done.set()
    loops = ([] if a.no_fill else [threading.Thread(target=fill_loop, args=(a, fill_done), name="fill")]) \
        + ([] if a.no_encoder else [threading.Thread(target=encoder_loop, args=(a, fill_done), name="encoder")]) \
        + ([] if a.no_gemma else [threading.Thread(target=gemma_loop, args=(a, fill_done), name="gemma")])
    try:
        for t in loops:
            t.start()
        for t in loops:
            t.join()
    except KeyboardInterrupt:
        STOP.set()
        for t in loops:
            t.join(timeout=30)
    summary = {"servable": ingest.servable_items(), **{k: v for k, v in _state.items() if k in ("fill", "encoder", "gemma")}}
    log("finished", **summary)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
