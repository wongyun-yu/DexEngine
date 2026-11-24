"""One bundle of the record driver (record.driver; rules in its module doc): its record.bundle call(s) as new sessions
and its end. --cpu-tail on = the render call (--until render), then the store call (--gpu none) in the worker's tail;
off = one call. End (_settle) = count and frame check, merge, cleanup, result.json and state. A mixin of
driver.Pilot (reads self.a, self.run, self.me).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

# bundle = one new session; the whole session is reaped at the cap and the end (run_session)
from dexengine.cluster import proc as PROC
from dexengine.core import paths as P

# completion threshold default (MIN_RATE) · RC_PARTIAL; owner of bundle processing
from dexengine.record import bundle as C
from dexengine.record import clean as CLN  # intermediate output cleanup of checked bundles
from dexengine.record import driver_spec as SPM  # spec mode, final tally
from dexengine.record import piece as PC
from dexengine.record import sources as OWN  # launch owner ledger vs frozen copy check
from dexengine.record.augment import subrun as SR  # bundle deadline env (read by the generation cap GenBound)
from dexengine.record.driver_check import aug_fail_line, bdir_name, du, now, reason_of, say, verify, write_atomic
from dexengine.record.piece import merge_into, piece_dir  # piece identity, merge call

KILL_AFTER_S = 120  # SIGTERM -> SIGKILL gap after the bundle cap (same as the old timeout -k 120)


class BundleRun:
    def run_bundle(self, b: dict, gpu: str, W: dict) -> dict:
        a = self.a
        d = self.run / "bundles" / bdir_name(b)
        d.mkdir(parents=True, exist_ok=True)
        out = piece_dir(a.hand, a.combo, a.main, b, a.round, a.part)
        rd = PC.round_dir(out, a.round, a.part)
        (d / "target.txt").write_text("".join(c + "\n" for c in b["clips"]))
        t0 = time.time()
        write_atomic(d / "state", f"RUNNING pid={self.me} gpu={gpu} start={now()} clips={len(b['clips'])}\n")
        W.update(
            state="running", bundle=bdir_name(b), dataset=b["dataset"], since=now(), clips=len(b["clips"]), note=""
        )
        timeout = a.bundle_timeout_base + a.bundle_timeout_per_clip * len(b["clips"])
        cmd = [
            a.py,
            "-m",
            "dexengine.record.bundle",
            "--hand",
            a.hand,
            "--owner",
            str(a.owner),
            "--owner-sha",
            a.owner_sha,
            "--owner-pin",
            str(a.ledger),
            "--clips",
            ",".join(b["clips"]),
            "--gpu",
            gpu,
            "--round",
            a.round,
            *(["--part", a.part] if a.part else []),
            "--out",
            str(out),
            "--min-rate",
            str(a.min_rate),
            "--main",
            str(a.main),
            "--queue-root",
            a.queue_root,
            *(["--orig-fov-out", ",".join(r0)] if (r0 := SPM.row0_of(a, b["clips"])) else []),
        ]
        say(f"start GPU {gpu} · {b['dataset']} · {bdir_name(b)} · clips {len(b['clips'])} · cap {timeout} s · {out}")
        tail_on = a.cpu_tail == "on"
        rc = self._session(b, d, cmd + (["--until", "render"] if tail_on else []), timeout, "w")
        if tail_on and rc in (0, C.RC_PARTIAL) and self._log_has(d, C.RENDER_DONE_LINE):
            t1 = time.time()
            say(f"GPU {gpu} free after the render · {bdir_name(b)} · {t1 - t0:.0f} s · store, check, merge on the CPU")

            def tail() -> dict:
                g = cmd.index("--gpu") + 1
                rc2 = self._session(b, d, cmd[:g] + [C.CPU_ONLY] + cmd[g + 1 :], timeout, "a")
                return self._settle(
                    b, d, gpu, rc2, t0, out, rd, {"gpu_s": round(t1 - t0, 1), "cpu_s": round(time.time() - t1, 1)}
                )

            return {"tail": tail}
        return self._settle(b, d, gpu, rc, t0, out, rd)

    def _session(self, b: dict, d: Path, cmd: list[str], timeout: float, mode: str) -> int:
        """One bundle call as a new session (its whole session reaped at the end) → rc; output to collect.log (w = a
        new log, a = appended: the store call after --until render). The session env carries this call's end time
        (its start + its cap: the deadline of the generation cap augment.subrun.GenBound)."""
        env = dict(os.environ, **SR.deadline_env(time.time(), timeout))
        with open(d / "collect.log", mode) as f:
            f.write("# " + " ".join(cmd) + "\n")
            f.flush()
            rc, swept = PROC.run_session(
                cmd, timeout, KILL_AFTER_S, cwd=self.a.tree, stdout=f, stderr=subprocess.STDOUT, env=env
            )
        say(f"[BUNDLE_SESSION] {bdir_name(b)} rc {rc} · {swept} processes left in the session reaped")
        return rc

    @staticmethod
    def _log_has(d: Path, head: str) -> bool:
        return any(ln.startswith(head) for ln in (d / "collect.log").read_text(errors="replace").splitlines())

    def _settle(
        self, b: dict, d: Path, gpu: str, rc: int, t0: float, out: Path, rd: Path, tail: dict | None = None
    ) -> dict:
        """The bundle's end (after its last call): check, merge, cleanup, result.json and state."""
        a = self.a
        sec = time.time() - t0
        log = (d / "collect.log").read_text(errors="replace")
        end = C.DONE_LINE
        done_line = next((ln for ln in reversed(log.splitlines()) if ln.startswith(end)), "")
        res = dict(
            bundle=b["bundle"],
            campaign=b["campaign"],
            dataset=b["dataset"],
            gpu=gpu,
            rc=rc,
            sec=round(sec, 1),
            out=str(out),
            round=a.round,
            round_dir=str(rd),
            clips=len(b["clips"]),
            aug_failed=bool(aug_fail_line(log)),
            spec_id=a.spec_id or None,
            started=dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
            ended=now(),
            done_line=done_line,
            **({"cpu_tail": tail} if tail else {}),
        )
        if rc in (0, C.RC_PARTIAL) and done_line and (emp := PC.empty_result(rd, b["clips"])):
            res.update(emp)
        elif rc in (0, C.RC_PARTIAL) and done_line and not aug_fail_line(log):
            try:
                v = verify(out, b["clips"], rd / "dataset", run_log=log)
            except Exception as e:  # noqa: BLE001 — a check that could not run is a failure reason too
                v = {"ok": False, "problems": [f"check failed {type(e).__name__}: {e}"]}
            if OWN.log_problems(log):  # a run without the owner ledger marker is invalid
                v = dict(v, ok=False, problems=v.get("problems", []) + OWN.log_problems(log))
            res.update(v)
            res["state"] = "DONE" if v["ok"] else "FAILED"
            res["reason"] = "" if v["ok"] else "count and frame check " + " | ".join(v["problems"])[:300]
            merge_into(res, a.merger, P.layer_path("collect", b["dataset"], combo=a.combo), out, a.round, a.part)
            # only checked ∧ merged bundles are removed (otherwise skipped; a guard violation removes nothing and dies)
            try:
                res["cleanup"] = CLN.clean(out, v if res["state"] == "DONE" else dict(v, ok=False), rd, b["clips"])
            except (SystemExit, OSError) as e:
                res["state"], res["reason"] = "FAILED", f"intermediate output cleanup failed: {e}"[:300]
            res["bytes_total"] = du(out)
        else:
            res["state"], res["reason"] = "FAILED", reason_of(d / "collect.log", rc)
        write_atomic(d / "result.json", json.dumps(res, indent=1, ensure_ascii=False))
        write_atomic(
            d / "state",
            f"{res['state']} rc={rc} sec={sec:.0f} end={now()}"
            + (f" reason={res['reason']}" if res["reason"] else "")
            + "\n",
        )
        say(
            f"{res['state']} GPU {gpu} · {bdir_name(b)} · rc {rc} · {sec:.0f} s · "
            + (
                f"clips {res.get('clips', res.get('episodes'))}/{res.get('target')} · episodes {res.get('episodes')} · "
                f"excluded {len(res.get('excluded', {}))}"
                if res["state"] == "DONE"
                else res["reason"][:200]
            )
        )
        return res
