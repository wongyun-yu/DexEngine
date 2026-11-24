"""Files of one RL campaign launch loop and the rules that read them.

The loop dir (run layer `run_rl_loops/<campaign>`) holds the pidfile, status.json, states.tsv and events.tsv, the
settings frozen at the first start (`params.json`) with their change ledger (`params_history.tsv`), and an optional
drain request (`drain.json`): while it exists the loop launches nothing new and ends once nothing is in flight.

A restart with a different code fingerprint or launch identity (`IDENT`: training-path fingerprint, command
fingerprint, inherited env) is a code swap: it is allowed only with nothing in flight, and an identity change only
with an explicit reason (`check_swap`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

PARAMS, PARAMS_LOG, DRAIN = "params.json", "params_history.tsv", "drain.json"
#: The parts of a campaign's launch identity (params.json keys).
IDENT = ("train_fp", "cmd_fp", "inherit_env")


def loop_dir(campaign: str, root=None) -> Path:
    """The loop dir of `campaign` (run layer run_rl_loops)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    P.check_rl_campaign(campaign)
    return P.run_path("run_rl_loops", root=root) / campaign


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def ledger_row(ldir: Path, kind: str, show: str, why: str) -> None:
    """Append one params_history.tsv row: time, kind, pid, what, why (whitespace folded)."""
    with open(ldir / PARAMS_LOG, "a", encoding="utf-8") as lg:
        lg.write(f"{_now()}\t{kind}\t{os.getpid()}\t{' '.join(show.split())}\t{' '.join(why.split())}\n")


def in_flight_keys(bundles: list[dict], states: dict, cell_of) -> list[str]:
    """ "hand/id" of the bundles in flight: launched and not terminal (training, trained, judging) or done but not
    finalized."""
    from dexengine.rl import cell_state as CH  # noqa: PLC0415

    return sorted(
        f"{h}/{i}"
        for b in bundles
        for h, i in [CH.bkey(b)]
        if states[(h, i)] in ("training", "trained", "judging")
        or (states[(h, i)] == "done" and not CH.cell_file(cell_of(b), "finalized").is_file())
    )


def read_request(ldir: Path) -> dict | None:
    """The drain request, or None. An unreadable request still drains (fail-closed)."""
    f = ldir / DRAIN
    if not f.is_file():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and d.get("why") else {"time": "?", "why": f"malformed {DRAIN}"}
    except (OSError, ValueError):
        return {"time": "?", "why": f"unreadable {DRAIN}"}


def drain_why(ldir: Path) -> str:
    """Why the launch gate is closed by a drain request ("" = no request)."""
    d = read_request(ldir)
    return (
        (
            f"drain requested ({d.get('time')} · {d.get('why')}) — no new launch; the loop ends with loop=drained "
            "once the running bundles finish"
        )
        if d
        else ""
    )


def clear(ldir: Path, kind: str, why: str) -> bool:
    """Remove the drain request with a ledger row (kind drain_cleared / drain_cancelled). False = there was none."""
    d = read_request(ldir)
    if d is None:
        return False
    (ldir / DRAIN).unlink()
    ledger_row(ldir, kind, f"request {d.get('time')} · {d.get('why')}", why)
    return True


def swap_status(bundles: list[dict], states: dict, cell_of, ldir: Path) -> dict:
    """The swap fields of status.json: bundles in flight and the drain request."""
    return {"in_flight": in_flight_keys(bundles, states, cell_of), "drain": read_request(ldir)}


def ident_of(p: dict) -> str:
    """One hash of a settings dict's launch identity (`IDENT`)."""
    return hashlib.sha1(json.dumps([p.get(k) for k in IDENT], sort_keys=True).encode()).hexdigest()


def ident_moved(old: dict, new: dict) -> list[str]:
    """The `IDENT` parts that differ between two settings dicts."""
    return [k for k in IDENT if old.get(k) != new.get(k)]


def _short(v) -> str:
    if isinstance(v, list):
        return hashlib.sha1(json.dumps(v).encode()).hexdigest()[:12] + f"({len(v)})"
    return str(v)[:12]


def check_swap(old: dict, new: dict, live: list[str], allow_why: str) -> tuple[bool, str]:
    """A restart's code and launch identity against the frozen settings → (go, kind). Unchanged = (True, ""). A
    change goes only with nothing in flight and, when the launch identity moved, with an allow reason."""
    moved = ident_moved(old, new)
    if old.get("code") == new.get("code") and not moved:
        return True, ""
    if live:
        return False, (
            f"a code or launch-identity change needs nothing in flight — running bundles {len(live)} "
            f"{live[:3]}; drain first (drain.json), wait for loop=drained, then restart"
        )
    if moved and not allow_why.strip():
        show = " · ".join(f"{k} {_short(old.get(k))}→{_short(new.get(k))}" for k in moved)
        return False, (
            f"the launch identity changed ({show}) — bundles before and after are not comparable; "
            'accept it with --allow-train-path-change "<why>"'
        )
    return True, "swap"


def swap_show(old: dict, new: dict, commit: str) -> str:
    """One-line description of a code swap for the ledger."""
    return f"commit {commit[:12]} · code {_short(old.get('code'))}→{_short(new.get('code'))} · " + " · ".join(
        f"{k} {_short(old.get(k))}→{_short(new.get(k))}" for k in IDENT
    )
