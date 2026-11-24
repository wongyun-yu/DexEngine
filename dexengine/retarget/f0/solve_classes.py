"""The class-wise solve loop with strict verification — one function for its caller (f0's sides in repair_solve). It
imports nothing, so a caller can use it without the f0 hand configuration."""

from __future__ import annotations


def _clr_mm(x: float) -> str:
    return f"{x * 1000:.1f}mm"


def class_loop(classes, request, verify, retries: int, clearance: float, clr: list, over_budget, retryable=None):
    """The class-wise solve with strict verification — the one loop of its caller (f0's sides). A generator: yields
    what `request(tag, mask, inits)` builds and receives (x, cost) (x None = no feasible result of that class). Per
    attempt the solve clearance clr[0] is clearance·2^attempt; classes are tried in order
    and the first that returns a result is the candidate; `verify(x)` counts what the exact test still finds (0 =
    accepted). A rejected candidate retries with the doubled clearance, `retries` times at most; `retryable()` (None =
    always) False after a rejection ends the loop there (a failure a larger clearance cannot change).
    → (status, candidate (x, cost, tag) | None, verification history): status "ok" · "none" (no class found a result,
    or every attempt was rejected) · "budget" (over_budget() before an attempt) · "budget_class" (cut between
    classes)."""
    hist, cut = [], False
    for attempt in range(max(1, retries + 1)):
        if over_budget():
            return "budget", None, hist
        clr[0] = clearance * (2**attempt)
        cand = None
        for tag, mask, inits in classes:
            if over_budget():
                cut = True
                break
            x, c = yield request(tag, mask, inits)
            if x is None:
                continue
            cand = (x, c, tag)
            break  # first class that CAN do it wins, in priority order
        clr[0] = 0.0
        if cand is None:
            break
        n = verify(cand[0])
        hist.append(f"clearance{_clr_mm(clearance * (2**attempt))}→overlap {n}")
        if n == 0:
            return "ok", cand, hist
        if retryable is not None and not retryable():
            break
    return ("budget_class" if cut else "none"), None, hist
