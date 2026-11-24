"""retarget/f0/solve_classes.class_loop — the one class-wise solve loop (f0 sides): class
order, the first feasible class wins, the verification retry doubles the clearance, budget statuses."""

from __future__ import annotations

from dexengine.retarget.f0.solve_classes import class_loop

CLASSES = (("fingers", "F", ["f0"]), ("wrist", "W", ["w0"]), ("free", "A", ["a0"]))


def _drive(gen, answers: dict, log: list):
    try:
        req = next(gen)
        while True:
            log.append(req)
            req = gen.send(answers.get(req[0], (None, float("inf"))))
    except StopIteration as e:
        return e.value


def test_first_feasible_class_wins_in_order():
    log, clr = [], [0.0]
    st = _drive(
        class_loop(CLASSES, lambda t, m, i: (t, clr[0]), lambda x: 0, 3, 0.001, clr, lambda: False),
        {"wrist": ("xw", 1.0), "free": ("xa", 0.5)},
        log,
    )
    assert st[0] == "ok" and st[1] == ("xw", 1.0, "wrist") and [r[0] for r in log] == ["fingers", "wrist"]


def test_rejected_candidate_retries_with_doubled_clearance_then_gives_up():
    log, clr, seen = [], [0.0], []
    st = _drive(
        class_loop(CLASSES, lambda t, m, i: (t, clr[0]), lambda x: seen.append(x) or 2, 2, 0.001, clr, lambda: False),
        {"fingers": ("xf", 0.1)},
        log,
    )
    assert st[0] == "none" and st[1] is None and len(st[2]) == 3
    assert [r[1] for r in log] == [0.001, 0.002, 0.004] and clr[0] == 0.0


def test_budget_before_an_attempt_and_between_classes():
    calls = iter([False, True])
    st = _drive(class_loop(CLASSES, lambda t, m, i: (t,), lambda x: 0, 1, 0.0, [0.0], lambda: next(calls)), {}, [])
    assert st[0] == "budget_class"
    st = _drive(class_loop(CLASSES, lambda t, m, i: (t,), lambda x: 0, 1, 0.0, [0.0], lambda: True), {}, [])
    assert st[0] == "budget"
