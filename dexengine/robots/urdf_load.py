"""URDF → pinocchio model, in one place — the urdfdom '<capsule>' warnings are folded into one line (the model is
exactly pin.buildModelFromUrdf's).

Why: the rby1 URDF states each link's collider as `<capsule>`, which is not a standard URDF shape (box · cylinder ·
sphere · mesh). urdfdom prints 'Could not parse collision element for Link [..]' and 'at line 559 in …/link.cpp' to the
C++ stderr per link and skips that collision element. That is harmless — the pinocchio model's colliders are used
nowhere: geom.load_capsules reads the capsules from the XML directly, and the judge colliders are built by sim_collider.
Per process that is links × 3 lines (about 1,000 lines per clip with 12 post-gate workers), burying real warnings, so
they become one line. Every other stderr line is printed again (unknown warnings are not hidden). Marker
`[URDF_CAPSULE]` — once per URDF per process (collision builds per context key · 0.01 s in total).
Entry: the code covered by the batch-augmentation judge fingerprint (collision · judge) calls pin.buildModelFromUrdf
directly (the fingerprint covers value code — folding warnings is not a value). The process entry (the `__main__`
blocks of judge · dexengine.record.augment.coll_gate · dexengine.record.augment.place_gen) calls install() once to swap
pinocchio's function for build_model (the same model — checked bit-identical on 5 URDFs × both sides; forked workers
inherit it). Marker line `[URDF_CAPSULE] entry`. Callers outside the judge call build_model directly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

MARK = "[URDF_CAPSULE]"
PAT = "Could not parse collision element"
TAIL = "urdf_parser/src/link.cpp"  # second and third line of the same warning ('at line … in …/link.cpp')
_SAID: set[str] = set()  # one line per URDF per process
_PIN: list = []  # pinocchio.buildModelFromUrdf before install() swapped it


def install() -> None:
    """Once at the process entry: swap pinocchio.buildModelFromUrdf for build_model (later calls do nothing)."""
    import pinocchio as pin

    if not _PIN:
        _PIN.append(pin.buildModelFromUrdf)
        pin.buildModelFromUrdf = build_model
        print(
            f"{MARK} entry — pinocchio.buildModelFromUrdf → urdf_load.build_model (same model · warnings in one line)",
            file=sys.stderr,
            flush=True,
        )


def build_model(urdf, *args):
    """The same model as pin.buildModelFromUrdf(str(urdf), *args) — fd 2 is captured briefly and only the capsule
    warnings are folded into one line."""
    import pinocchio as pin

    sys.stderr.flush()
    with tempfile.TemporaryFile() as tf:
        keep = os.dup(2)
        os.dup2(tf.fileno(), 2)
        try:
            model = (_PIN[0] if _PIN else pin.buildModelFromUrdf)(str(urdf), *args)
        finally:
            os.dup2(keep, 2)
            os.close(keep)
        tf.seek(0)
        text = tf.read().decode(errors="replace")
    n = text.count(PAT)
    other = [ln for ln in text.splitlines() if ln.strip() and PAT not in ln and TAIL not in ln]
    if n and str(urdf) not in _SAID:
        _SAID.add(str(urdf))
        print(
            f"{MARK} {Path(urdf).name}: urdfdom skipped {n} '<capsule>' collision elements — harmless (pinocchio "
            "colliders unused · capsules come from the XML (collision.geom) · judge colliders from sim_collider)",
            file=sys.stderr,
            flush=True,
        )
    if other:
        sys.stderr.write("\n".join(other) + "\n")
        sys.stderr.flush()
    return model
