"""Record render track points (track_points: hand links and object centres of the state written per step) and their
cache (in memory, inside a reused Kit worker).

Cache (always on): the same worker process re-measured the same trajectory for every session of the same clip
(original, D, four random, augment variants; per trajectory median 0.75, mean 1.72 s over 1409 render_clip lines);
now it is measured once. Key = sha1 of the bytes of the written state array (st_r after the body is applied) · clip ·
step count · sides of the record meta · force channels · binding (entity and body names) · head view on. A hit
returns a deep copy of the stored value; an oracle re-measures the first and last steps and checks the bits match
(else dies). The check writes the last step last, so the scene ends where a full measurement leaves it. Up to
MAX_KEEP entries are kept (least recently used dropped first; original and variant trajectory names differ, so
clearing on a name change would re-measure the original every session, as measured on the GPU server).
Markers: '[TRACK_CACHE] fill …' · '[TRACK_CACHE] hit … checked 2 steps equal' (no reader parses them).
"""

from __future__ import annotations

import copy
import hashlib
import json

from dexengine.record.rollout import recorder as REC  # owner of the force channel kind (old records included)

MARK = "[TRACK_CACHE]"
_MEM: dict = {}  # key -> (groups, per, sides | None, trajectory name) · insertion/use order (LRU)
MAX_KEEP = 16  # one trajectory ≈ 1 MB (steps x hand links x 3): room for one clip's original + variants


def track_points(tf, binds, write, n: int, sides: dict | None = None) -> tuple[dict, dict]:
    """Per step (the written state as it is, no physics) world positions of hand links and object centres -> the point
    clouds of render.framing.
    Hand links = body names of the record channel force_total (unfiltered contact sensor; side = sensor name head; the
    old record 'force' means the same, rollout.recorder.force_kind); each must be in exactly one of the robot
    articulations the trajectory writes (binds) · active hands = record meta sides · objects = object* of binds.
    all = every hand link + objects · core = per-step link centre of the active hands + objects.
    Second return = per step {hand: every active hand link, object: object centres} (occlusion measurement).
    sides (filled when given) = per side (idle hand included) {links: link names, pts: per step [N, 3]} for the
    hand-to-head image marking (render.visibility)."""
    import numpy as np

    side_links: dict = {}
    for c in tf.meta["env"]["collect"]["channels"]:
        if REC.force_kind(c) != REC.KIND_TOTAL:
            continue
        side = c["name"].split("/", 1)[1].split("_", 1)[0]
        if side not in ("right", "left"):
            raise SystemExit(f"[COLLECT_FRAME] force channel {c['name']}: side unknown")
        side_links.setdefault(side, set()).update(c["names"])
    arts = [b.asset for b in binds if b.kind == "artic" and not b.name.startswith("object")]
    objs = [b.asset for b in binds if b.name.startswith("object")]
    idx = {
        s: [(a, [a.body_names.index(b) for b in sorted(ln) if b in a.body_names]) for a in arts]
        for s, ln in side_links.items()
    }
    for s, ln in side_links.items():
        got = sorted(b for a in arts for b in a.body_names if b in ln)
        if got != sorted(ln):
            raise SystemExit(
                f"[COLLECT_FRAME] {s} hand links {len(ln)} != {len(got)} found in the robot articulations "
                f"(missing {sorted(set(ln) - set(got))[:4]} · articulations "
                f"{[len(a.body_names) for a in arts]})"
            )
    active = [s["side"] for s in tf.meta["sides"]]
    hands, core, obj, per = [], [], [], {"hand": [], "object": []}
    for t in range(n):
        write(t)
        for a in (*arts, *objs):
            a.update(1e-6)  # state written without a physics step -> reread the buffers from PhysX (as Rig.pose)
        hp = []
        for s in side_links:
            P = np.concatenate([a.data.body_pos_w[0, i].detach().cpu().numpy() for a, i in idx[s] if i])
            hands.append(P)
            if sides is not None:
                sd = sides.setdefault(s, {"links": [a.body_names[j] for a, i in idx[s] for j in i], "pts": []})
                sd["pts"].append(P.astype(float))
            if s in active:
                core.append(P.mean(0, keepdims=True))
                hp.append(P.astype(float))
        obj += [o.data.root_pos_w[0:1].detach().cpu().numpy() for o in objs]
        per["hand"].append(np.concatenate(hp) if hp else np.zeros((0, 3)))
        per["object"].append(np.concatenate(obj[len(obj) - len(objs) :]).astype(float) if objs else np.zeros((0, 3)))
    H, Hc = np.concatenate(hands).astype(float), np.concatenate(core).astype(float)
    ob = np.concatenate(obj).astype(float) if obj else np.zeros((0, 3))
    return {"all": np.concatenate([H, ob]), "core": np.concatenate([Hc, ob]), "hand": Hc, "object": ob}, per


def key(tf, st, clip: str, n: int, binds, hv: bool) -> str:
    import numpy as np

    sig = {
        "clip": clip,
        "n": int(n),
        "hv": bool(hv),
        "sides": tf.meta["sides"],
        "ch": tf.meta["env"]["collect"]["channels"],
        "binds": [(b.name, b.kind, list(getattr(b.asset, "body_names", None) or [])) for b in binds],
    }
    a = np.ascontiguousarray(st)
    h = hashlib.sha1(f"{a.dtype}{a.shape}".encode() + a.tobytes()).hexdigest()
    return h + ":" + hashlib.sha1(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()


def _same(a, b) -> bool:
    import numpy as np

    return a.shape == b.shape and bool(np.array_equal(a, b))


def tracked(tf, binds, write, n: int, sides: dict | None, st, clip: str) -> tuple[dict, dict]:
    """track_points as is (off, first time) or the cached value (hit, after the first/last step bit check). sides is
    filled in place."""
    k = key(tf, st, clip, n, binds, sides is not None)
    got = _MEM.pop(k, None)
    if got is None:
        while len(_MEM) >= MAX_KEEP:
            _MEM.pop(next(iter(_MEM)))
        groups, per = track_points(tf, binds, write, n, sides)
        _MEM[k] = (copy.deepcopy(groups), copy.deepcopy(per), copy.deepcopy(sides), clip)
        print(f"{MARK} fill · {clip} · steps {n} · key {k[:12]} · stored in this worker {len(_MEM)}", flush=True)
        return groups, per
    g0, p0, s0, _c = got
    pick = [0, n - 1]
    chk = {} if s0 is not None else None
    _g, p1 = track_points(tf, binds, lambda t: write(pick[t]), len(pick), chk)
    bad = [
        f"{w} step {pick[j]}"
        for w in ("hand", "object")
        for j in range(len(pick))
        if not _same(p1[w][j], p0[w][pick[j]])
    ]
    if s0 is not None:
        bad += [
            f"head view {s} step {pick[j]}"
            for s in s0
            for j in range(len(pick))
            if not _same(chk[s]["pts"][j], s0[s]["pts"][pick[j]])
        ]
    if bad:
        raise SystemExit(f"{MARK} {clip}: cached value != re-measured value: {bad[:4]} (key {k[:12]})")
    _MEM[k] = got  # hit = most recently used (to the LRU end)
    if sides is not None:
        sides.update(copy.deepcopy(s0))
    print(f"{MARK} hit · {clip} · steps {n} · key {k[:12]} · checked {len(pick)} steps equal", flush=True)
    return copy.deepcopy(g0), copy.deepcopy(p0)
