"""The bundling rule — object/family packing of ONE run pool (pure functions, no data, no randomness).

train_set/select.py (the bundle generator) calls it per run-key pool (it reads the object keys and the shape extents
from the data) and stamps PACK_RULE into every bundle meta.

The rule:
  pool     one run key (ledger axis × run group) — never across datasets.
  key      the object key per clip (object_shape.target_object), a rare two-object key re-keyed (`rekey`).
  family   the shape family of a key (`family_of`: PCA extents of its surface points → rod/flat/compact × size band,
           FAM_SIZE_EDGES). Similar = same family. A key that names no mesh (TACO `clip` keys) is its own family.
  bundles  ceil(pool / n), primary targets balanced; every bundle (primaries + dups) <= n. A pool of <= n clips is
           one bundle. When the rule finds no room at that count (`NoRoom` — little slack), one more bundle, up to
           NB_EXTRA_MAX more, each printed `[PACK]`. No leftover bundle is ever made of odd clips — the odd clips
           are placed with their family.
  1 rigid  whole units, largest first: family chunks of small keys (< CO_MIN clips each; `_chunks` — a chunk reaches
           CO_MIN when its family can) and big keys too short to split (CO_MIN..2*CO_MIN-1). A chunk short of CO_MIN
           goes where its family already makes CO_MIN (`_place_whole`).
  2 fill   splittable keys (>= 2*CO_MIN) fill the target room bundle by bundle (`_fill`, `_piece`): round robin, a key
           whose family is short of companions in the bundle first, pieces near `block`, a key never left with
           1..CO_MIN-1 clips in a bundle (block minimum); what does not fit goes to a bundle holding the key, room
           gathered by moving single clips of over-full keys (`_place_big`, `_make_room`).
  3 dups   a small key still short of CO_MIN same-or-similar clips in its bundle borrows clips of its family from the
           other bundles of the pool (nearest shape first) — only into the room below n, only when that reaches
           CO_MIN, never a clip twice in one bundle. A dup is recorded with its primary bundle (`origin`).
  check    `violations` — the post-conditions over the result (primaries = pool · block minimum · small keys whole ·
           dup origins · size); `pack` dies on any.
"""

from __future__ import annotations

import bisect
import collections
import math
from collections import defaultdict

#: The packing rule's version — stamped into every bundle meta and campaign manifest. Bump it when `pack` changes.
PACK_RULE = "object-family-v2"
#: A two-object key (`A+B`) with fewer clips than this in its pool is re-keyed to its member object with the most
#: clips in the pool (so a rare pairing joins its object's block instead of standing alone).
REKEY_MIN = 4
#: Same-or-similar companions wanted per clip in one run (the same object at least 4 times per run where possible).
#: Also the block minimum of a big key.
CO_MIN = 4
#: Shape family bins over the PCA extents a >= b >= c (metres) of an object's surface points
#: (meshes/<obj>/surface_points.npz). Size bands of ×1.5 on the longest extent a; shape: b/a < FAM_THIN → rod,
#: else c/b < FAM_THIN → flat, else compact. Measured over data/<ds>/meshes (median a: hocap 0.19 · oakink2 0.14 ·
#: grab 0.125 · hrdexdb 0.17 · hot3d 0.15 · taco 0.23 · arctic 0.23 m; b/a 5–95 %: 0.09–1.0; c/b 0.06–1.0). Chosen
#: over ×2 bands (0.12, 0.24) and an object-name rule by measurement: ×2 bands raise the same-or-similar share by
#: 0.6 pp but pool a third more objects into the largest family; a name rule folds coded names (oakink2 o02_…,
#: hocap G07_…) into one family.
FAM_SIZE_EDGES = (0.08, 0.12, 0.18, 0.27, 0.40)
FAM_THIN = 0.4
#: A pool the rule cannot pack in ceil(pool / n) bundles (little slack — `NoRoom`) gets one more bundle, up to this
#: many more; each step prints `[PACK] … bundles a → a+1`. Past it, pack dies.
NB_EXTRA_MAX = 2


class NoRoom(ValueError):
    """No room at this bundle count for a rigid unit (`_place_whole`) or a block minimum (`_place_big`) — the one
    failure `pack` answers with one more bundle; every other failure dies at once."""


def pca_extents(points) -> tuple[float, float, float]:
    """Sorted extents (a >= b >= c) of a point cloud along its principal axes — orientation-free."""
    import numpy as np

    p = np.asarray(points, dtype=np.float64)
    p = p - p.mean(0)
    _w, v = np.linalg.eigh(np.cov(p.T))
    q = p @ v
    e = sorted((float(x) for x in q.max(0) - q.min(0)), reverse=True)
    return e[0], e[1], e[2]


def shape_family(ext: tuple[float, float, float]) -> str:
    """`<rod|flat|compact>-<size band>` of sorted extents (FAM_SIZE_EDGES in metres · FAM_THIN), e.g. `rod-12-24cm`."""
    a, b, c = ext
    if not a > 0:
        raise ValueError(f"shape family: object of size 0, extents={ext}")
    i = bisect.bisect_right(FAM_SIZE_EDGES, a)
    lo = round(100 * FAM_SIZE_EDGES[i - 1]) if i else 0
    band = f"{lo}-{round(100 * FAM_SIZE_EDGES[i])}cm" if i < len(FAM_SIZE_EDGES) else f"{lo}cm+"
    shape = "rod" if b / a < FAM_THIN else "flat" if (b <= 0 or c / b < FAM_THIN) else "compact"
    return f"{shape}-{band}"


def family_of(key: str, ext_of) -> str:
    """The family of an object key: the shape families of its member objects (`A+B` → both, sorted, joined by '+').
    `ext_of(member)` gives a member's extents, or None when the key does not name objects (a `clip` key) — then the
    key is its own family."""
    fams = set()
    for m in key.split("+"):
        e = ext_of(m)
        if e is None:
            return f"key:{key}"
        fams.add(shape_family(e))
    return "+".join(sorted(fams))


def rekey(keys: dict[str, str]) -> dict[str, str]:
    """Object key per clip inside ONE pool: a two-object key with fewer than REKEY_MIN clips in the pool is re-keyed
    to its member with the most clips in the pool (ties → name order); everything else keeps its key."""
    whole = collections.Counter(keys.values())
    member = collections.Counter(o for k in keys.values() for o in set(k.split("+")))
    out = {}
    for c, k in keys.items():
        if "+" not in k or whole[k] >= REKEY_MIN:
            out[c] = k
        else:
            out[c] = max(k.split("+"), key=lambda o: (member[o], o))
    return out


class Bundle:
    """One bundle under construction: primary clips, dups {clip: origin bundle index}, key and family counts over
    both (a dup is a companion like any other clip of the run)."""

    def __init__(self, target: int, small_keys: set):
        self.target, self.prim, self.dups, self.small_keys = target, [], {}, small_keys
        self.kc, self.fc = collections.Counter(), collections.Counter()
        self.small: set = set()  # the small keys (pool count < CO_MIN) primary here

    def size(self) -> int:
        return len(self.prim) + len(self.dups)

    def add(self, clips, keys, fam, origin=None) -> None:
        for c in clips:
            if origin is None:
                self.prim.append(c)
                if keys[c] in self.small_keys:
                    self.small.add(keys[c])
            else:
                self.dups[c] = origin
            self.kc[keys[c]] += 1
            self.fc[fam[keys[c]]] += 1


def _chunks(small: dict[str, list[str]], fam: dict[str, str]) -> list[list[str]]:
    """Small keys (fewer than CO_MIN clips each) as family chunks of whole keys: per family (name order), keys by
    size then name, cut as soon as a chunk reaches CO_MIN clips (a similar group on its own); a last chunk short of
    CO_MIN joins the one before it (a family whose small keys total < CO_MIN stays one chunk)."""
    by: dict[str, list[str]] = defaultdict(list)
    for k in small:
        by[fam[k]].append(k)
    out = []
    for f in sorted(by):
        cur: list[list[str]] = [[]]
        for k in sorted(by[f], key=lambda k: (-len(small[k]), k)):
            if len(cur[-1]) >= CO_MIN:
                cur.append([])
            cur[-1] += small[k]
        if len(cur) > 1 and len(cur[-1]) < CO_MIN:
            last = cur.pop()
            cur[-1] += last
        out += cur
    return out


def _needy(b: Bundle, f: str, fam) -> bool:
    """The bundle holds a small key of family `f` that is still short of CO_MIN similar clips there."""
    return b.fc[f] < CO_MIN and any(fam[k] == f for k in b.small)


def _place_whole(bs: list[Bundle], unit: list[str], keys, fam, n: int) -> None:
    """A rigid unit (a family chunk of small keys, or a big key too short to split into two >= CO_MIN parts) goes
    whole: target room first, then — for a chunk short of CO_MIN — a bundle where its family already makes it
    CO_MIN, then a bundle whose small keys of the unit's family are short of companions, then most target room, then
    index; when no target room fits it, the most hard room (n). No hard room → dies (never split)."""
    s, f = len(unit), fam[keys[unit[0]]]
    fits = [j for j in range(len(bs)) if n - bs[j].size() >= s]
    if not fits:
        raise NoRoom(
            f"[pack] no room for {sorted({keys[c] for c in unit})}({s}) as a whole: "
            f"bundle sizes {[b.size() for b in bs]} / {n}"
        )
    j = min(
        fits,
        key=lambda j: (
            bs[j].target - bs[j].size() < s,
            s < CO_MIN and bs[j].fc[f] + s < CO_MIN,
            not _needy(bs[j], f, fam),
            -(bs[j].target - bs[j].size()),
            j,
        ),
    )
    bs[j].add(unit, keys, fam)


def _piece(q: int, room: int, held: int, block: int) -> int:
    """How many clips of a splittable key (q left) go into a bundle (room left, `held` of the key already there):
    the piece keeps the bundle's count of the key >= CO_MIN, leaves the key 0 or >= CO_MIN, and — when it can —
    leaves the room 0 or >= CO_MIN (else a crumb a held key may fill). Nearest to `block`. 0 = none fits."""
    lo, hi = max(1, CO_MIN - held), min(q, room)
    if hi < lo:
        return 0
    want = min(block, hi)
    cands = sorted(range(lo, hi + 1), key=lambda p: (abs(p - want), p))
    for strict in (True, False):
        for p in cands:
            if q - p not in range(1, CO_MIN) and (not strict or room - p not in range(1, CO_MIN)):
                return p
    return 0


def _fill(bs: list[Bundle], flex: dict[str, list[str]], keys, fam, n: int, block: int) -> dict[str, list[str]]:
    """Splittable keys (>= 2*CO_MIN clips) fill each bundle's target room, bundle by bundle: keys round robin (largest
    first), a key whose family is short of companions in the bundle first, a key the bundle does not hold yet next
    (several objects per bundle); pieces from `_piece`. A key's clips are taken in i::m interleaved order (m =
    ceil(len/block) — no time/segment bias inside a piece). Returns what could not be placed."""
    rem = {}
    for k, cs in flex.items():
        m = max(1, math.ceil(len(cs) / block))
        rem[k] = [c for i in range(m) for c in cs[i::m]]
    order = sorted(flex, key=lambda k: (-len(flex[k]), k))
    ptr = 0
    for b in bs:
        while True:
            room = b.target - b.size()
            live = [order[(ptr + i) % len(order)] for i in range(len(order)) if rem[order[(ptr + i) % len(order)]]]
            if room <= 0 or not live:
                break
            live.sort(key=lambda k: (not _needy(b, fam[k], fam), k in b.kc))  # stable: round robin inside
            for k in live:
                p = _piece(len(rem[k]), room, b.kc[k], block)
                if p:
                    b.add(rem[k][:p], keys, fam)
                    rem[k] = rem[k][p:]
                    ptr = (order.index(k) + 1) % len(order)
                    break
            else:
                break  # a crumb no key can fill — the bundle ends below target
    return {k: v for k, v in rem.items() if v}


def _place_big(bs: list[Bundle], blk: list[str], keys, fam, n: int) -> list[str]:
    """Put (part of) one block of a big key (>= CO_MIN clips in the pool) — returns the rest (back to the queue head).
    Bundle order: target room left first · then a bundle whose small keys of this family are still short of CO_MIN
    similar clips (it lends them companions) · then one not yet holding the key · then most target room · index.
    Cuts tried in a bundle: whole in the target room · the target room · whole in the hard room (n) · the hard room.
    A cut is valid when the bundle then holds >= CO_MIN of the key (block minimum — a piece may join the key's piece
    already there) and the rest is 0, >= CO_MIN, or fits (room and minimum) another bundle already holding the key."""
    k, s = keys[blk[0]], len(blk)
    f = fam[k]

    def rest_ok(j: int, r: int) -> bool:
        return (
            r == 0
            or r >= CO_MIN
            or any(
                i != j and bs[i].kc[k] and bs[i].kc[k] + r >= CO_MIN and n - bs[i].size() >= r for i in range(len(bs))
            )
        )

    order = sorted(
        range(len(bs)),
        key=lambda j: (
            bs[j].target - bs[j].size() <= 0,
            not _needy(bs[j], f, fam),
            k in bs[j].kc,
            -(bs[j].target - bs[j].size()),
            j,
        ),
    )
    for j in order:
        rt, rh = bs[j].target - bs[j].size(), n - bs[j].size()
        for cut in (s if s <= rt else 0, min(rt, s - 1), s if s <= rh else 0, min(rh, s - 1)):
            if cut > 0 and bs[j].kc[k] + cut >= CO_MIN and rest_ok(j, s - cut):
                bs[j].add(blk[:cut], keys, fam)
                return blk[cut:]
    # No bundle has the room: gather the scattered room into one bundle (`_make_room`) — one holding the key needs
    # room for the whole rest, any other needs max(CO_MIN, rest).
    for j in sorted(range(len(bs)), key=lambda j: (not bs[j].kc[k], -(n - bs[j].size()), j)):
        need = s if bs[j].kc[k] + s >= CO_MIN else max(CO_MIN, s)
        if need == s and _make_room(bs, j, need, keys, fam, n):
            bs[j].add(blk, keys, fam)
            return []
    raise NoRoom(
        f"[pack] no room that keeps the block minimum {CO_MIN}: object {k} piece {s} · bundle sizes "
        f"{[b.size() for b in bs]} / {n} — a pool the rule cannot pack (needs a decision)"
    )


def _make_room(bs: list[Bundle], j: int, need: int, keys, fam, n: int) -> bool:
    """Free room in bundle j by moving single primaries of a key it holds more than CO_MIN of (so it keeps >= CO_MIN)
    to another bundle that already holds that key and has room (last clip by name first; key by most held, then
    name; target bundle by most room, then index). True when j has `need` room."""
    while n - bs[j].size() < need:
        moved = False
        for k in sorted(
            (k for k, m in bs[j].kc.items() if m > CO_MIN and k not in bs[j].small_keys),
            key=lambda k: (-bs[j].kc[k], k),
        ):
            dst = [i for i in range(len(bs)) if i != j and bs[i].kc[k] and n - bs[i].size() >= 1]
            if not dst:
                continue
            i = min(dst, key=lambda i: (-(n - bs[i].size()), i))
            c = max(x for x in bs[j].prim if keys[x] == k)
            bs[j].prim.remove(c)
            bs[j].kc[k] -= 1
            bs[j].fc[fam[k]] -= 1
            bs[i].add([c], keys, fam)
            moved = True
            break
        if not moved:
            return False
    return True


def _borrow(bs: list[Bundle], keys, fam, n: int, near) -> None:
    """Dups: for each bundle (index order) and each small key in it still short of CO_MIN same-or-similar clips
    (cheapest deficit first, then key name), borrow the deficit from its family's clips in the other bundles —
    nearest key shape first (`near(k, other)`), then key name, then clip name — only when the whole deficit fits
    the room below n. A clip is borrowed into a bundle at most once and never where it is primary."""
    home = {c: j for j, b in enumerate(bs) for c in b.prim}
    by_fam: dict[str, list[str]] = defaultdict(list)
    for c in sorted(home):
        by_fam[fam[keys[c]]].append(c)
    cnt = collections.Counter(keys[c] for c in home)
    for j, b in enumerate(bs):
        short = sorted(
            {keys[c] for c in b.prim if cnt[keys[c]] < CO_MIN and b.fc[fam[keys[c]]] < CO_MIN},
            key=lambda k: (CO_MIN - b.fc[fam[k]], k),
        )
        for k in short:
            f = fam[k]
            need = CO_MIN - b.fc[f]
            if need <= 0:
                continue  # an earlier borrow of the same family already covered it
            cand = sorted(
                (c for c in by_fam[f] if home[c] != j and c not in b.dups), key=lambda c: (near(k, keys[c]), keys[c], c)
            )
            if len(cand) < need or n - b.size() < need:
                continue  # cannot reach CO_MIN here — no useless dup
            for c in cand[:need]:
                b.add([c], keys, fam, origin=home[c])


def pack(keys: dict[str, str], fam: dict[str, str], n: int, block: int, near=None) -> list[dict]:
    """Object/family packing of ONE pool — deterministic, no randomness (PACK_RULE; module doc).

    keys = {clip: object key} (already `rekey`ed), fam = {key: family} (`family_of`), near(k1, k2) = a shape distance
    (default: 0 — then key name order). Returns [{"clips": sorted primaries, "dups": {clip: origin bundle index}}]."""
    total = len(keys)
    if total == 0:
        return []
    if block < 2 * CO_MIN - 1:
        raise ValueError(f"[pack] block {block} < {2 * CO_MIN - 1} — big-object blocks would be cut below {CO_MIN}")
    missing = sorted({k for k in keys.values() if k not in fam})
    if missing:
        raise ValueError(f"[pack] object keys without a family {missing[:5]}")
    near = near or (lambda _a, _b: 0.0)
    nb0 = math.ceil(total / n)
    for nb in range(nb0, nb0 + NB_EXTRA_MAX + 1):
        try:
            return _pack_at(keys, fam, n, block, near, nb)
        except NoRoom as e:  # noqa: PERF203 — no room moves on to the next bundle count
            if nb == nb0 + NB_EXTRA_MAX:
                raise
            print(
                f"[PACK] pool of {total} clips: bundles {nb} → {nb + 1} (rule NB_EXTRA_MAX={NB_EXTRA_MAX}) — {e}",
                flush=True,
            )
    raise AssertionError("unreachable")


def _pack_at(keys: dict[str, str], fam: dict[str, str], n: int, block: int, near, nb: int) -> list[dict]:
    """`pack` at a given bundle count nb (targets balanced over nb) — NoRoom when the rule finds no room."""
    total = len(keys)
    by: dict[str, list[str]] = defaultdict(list)
    for c in sorted(keys):
        by[keys[c]].append(c)
    small = {k: cs for k, cs in by.items() if len(cs) < CO_MIN}
    bs = [Bundle(total // nb + (1 if i < total % nb else 0), set(small)) for i in range(nb)]
    rigid = _chunks(small, fam) + [cs for k, cs in by.items() if CO_MIN <= len(cs) < 2 * CO_MIN]
    for unit in sorted(rigid, key=lambda u: (-len(u), u[0])):
        _place_whole(bs, unit, keys, fam, n)
    left = _fill(bs, {k: cs for k, cs in by.items() if len(cs) >= 2 * CO_MIN}, keys, fam, n, block)
    for k in sorted(left, key=lambda k: (-len(left[k]), k)):
        rest = left[k]
        while rest:
            rest = _place_big(bs, rest, keys, fam, n)
    if nb > 1:
        _borrow(bs, keys, fam, n, near)
    out = [{"clips": sorted(b.prim), "dups": dict(sorted(b.dups.items()))} for b in bs]
    bad = violations(keys, out, n)
    if bad:
        raise ValueError(f"[pack] rule violations {bad[:3]}")
    return out


def violations(keys: dict[str, str], bundles: list[dict], n: int) -> list[str]:
    """The packed pool's post-conditions (an oracle over the result, not the heuristic): primaries == the pool, each
    clip primary once · a big key (>= CO_MIN in the pool) has 0 or >= CO_MIN primaries in every bundle · a small key
    is not split · a dup's origin is the bundle where it is primary, it is not primary in its own bundle, no clip
    twice in a bundle · every bundle (primaries + dups) <= n. [] = holds."""
    bad = []
    prim = [c for b in bundles for c in b["clips"]]
    if len(prim) != len(set(prim)) or set(prim) != set(keys):
        bad.append(f"primary clips ≠ pool: primary {len(prim)} (unique {len(set(prim))}) · pool {len(keys)}")
    home = {c: j for j, b in enumerate(bundles) for c in b["clips"]}
    cnt = collections.Counter(keys.values())
    for j, b in enumerate(bundles):
        kc = collections.Counter(keys[c] for c in b["clips"])
        for k, m in kc.items():
            if cnt[k] >= CO_MIN and m < CO_MIN:
                bad.append(f"bundle {j}: big object {k} primary {m} < {CO_MIN} (block minimum)")
            if cnt[k] < CO_MIN and m != cnt[k]:
                bad.append(f"bundle {j}: small object {k} split {m}/{cnt[k]}")
        for c, o in b["dups"].items():
            if c in b["clips"]:
                bad.append(f"bundle {j}: {c} is both a primary clip and a dup")
            elif home.get(c) != o:
                bad.append(f"bundle {j}: dup {c} origin {o} ≠ primary bundle {home.get(c)}")
        if len(b["clips"]) + len(b["dups"]) > n:
            bad.append(f"bundle {j}: size {len(b['clips']) + len(b['dups'])} > {n}")
    return bad
