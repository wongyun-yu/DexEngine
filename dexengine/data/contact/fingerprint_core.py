"""Contact-sidecar input fingerprint: what a sidecar was built from (the base tree's inputs, object poses, surface
samples, generation parts), how two fingerprints compare, and the check a consumer runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dexengine.core import paths as P  # noqa: E402  — owner of the path conventions
from dexengine.core.errors import InputMissingError
from dexengine.data import (
    align as RD,  # noqa: E402  — owner of the scene alignment reference hand direction (fixed value · per-generation values)
)
from dexengine.data.cases.assign_version import (
    ASSIGN_KEY,  # noqa: E402  — key of the assignment version marker (dependency-free owner)
)
from dexengine.data.contact.fingerprint_check import FP_SCHEMA, LAYER_ALL, LAYER_BASE, SidecarFingerprintError
from dexengine.data.contact.source import PALM_ALPHA_FILE as _ALPHA_FILE  # noqa: E402
from dexengine.data.oakink2.parts import (
    SUPP_KEY,  # noqa: E402  — key of the supplement copy rule version marker (owner)
)

dl = P.data_layout
#: Version of the contact generation code; owned here (contact.sidecar_compute.VERSION is this value). Bump it when
#: contact.sidecar or the dexengine.data.contact.common (thresholds, smoothing) · dexengine.data.contact.geom (geometry)
#: code it uses changes, or when other dependencies (h2o_frame tilt etc.) change contact. Surface samples are compared
#: by content hash in the fingerprint.
CODE_VERSION = "2026-09-11d+h2o_tilt"
LAYERS = (LAYER_ALL, LAYER_BASE)
FP_KEYS = ("fp_schema", "fp_exact", "fp_pose", "fp_input_id", "fp_built_from", "fp_base_time", "fp_base_adapter")


def palm_alpha() -> float:
    if not _ALPHA_FILE.is_file():
        raise InputMissingError(f"[contact] MANO palm coefficients missing ({_ALPHA_FILE})")
    return float(json.loads(_ALPHA_FILE.read_text())["alpha"])


def layer_of(all_objects: bool) -> str:
    return LAYER_ALL if all_objects else LAYER_BASE


def sidecar_path(root: Path, ds: str, layer: str, clip: str, side: str, spec_dir: str) -> Path:
    """One sidecar file; location resolution is owned by data_layout.sidecar_rel (shared and per-hand locations)."""
    return Path(root) / dl.sidecar_rel(root, layer, ds, clip, side, spec_dir)


def sidecar_is_own(root: Path, ds: str, layer: str, clip: str, side: str, spec_dir: str) -> bool:
    """Whether this hand's file is in a per-hand location; then it is not shared with other hands and there is no
    shared-layer comparison (live_disagreement)."""
    return dl.sidecar_rel(root, layer, ds, clip, side, spec_dir) != dl.sidecar_shared_rel(layer, ds, clip, side)


def surface_path(root: Path, ds: str, obj: dict) -> Path:
    """Surface sample file of the meshes layer: rigid `surface_points.npz` · articulated `surface_parts.npz`."""
    name = "surface_parts.npz" if obj.get("articulation") else "surface_points.npz"
    return Path(root) / dl.render("meshes", ds) / str(obj["obj_id"]) / name


def objects_for(ti: dict, side: str, layer: str) -> list[dict]:
    """Objects one (clip, side) sidecar holds; producers and consumers decide with **the same function**.

    all  = every object of the tree when that side exists (roles are not looked at, to avoid the cycle; see
    stack_objects)
    base = that side's main object (role main · side ∈ {side, both}). More than one cannot fit in one file and fails
           (the last one is never picked silently).
    """
    if side not in (ti.get("hands") or []):
        return []
    objs = list(ti.get("objects") or [])
    if layer == LAYER_ALL:
        return objs
    if layer != LAYER_BASE:
        raise ValueError(f"layer {layer!r} — one of {LAYERS}")
    m = [o for o in objs if o.get("role") == "main" and o.get("side") in (side, "both")]
    if len(m) > 1:
        names = [o["name"] for o in m]
        raise ValueError(f"{side}: {len(m)} main objects {names} — they do not fit in one sidecar (run D23 first)")
    return m


def sides_for(ti: dict, layer: str) -> list[str]:
    """Sidecar sides expected for the tree: only sides with objects to hold."""
    return [s for s in (ti.get("hands") or []) if objects_for(ti, s, layer)]


def pose_witness(kin: np.ndarray, objs: list[dict]) -> np.ndarray:
    T = len(kin)
    W = np.full((T, len(objs), 8), np.nan, dtype=np.float64)
    for i, o in enumerate(objs):
        s = int(o["qpos_start"])
        W[:, i, :7] = kin[:, s : s + 7]
        art = o.get("articulation")
        if art:
            W[:, i, 7] = kin[:, int(art["qpos_index"])]
    return W


def _sha_arrays(items) -> str:
    h = hashlib.sha256()
    for k, a in items:
        a = np.ascontiguousarray(np.asarray(a))
        for part in (str(k).encode(), str(a.dtype).encode(), str(a.shape).encode(), a.tobytes()):
            h.update(part)
    return h.hexdigest()


_SURF_CACHE: dict = {}


def surface_sha(root: Path, ds: str, obj: dict) -> str:
    """Hash of the surface sample **content** (the arrays, not the file bytes, so np.savez zip times never mix in).
    'missing' without it."""
    p = surface_path(root, ds, obj)
    try:
        st = p.stat()
    except OSError:
        return "missing"
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _SURF_CACHE:
        with np.load(p, allow_pickle=False) as z:
            _SURF_CACHE[key] = _sha_arrays((k, z[k]) for k in sorted(z.files))[:24]
    return _SURF_CACHE[key]


def base_time_of(ti: dict) -> str:
    return str((ti.get("provenance") or {}).get("time") or "")


#: Per layer, the **base step versions that enter the generation**: keys of task_info.rules. Only steps that change the
#: layer's expected objects or poses: 2b D1g (part pair → one articulation · object list) · 2c D25 (joint values · pose
#: witnesses) · for the main-object layer 3 assignment · 4 roles (D23 · D24). Versions that do not change expected
#: content, such as quality (4b D12c), are left out; measured (G15, three hands' bases from the same code): they are
#: present in some hands and absent in others as processing progresses, which splits one generation into two and
#: hides mismatches between new bases behind generation replacement.
GEN_RULES = {LAYER_ALL: ("D1g", "D25", SUPP_KEY), LAYER_BASE: ("D1g", "D25", SUPP_KEY, ASSIGN_KEY, "D23", "D24")}


def _rule_version(v) -> str:
    return str(v.get("version") or "") if isinstance(v, dict) else str(v)


#: Generation of the source → +z-up coordinate rule, written by the reader into the top-level task_info
#: `source_frame.version` (owner dexengine/data/readers/source_frames.py FRAME_VERSION; carried unchanged reader →
#: retarget
#: → base). A changed coordinate rule moves the dataset's hand and object poses as a whole, so it enters the generation
#: of
#: **both layers**. Trees without the field (readers before the rule, datasets without a rule) do not get this part, so
#: bases before and after a repair differ in generation.
GEN_FRAME_KEY = "source_frame"
#: Reference hand direction of scene alignment: the value the tree wrote itself (frame.hand_dir_deg_taco) becomes the
#: generation part `REF=<value>`. A different reference direction rotates the dataset's hand and object poses as a
#: whole,
#: so it enters the generation of **both layers**. Current version = the value the dataset must use now
#: (align.for_dataset: the generation declaration's value; without one the constant, or a failure for
#: align.BUILT_OFF_REF). Trees without the
#: value have no such part.
GEN_REF_KEY = "REF"
#: Lever-pinned rigid bodies (rules.D1g[.multi].static_artic.lever_rigid) build a **different base** under the same D1g
#: version than the static-articulation skip; that decision goes into the generation part `LEVER_PIN=<sorted fixed
#: parts>` (so one tag never stands for two states). Trees without lever pins have no part (old generation text
#: unchanged). Version order: the side **with the part is current** (lever pins came after the static skip); the side
#: without it is the old skip state or a base before 2b. Both present with different values = order unknown (conflict).
GEN_LEVER_KEY = "LEVER_PIN"
#: Static articulated objects (rules.D1g[.multi].static_artic: rigid joint · out-of-band skip) carry **different
#: objects** under the same D1g version than old bases that kept the articulation. That generation difference goes into
#: the generation part `STATIC_ARTIC=<sorted joint assets>`, the same form as LEVER_PIN (the side with it is current ·
#: both present with different values = order unknown). Trees without static_artic have no part (old generation text
#: unchanged; fingerprint comparison does not look at generations).
GEN_STATIC_KEY = "STATIC_ARTIC"
#: Parts whose presence means current (version order from presence, not from an owner constant)
GEN_PRESENCE_KEYS = (GEN_STATIC_KEY, GEN_LEVER_KEY)


def _lever_part(rules: dict) -> str:
    rec = rules.get("D1g")
    if not isinstance(rec, dict):
        return ""
    return ",".join(
        sorted(
            {
                str(x)
                for r in [rec, *(rec.get("multi") or [])]
                for x in ((r.get("static_artic") or {}).get("lever_rigid") or [])
            }
        )
    )


def _static_part(rules: dict) -> str:
    rec = rules.get("D1g")
    if not isinstance(rec, dict):
        return ""
    recs = [r for r in [rec, *(rec.get("multi") or [])] if isinstance(r, dict) and r.get("static_artic")]
    return ",".join(sorted({str(r["static_artic"].get("joint_asset") or "?") for r in recs}))


def generation_of(ti: dict, layer: str) -> str:
    """Generation of a base layer = producer version (`provenance.adapter_version`, written by the tree itself) +
    GEN_RULES[layer] versions + coordinate rule version (GEN_FRAME_KEY). Without a producer version "" (generation
    unknown; disagreement_kind counts it as gen_unknown). An unknown layer name fails."""
    av = str((ti.get("provenance") or {}).get("adapter_version") or "")
    if not av:
        return ""
    rules = ti.get("rules") or {}
    parts = [av] + [f"{k}={_rule_version(rules[k])}" for k in GEN_RULES[layer] if k in rules]
    if static := _static_part(rules):
        parts.append(f"{GEN_STATIC_KEY}={static}")
    if lever := _lever_part(rules):
        parts.append(f"{GEN_LEVER_KEY}={lever}")
    frame = (ti.get(GEN_FRAME_KEY) or {}).get("version")
    ref = RD.part(ti)
    return "|".join(parts + ([f"FRAME={frame}"] if frame else []) + ([f"{GEN_REF_KEY}={ref}"] if ref else []))


#: Kinds of shared-layer mismatch between two hands, decided by disagreement_kind alone and read by both the producer
#: (overwrite) and the check (--gen-check).
SAME_GEN, GEN_CHANGE, GEN_BEHIND, GEN_UNKNOWN = "same_gen", "gen_change", "gen_behind", "gen_unknown"
#: Kinds the check counts as defects (mismatch between new bases · unclassifiable); the rest (generation replaced ·
#: behind) are records of generation differences.
GEN_DEFECT = (SAME_GEN, GEN_UNKNOWN)
GEN_LABEL = {
    SAME_GEN: "same-generation mismatch (between new bases)",
    GEN_CHANGE: "this hand is newer",
    GEN_BEHIND: "that hand is newer",
    GEN_UNKNOWN: "generation unknown",
}


def disagreement_kind(mine_gen: str, other_gen: str, newer: str) -> str:
    """Kind of one mismatch (the shared-layer conflict rule in the module docstring). The producer overwrites only
    gen_change.
    newer = answer of newer_by_generation (mine · other · anything else = order unknown)."""
    if not mine_gen or not other_gen:
        return GEN_UNKNOWN
    if mine_gen == other_gen:
        return SAME_GEN
    return GEN_CHANGE if newer == "mine" else GEN_BEHIND if newer == "other" else GEN_UNKNOWN


def gen_parts(gen: str) -> dict:
    """Generation string (generation_of) → {part: version}; the first part = producer version (key 'adapter'), the rest
    `key=version`."""
    if not gen:
        return {}
    head, *rest = gen.split("|")
    return {"adapter": head, **dict(r.partition("=")[::2] for r in rest)}


def current_part(key: str, spec_dir: str, ref: str = "") -> str:
    """**Current version** of one generation part, read from the owner constant of the step that writes it (imported
    late to measure only the other parts and avoid circular imports). Parts outside the registry raise KeyError (no
    invented order; the caller counts it as incomparable). ref = the dataset's current reference direction (ds_ref); the
    current version of part REF differs per dataset, so the caller passes it (KeyError without it)."""
    if key == GEN_REF_KEY:
        if not ref:
            raise KeyError(
                "the current version of generation part REF is the dataset's reference direction — not "
                "passed by the caller (current_part)"
            )
        return ref
    if key == "adapter":
        from dexengine.data.base_layer.common import ADAPTER_VERSION as v  # noqa: PLC0415
    elif key == "D1g":
        from dexengine.data.artic.convert_version import D1G_VERSION as v  # noqa: PLC0415
    elif key == "D25":
        from dexengine.data.artic.arctic_align import VERSION as v  # noqa: PLC0415
    elif key == SUPP_KEY:
        from dexengine.data.oakink2.parts import SUPP_VERSION as v  # noqa: PLC0415
    elif key == ASSIGN_KEY:
        from dexengine.data.cases.assign_version import hand_version  # noqa: PLC0415

        v = hand_version(spec_dir)
    elif key == "D23":
        from dexengine.data.object_roles import VERSION as v  # noqa: PLC0415
    elif key == "D24":
        from dexengine.data.desk_penetration import D24_VERSION as v  # noqa: PLC0415
    elif key == "FRAME":
        from dexengine.data.readers import source_frames as _sf  # noqa: PLC0415

        v = _sf.FRAME_VERSION  # owner of the coordinate rule (reader tree; the loader is paths alone)
    else:
        raise KeyError(f"generation part {key!r} has no owner of its current version (current_part)")
    return str(v)


def newer_by_generation(
    mine_gen: str, other_gen: str, mine_spec: str, other_spec: str, ref: str = ""
) -> tuple[str, str]:
    """Which side is the newer generation (version order, see the module docstring) → (mine | other | same | unknown,
    reason). ref = current_part's ref."""
    pm, po = gen_parts(mine_gen), gen_parts(other_gen)
    diff = sorted(k for k in pm.keys() | po.keys() if pm.get(k) != po.get(k))
    if not diff:
        return "same", ""

    def now(
        p: dict, k: str, spec: str
    ) -> bool:  # STATIC_ARTIC · LEVER_PIN = current when present · the rest = owner constants
        return p.get(k) is not None if k in GEN_PRESENCE_KEYS else p.get(k) == current_part(k, spec, ref)

    for k in GEN_PRESENCE_KEYS:
        if k in diff and pm.get(k) and po.get(k):
            return "unknown", f"part {k} present on both sides with different values ({pm[k]} ≠ {po[k]})"
    try:
        mc = all(now(pm, k, mine_spec) for k in diff)
        oc = all(now(po, k, other_spec) for k in diff)
    except (KeyError, ImportError) as e:
        return "unknown", f"version order cannot be decided: {e}"
    if mc != oc:
        return (
            "mine" if mc else "other"
        ), f"differing parts {diff}: current version = {'this hand' if mc else 'that hand'}"
    return "unknown", f"differing parts {diff}: not exactly one side is current (both old or mixed)"


class TreeInputs:
    """Inputs of one base tree (task_info · object poses · MANO source), read once, giving expected fingerprints per
    layer and side."""

    def __init__(self, root: Path, ds: str, clip: str, ti: dict, kin: np.ndarray, spec_dir: str):
        self.root, self.ds, self.clip = Path(root), ds, clip
        self.ti, self.kin, self.spec_dir = ti, kin, spec_dir
        self._mano: dict | None = None
        self._tilt: str | None | bool = False

    @classmethod
    def load(cls, root: Path, ds: str, tree: Path, spec_dir: str) -> TreeInputs:
        tree = Path(tree)
        ti = json.loads((tree / "task_info.json").read_text())
        with np.load(tree / "0/trajectory_kinematic.npz") as z:
            kin = z["qpos"]
        return cls(root, ds, tree.name, ti, kin, spec_dir)

    def mano_raw(self) -> dict:
        """MANO keypoint source (before rotation); the producer (compute_side) applies the h2o rotation to this
        dictionary."""
        if self._mano is None:
            p = self.root / self.ti["provenance"]["mano_dir"] / "0/trajectory_keypoints.npz"
            with np.load(p) as kp:
                self._mano = {k: kp[k] for k in kp.files}
        return self._mano

    def _tilt_id(self):
        # only successful values are cached; caching a failure as None (= no rotation) would let a second call pass
        # silently.
        if self._tilt is False:
            tilt = None
            if self.ds == "h2o":  # D10-e'': input record of the desk plane rotation (owner h2o_frame)
                from dexengine.data.readers import h2o_frame as HF  # noqa: PLC0415

                rec = HF.tilt_transform(self.clip)["plane"]
                keep = {k: rec.get(k) for k in ("version", "n_capture", "pivot_capture")}
                tilt = hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:24]
            self._tilt = tilt
        return self._tilt

    def expected(self, layer: str, side: str, own: bool = True) -> dict:
        """own = the side **building or checking** the shared layer from this tree (default). Then the tree's reference
        direction (layer value) must equal the dataset's generation value, otherwise it fails. When reading another
        hand's
        base for comparison (own=False) it is not checked: that difference is resolved by part REF as a generation
        replacement."""
        if own:
            mine, want = RD.part(self.ti), ds_ref(self.root, self.ds)
            if mine and mine != want:
                raise SidecarFingerprintError(
                    RD.check_tree(self.ds, self.ti, self.root / dl.render("intake", self.ds))
                    or f"{RD.TAG} generation value {want} ↔ layer value {mine}"
                )
        objs = objects_for(self.ti, side, layer)
        if not objs:
            raise ValueError(
                f"{self.clip}/{side}: layer {layer} has no objects to hold — no sidecar is expected for this side"
            )
        kp = self.mano_raw()
        kf, kw = f"qpos_finger_{side}", f"qpos_wrist_{side}"
        if kf not in kp or kw not in kp:
            raise KeyError(f"{self.clip}: MANO keypoints lack {kf}/{kw}")
        sg = self.ti["segment"]
        exact = {
            "schema": FP_SCHEMA,
            "code": {"version": CODE_VERSION, "alpha": (palm_alpha() if layer == LAYER_ALL else None)},
            "layer": layer,
            "side": side,
            "T": int(len(self.kin)),
            "objects": [
                {
                    "name": str(o["name"]),
                    "obj_id": str(o["obj_id"]),
                    "articulated": bool(o.get("articulation")),
                    "surface": surface_sha(self.root, self.ds, o),
                }
                for o in objs
            ],
            "mano": {
                "sha": _sha_arrays(((kf, kp[kf]), (kw, kp[kw])))[:24],
                "start": int(sg["start"]),
                "offset": int((sg.get("frame_map") or {}).get("offset", 0)),
                "yaw_deg": float((self.ti.get("frame") or {}).get("yaw_deg", 0.0)),
                "tilt": self._tilt_id(),
            },
        }
        pv = self.ti.get("provenance") or {}
        return {
            "schema": FP_SCHEMA,
            "exact": exact,
            "pose": pose_witness(self.kin, objs),
            "prov": {
                "built_from": self.spec_dir,
                "base_time": base_time_of(self.ti),
                "base_adapter": str(pv.get("adapter_version") or ""),
                "gen": generation_of(self.ti, layer),
            },
        }


def _exact_json(exact: dict) -> str:
    return json.dumps(exact, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def input_id(fp: dict) -> str:
    """Short name, bit-identical when recomputed from the same base (case → role comparison). Use compare() across
    hands."""
    h = hashlib.sha256(_exact_json(fp["exact"]).encode())
    p = np.ascontiguousarray(fp["pose"], dtype=np.float64)
    h.update(str(p.shape).encode())
    h.update(p.tobytes())
    return h.hexdigest()[:24]


def payload(fp: dict) -> dict:
    """Fingerprint fields for the sidecar npz (FP_KEYS)."""
    return {
        "fp_schema": FP_SCHEMA,
        "fp_exact": _exact_json(fp["exact"]),
        "fp_pose": np.asarray(fp["pose"], np.float64),
        "fp_input_id": input_id(fp),
        "fp_built_from": fp["prov"]["built_from"],
        "fp_base_time": fp["prov"]["base_time"],
        "fp_base_adapter": fp["prov"]["base_adapter"],
    }


_REF_CACHE: dict = {}


def ds_ref(root: Path, ds: str) -> str:
    """Reference direction the dataset must use now (generation part text): dexengine.data.align.for_dataset (the
    declaration's value,
    or without one the constant). A declaration without a value, or a missing declaration of a dataset built under
    another value (align.BUILT_OFF_REF), fails (align.RefDirError)."""
    key = (str(root), ds)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = RD.fmt(RD.for_dataset(ds, Path(root) / dl.render("intake", ds))[0])
    return _REF_CACHE[key]
