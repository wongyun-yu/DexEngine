#!/usr/bin/env python3
"""**Reference hand direction** of scene alignment (D5): single owner of the fixed value, its source and the
per-generation dataset values.

Every dataset's scene is rotated as base = Rz(yaw)·capture, with yaw = reference direction − the dataset's (or clip's)
hand direction. The reference direction is **one named constant** (REF_DEG); a re-measured TACO mean is used only as a
check (oracle), so rebuilding TACO never moves it.

Reading rule (for_dataset):
  the dataset has a current generation declaration (GENERATION) → the declaration's
      `hand_dir_ref_deg` (the value when that generation was built). Rebuilding some clips within the generation gives
      the same value. A missing field fails (no invented value).
  no declaration, dataset in BUILT_OFF_REF (built under another value than REF_DEG) → fails, naming the shipped input
      `generation` (data/<ds>/GENERATION): the constant would rotate that dataset's scenes differently.
  no declaration, any other dataset (built under REF_DEG, or a new dataset) → the constant REF_DEG.
Checks:
  check_tree   the value the tree wrote (frame.hand_dir_deg_taco = layer value) ↔ for_dataset (generation value). The
      reason when they differ.
  oracle       re-measured TACO mean ↔ the constant. Reported when they differ by more than TOL_DEG (the value is not
      changed).
  ship         the declared values ↔ BUILT_OFF_REF: the datasets whose declaration differs from REF_DEG are exactly that
      tuple, or the shipped input is not written.
Marker: [REF_DIR] …
statement() builds the source sentence from SOURCE (printed by `--statement`).
Usage: python3 dexengine/data/align.py --statement | --show | --ship <dir>
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

from dexengine.core.errors import input_missing

TAG = "[REF_DIR]"

#: Reference hand direction (°), a fixed value. Its source is SOURCE; the human-readable sentence is statement() alone
#: (rule cards and audit evidence read that sentence).
REF_DEG = 119.91
#: Where the value came from: the statement copied from the measurement record (TACO INTAKE/hand_dir.json) when it was
#: fixed. Changes together with the value.
SOURCE = {
    "statistic": (
        "the TACO mean hand direction (xy angle of MANO wrist → mean of the five fingertips · circular "
        "mean of the per-clip circular means)"
    ),
    "record": "data/taco/INTAKE/hand_dir.json",
    "measured_at": "2026-10-13T00:17:45+0900",
    "measured_on": "the MANO layer of TACO rebuilt after generation taco_tool_right_20261012 was archived",
    "mano_input": "927bebabce15d69d",
    "n_clips": 2308,
    "n_hands": 4616,
    "spread_deg": 25.36,
    "previous": {
        "deg": 119.79,
        "measured_at": "2026-10-07T04:17:41+0900",
        "measured_on": "the MANO layer of the archived generation taco_tool_right_20261012",
        "mano_input": "94fb45984ad04ca4",
        "n_clips": 2317,
        "n_hands": 4634,
        "spread_deg": 25.51,
    },
    "fixed_at": "2026-10-13",
    "fixed_because": (
        "when it was fixed, the four HOT3D hands and OakInk2 XHand were already built with this value, so "
        "nothing had to be rebuilt"
    ),
}
#: Check threshold (°): reported when the re-measured TACO mean moves further than this from the constant. Basis: one
#: shift when the TACO roster went from 2,317 to 2,308 clips was 0.12°; this is about four times that. 0.5° is 4.4 mm
#: at 50 cm and 2 % of the TACO hand direction spread (25°).
TOL_DEG = 0.5
#: Field name in the generation declaration file (GENERATION): that generation's reference direction.
KEY = "hand_dir_ref_deg"
#: Datasets whose current generation was built under another reference direction than REF_DEG (the previous value,
#: SOURCE["previous"]). For them the declaration is required: without it the constant would rotate every scene by the
#: difference. The other datasets were built under REF_DEG (SOURCE["fixed_because"]; TACO is its source), so the
#: constant gives their value. Names only — the values live in the declarations; ship() checks this tuple against them.
BUILT_OFF_REF = ("arctic", "dexcanvas", "grab", "h2o", "hocap", "hrdexdb")
#: The shipped input that carries the declarations (configs/release/shipped_inputs.tsv, dexengine.core.shipped).
SHIPPED_INPUT = "generation"
#: Field the tree writes itself (task_info.frame): the reference direction used to build that tree (layer value).
TREE_KEY = "hand_dir_deg_taco"


class RefDirError(RuntimeError):
    """The generation value cannot be decided or disagrees with the layer value; never passed silently."""


def _mm_at(deg: float, dist_m: float = 0.5) -> float:
    import math  # noqa: PLC0415

    return 2.0 * dist_m * math.sin(math.radians(abs(deg)) / 2.0) * 1000.0


def statement() -> str:
    """Source sentence, shared by the constant, rule cards and audit evidence (built here only)."""
    s, p = SOURCE, SOURCE["previous"]
    d = round(REF_DEG - p["deg"], 2)
    return (
        f"The reference direction {REF_DEG}° is {s['statistic']}, measured {s['measured_at'][:16].replace('T', ' ')} "
        f"on {s['measured_on']} (fingerprint {s['mano_input']}) over {s['n_clips']:,} clips · {s['n_hands']:,} hands"
        f" (record {s['record']}). The previous value was {p['deg']}°, measured "
        f"{p['measured_at'][:16].replace('T', ' ')} "
        f"on {p['measured_on']} (fingerprint {p['mano_input']}) over {p['n_clips']:,} clips · {p['n_hands']:,} hands. "
        f"Why the new value was fixed ({s['fixed_at']}): {s['fixed_because']} · the difference of {d}° is about "
        f"{_mm_at(d):.0f} mm at 50 cm. Rebuilding TACO does not change this constant — the re-measured mean is only a "
        f"check, reported when it moves more than {TOL_DEG}°. A dataset already built uses its own generation's value."
    )


#: The current generation declaration of a dataset: one file next to its INTAKE layer. First line
#: `generation=<name>`, then `key=value` lines.
GENERATION_FILE = "GENERATION"


def decl_path(intake_dir: Path) -> Path:
    """The generation declaration of the dataset whose INTAKE layer is `intake_dir`."""
    return Path(intake_dir).parent / GENERATION_FILE


def read_decl(f: Path) -> dict | None:
    """A generation declaration -> {key: value}; None when the file is missing. A file without a
    `generation=<name>` line raises ValueError."""
    f = Path(f)
    if not f.is_file():
        return None
    kv = dict(ln.split("=", 1) for ln in f.read_text(encoding="utf-8").splitlines() if "=" in ln)
    if not kv.get("generation"):
        raise ValueError(f"{f}: no `generation=<name>` line -- not a generation declaration")
    return kv


@functools.lru_cache(maxsize=8)
def _shipped_state(datasets_dir: Path) -> str:
    """The declarations under `datasets_dir` (`<root>/data`) as the shipped input `generation`: verified in a release
    root, `site` on an unverified root (dexengine.core.shipped) — once per process and folder."""
    from dexengine.core import shipped  # noqa: PLC0415

    return shipped.state(SHIPPED_INPUT, datasets_dir) if datasets_dir.is_dir() else shipped.SITE


def for_dataset(ds: str, intake_dir: Path | None = None) -> tuple[float, str]:
    """Reference direction the dataset must use now → (value, source word: `generation:<name>` | `fixed`).
    intake_dir = the dataset's start layer location (relative to the caller's data root); default the canonical root."""
    if intake_dir is None:
        from dexengine.core import paths as P  # noqa: PLC0415 — owner of the layer locations

        intake_dir = P.layer_path("intake", ds)
    f = decl_path(intake_dir)
    _shipped_state(Path(f).parent.parent)
    try:
        cur = read_decl(f)
    except ValueError as e:
        raise RefDirError(f"{TAG} ✗ {ds}: {e}") from e
    if cur is None:
        if ds in BUILT_OFF_REF:
            from dexengine.core import shipped  # noqa: PLC0415

            at = shipped.row(SHIPPED_INPUT)["path"]
            raise RefDirError(
                input_missing(
                    f"{TAG} ✗ {ds}: no generation declaration {f} — {ds} was built under another reference direction "
                    f"than the constant {REF_DEG}°, so the constant would rotate its scenes. Unpack the shipped input "
                    f"`{SHIPPED_INPUT}` into <data root>/{at} (it holds {at}/{ds}/{GENERATION_FILE} and "
                    f"{at}/{shipped.SUMS_PREFIX}{SHIPPED_INPUT}; docs/datasets.md, 'Shipped inputs')"
                )
            )
        return REF_DEG, "fixed"
    if KEY not in cur:
        raise RefDirError(
            input_missing(
                f"{TAG} ✗ {ds}: generation declaration {f} has no `{KEY}` field — that "
                "generation's reference direction is unknown. The generation tool's `--current "
                f"{ds} --ref-dir --apply` fills it from the trees' value"
            )
        )
    return float(cur[KEY]), f"generation:{cur['generation']}"


def tree_value(ti: dict) -> float | None:
    v = (ti.get("frame") or {}).get(TREE_KEY)
    return None if v is None else float(v)


def part(ti: dict) -> str:
    """Generation part text (the form inside the shared-layer guard's generation string); "" when the tree wrote no
    value."""
    v = tree_value(ti)
    return "" if v is None else fmt(v)


def fmt(v: float) -> str:
    return f"{float(v):.2f}"


def check_tree(ds: str, ti: dict, intake_dir: Path | None = None) -> str:
    """Generation value ↔ layer value. "" when equal or when the tree wrote no value (before the rule · synthetic trees)
    · otherwise the reason."""
    v = tree_value(ti)
    if v is None:
        return ""
    ref, src = for_dataset(ds, intake_dir)
    if fmt(v) == fmt(ref):
        return ""
    return (
        f"{TAG} generation value {fmt(ref)}° ({src}) ↔ layer value {fmt(v)}° (tree frame.{TREE_KEY}) — this "
        "tree was not "
        "built with that generation's reference direction. Rebuild base with the generation's value, or "
        "change the declaration for a new generation"
    )


def oracle(measured_deg: float) -> tuple[bool, str]:
    """Re-measured TACO mean ↔ the constant → (within threshold, one line). The value is not changed; this only
    reports."""
    d = float(measured_deg) - REF_DEG
    ok = abs(d) <= TOL_DEG
    tail = "" if ok else " ✗ over the threshold — the constant stays; find out why it moved"
    return ok, (
        f"{TAG} fixed {fmt(REF_DEG)}° · TACO mean measured now {fmt(measured_deg)}° · diff {d:+.2f}° "
        f"{'≤' if ok else '>'} {TOL_DEG}°{tail}"
    )


def layer_values(ds: str, data_root: Path) -> dict[str, int]:
    """Values written by all live base trees of the dataset → {value text: tree count} (trees without a value count as
    'none')."""
    from dexengine.core import paths as P  # noqa: PLC0415 — owner of the layer locations

    out: dict[str, int] = {}
    for hand in sorted({P.hand_spec_dir(h) for h in P.HAND_IDS}):
        for emb in P.data_layout.EMBODIMENTS:
            d = Path(data_root) / P.data_layout.render("base", ds, hand=hand, emb=emb)
            for f in sorted(d.glob("*/task_info.json")) if d.is_dir() else []:
                v = tree_value(json.loads(f.read_text(encoding="utf-8")))
                k = "none" if v is None else fmt(v)
                out[k] = out.get(k, 0) + 1
    return out


#: The declaration fields a reader uses (read_decl: generation · for_dataset: KEY) — all the shipped copy carries.
SHIPPED_FIELDS = ("generation", KEY)


def ship(out: Path) -> str:
    """Write each dataset's declaration, reduced to SHIPPED_FIELDS, as `<out>/<ds>/GENERATION` and the checksum file of
    the shipped input `generation` (dexengine.core.shipped); returns its digest. A data root built from the raw
    datasets needs them: for_dataset refuses a dataset of BUILT_OFF_REF without its declaration. The datasets whose
    declared value differs from REF_DEG must be exactly BUILT_OFF_REF (the oracle of that tuple), or nothing is
    written."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.core import shipped  # noqa: PLC0415

    decls = {}
    for ds in sorted(P.data_layout.DATASETS):
        cur = read_decl(decl_path(P.layer_path("intake", ds)))
        if cur is None:
            continue
        if missing := [k for k in SHIPPED_FIELDS if k not in cur]:
            raise RefDirError(input_missing(f"{TAG} ✗ {ds}: declaration has no {missing}"))
        decls[ds] = cur
    off = sorted(ds for ds, cur in decls.items() if fmt(cur[KEY]) != fmt(REF_DEG))
    if off != sorted(BUILT_OFF_REF):
        raise RefDirError(
            f"{TAG} ✗ datasets declared under another value than {fmt(REF_DEG)}: {off} ≠ BUILT_OFF_REF "
            f"{sorted(BUILT_OFF_REF)} — correct the tuple before shipping"
        )
    for ds, cur in decls.items():
        (out / ds).mkdir(parents=True, exist_ok=True)
        (out / ds / GENERATION_FILE).write_text("".join(f"{k}={cur[k]}\n" for k in SHIPPED_FIELDS), encoding="utf-8")
    return shipped.write_sums(SHIPPED_INPUT, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--statement", action="store_true", help="source sentence")
    g.add_argument("--show", action="store_true", help="generation value and layer value per dataset")
    g.add_argument("--ship", type=Path, metavar="DIR", help="write the shipped input `generation` into DIR")
    a = ap.parse_args()
    if a.statement:
        print(statement())
        return 0
    if a.ship:
        print(f"{TAG} shipped declarations -> {a.ship} digest={ship(a.ship)}")
        return 0
    from dexengine.core import paths as P  # noqa: PLC0415

    root, rc = P.main_root(), 0
    for ds in sorted(P.data_layout.DATASETS):
        vals = layer_values(ds, root)
        try:
            ref, src = for_dataset(ds)
            bad = [k for k in vals if k != fmt(ref)]
            print(
                f"{TAG} ds={ds} generation value {fmt(ref)} ({src}) · layer values {vals or 'no trees'}"
                + (f" ✗ mismatch {bad}" if bad else "")
            )
            rc |= bool(bad)
        except RefDirError as e:
            print(f"{e} · layer values {vals or 'no trees'}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
