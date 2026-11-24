"""DexMachina row variants: the one owner of what a variant changes in the DexMachina pipeline.

No variant (DM_VARIANT unset or empty) = the original rows: no script reads this file then.
  m0p2       every object at our fixed object mass (common/prep_mass_0p2.py owner_mass(), whose label ties the value to
  the
             name m0p2). Inertia = the same shape at uniform density: principal moments x M/m_old, ipos and iquat
             unchanged (taco_to_dexmachina.py --dump-inertia … --mass-variant m0p2 ->
             data/dexmachina/<key>/inertia_m0p2.json). Own clip keys (alphanumeric only: DexMachina's clip strings
             split on '-' and file names on '_'), own object
             assets, own curve and record dirs, own record label. RL cap = 2 h of NVIDIA L40S time
             (scaled by card speed on another card).
  m0p2match  the m0p2 clips (same keys, assets and inertia) rerun on the DexMachina port's DEXM_MATCH switch:
  hand-object
             friction right / left, the run-time mass check and the virtual object controller decayed to 0 inside the
             cap (0 for its last third); score = the final checkpoint with the controller off, same scorer and cap. Own
             curve and record dirs, label and queue.
usage: variant.py --sh <variant>          shell assignments DMV_* (unknown variant -> exit 2)
       variant.py --key <variant> <idx>   the variant's clip key of an E123_CLIPS_v1 row
"""

import shlex
import sys

VARIANTS = {
    "m0p2": {
        "key_prefix": "e123m",
        "obj_root": "taco_e123_m0p2/objects",
        "curve": "curve_m0p2",
        "records": "records_m0p2",
        "method": "DexMachina@mass_0p2",
        "inertia": "inertia_m0p2.json",
        "queue": "queue_dexmachina_m0p2.tsv",
    },
}
VARIANTS["m0p2match"] = dict(
    VARIANTS["m0p2"],
    curve="curve_m0p2match",
    records="records_m0p2match",
    method="DexMachina@mass_0p2+match",
    queue="queue_dexmachina_m0p2match.tsv",
)


def get(name):
    """The variant's settings; stops on an unknown name."""
    if name not in VARIANTS:
        raise SystemExit(f"[DM_VARIANT] FAIL unknown variant {name!r} (known: {', '.join(VARIANTS)})")
    v = VARIANTS[name]
    assert v["key_prefix"].isalnum() and v["key_prefix"] != "e123c", v
    return v


def key(name, idx):
    """The variant's clip key of clip index <idx>."""
    return f"{get(name)['key_prefix']}{int(idx):03d}"


if __name__ == "__main__":
    if sys.argv[1:2] == ["--sh"] and len(sys.argv) == 3:
        v = get(sys.argv[2])
        print("; ".join(f"DMV_{k.upper()}={shlex.quote(str(x))}" for k, x in [("name", sys.argv[2])] + list(v.items())))
    elif sys.argv[1:2] == ["--key"] and len(sys.argv) == 4:
        print(key(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(__doc__)
