"""CHORD support-surface USDA files (reconstruct_support_surfaces.py output) -> per sequence the lowest support top z
(JSON); the ManipTrans TACO loader reads it. Needs pxr (CHORD's .venv-prep).
usage: support_tops.py <reconstructed_stage dir> <out json>"""

import glob
import json
import os
import re
import sys

from pxr import Usd, UsdGeom


def main(src: str, out_path: str) -> None:
    out = {}
    for f in glob.glob(src + "/*_support.usda"):
        seq = re.sub(r"(_sharpa_wave)?_support\.usda$", "", os.path.basename(f))
        s = Usd.Stage.Open(f)
        tops = [
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
            .ComputeWorldBound(p)
            .ComputeAlignedRange()
            .GetMax()[2]
            for p in s.Traverse()
            if p.GetParent() and p.GetParent().GetName() == "support_surfaces"
        ]
        if tops:
            out[seq] = min(tops)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=0, sort_keys=True)
    print(f"[SUPPORT_TOPS] {len(out)} sequences → {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
