"""SharpA Wave hand asset for DexMachina (Tab. 2 baseline): their hand-processing steps 1-2.

Source = the vendor URDF the SPIDER/EgoEngine SharpA MJCF was compiled from (read only):
  assets/robots/hands/sharpa_wave/{side}_sharpa_wave/{side}_sharpa_wave_with_flange.urdf + meshes/, put there by
      `python -m dexengine.robots.vendor fetch` (sha256 checked against configs/robots/vendor/files.tsv)
Output (dexmachina asset dir, hand name `sharpa_wave_hand`):
  dexmachina/assets/sharpa_wave_hand/{side}/meshes/*.STL           byte copies
  dexmachina/assets/sharpa_wave_hand/{side}_sharpa_wave_copy.urdf  = vendor URDF with
      (a) mesh paths package://{side}_sharpa_wave/meshes/X -> {side}/meshes/X
      (b) one root link `{side}_hand_root` + fixed joint root->`{side}_hand_flange` inserted as the FIRST joint:
          their add_wrist_dof.py rewires the first joint's child into the 6 forearm dof links, and the vendor
          URDF's first joint is flange->wrist (no dummy root), which would make the flange its own descendant.
Nothing else is changed (no mass/inertia/limit/collision edits). Then run their script (unchanged):
  python hand_proc/add_wrist_dof.py assets/sharpa_wave_hand/{side}_sharpa_wave_copy.urdf {side}_sharpa_wave_6dof.urdf \
      --dof_choices all --base_link {side}_hand_flange       (right: side prefix R_, left: L_ -- see below)
The 6-dof base sits at the flange frame = the SPIDER MJCF base body frame (checked: MJCF index_MCP_VL at zero pose is
the URDF C_MC->index_MCP_FE offset plus flange->C_MC 0.0295 m, rotated by the base quat).
usage: build_sharpa_asset.py <dexmachina assets/sharpa_wave_hand dir>
"""

import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VENDOR = os.path.join(REPO, "assets/robots/hands/sharpa_wave")
FILES_TSV = os.path.join(REPO, "configs/robots/vendor/files.tsv")  # the fetch table: destination -> sha256


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fetched_sums() -> dict:
    """destination path (repo-relative) -> sha256 of every file the vendor fetch puts in place."""
    with open(FILES_TSV, encoding="utf-8") as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip() and not ln.startswith("#")]
    return {r[1]: r[3] for r in rows}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fetched = fetched_sums()
    for side in ("right", "left"):
        src = f"{VENDOR}/{side}_sharpa_wave"
        rel = os.path.relpath(src, REPO)
        sums = {k[len(rel) + 1 :]: v for k, v in fetched.items() if k.startswith(rel + "/")}
        urdf = f"{src}/{side}_sharpa_wave_with_flange.urdf"
        assert sums[f"{side}_sharpa_wave_with_flange.urdf"] == sha(urdf), f"vendor URDF sha mismatch {urdf}"
        md = f"{out_dir}/{side}/meshes"
        os.makedirs(md, exist_ok=True)
        n = 0
        for f in sorted(os.listdir(f"{src}/meshes")):
            k = f"meshes/{f}"
            assert k in sums and sums[k] == sha(f"{src}/{k}"), f"mesh sha mismatch {k}"
            shutil.copyfile(f"{src}/{k}", f"{md}/{f}")
            n += 1
        txt = Path(urdf).read_text()
        txt, k = re.subn(rf"package://{side}_sharpa_wave/meshes/", f"{side}/meshes/", txt)
        assert k > 0 and "package://" not in txt, "unexpected mesh path form"
        root = (
            f'  <link name="{side}_hand_root"/>\n'
            f'  <joint name="{side}_hand_root_fix_joint" type="fixed">\n'
            f'    <origin xyz="0 0 0" rpy="0 0 0"/>\n'
            f'    <parent link="{side}_hand_root"/>\n'
            f'    <child link="{side}_hand_flange"/>\n'
            f"  </joint>\n"
        )
        i = txt.index("<link ")  # before the first link -> the root joint is the first <joint> in document order
        txt = txt[:i] + root.lstrip() + "  " + txt[i:]
        out = f"{out_dir}/{side}_sharpa_wave_copy.urdf"
        Path(out).write_text(txt)
        print(
            f"[SHARPA_ASSET] {side}: {n} meshes · {out} · vendor sha "
            f"{sums[f'{side}_sharpa_wave_with_flange.urdf'][:12]}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
