"""The 64 Tab. 2 clips -> SPIDER's processed input layout, copied (never linked) from our data root
(dexengine.core.paths.main_root(), read only):
  <out>/processed/taco/sharpa_wave/bimanual/<clip>/{scene.xml, scene_eq.xml, task_info.json, 0/*.npz}
  <out>/taco_retarget/processed/taco/assets/{objects/<id>, robots/sharpa_wave}   (scene.xml meshdir relative paths as
  is)
  <out>/processed/taco/assets -> ../../taco_retarget/processed/taco/assets      (for task_info's *_mesh_dir)
Checks at the end: 64 clips copied and every file scene.xml references exists.
usage: prep_inputs.py <clips.tsv with a 'clip' column> <out dataset root>   (the dexengine Python)"""

import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

from dexengine.core import layout, paths


def main(clips_tsv: str, out: str) -> int:
    src = str(paths.main_root() / "data")
    base = paths.main_root() / layout.render("base", "taco", hand="sharpa_wave", emb="bimanual")
    with open(clips_tsv) as fh:
        clips = [r["clip"] for r in csv.DictReader(fh, delimiter="\t")]
    a_src = f"{src}/taco_retarget/processed/taco/assets"
    a_out = f"{out}/taco_retarget/processed/taco/assets"
    os.makedirs(f"{a_out}/objects", exist_ok=True)
    if not os.path.isdir(f"{a_out}/robots/sharpa_wave"):
        shutil.copytree(f"{a_src}/robots/sharpa_wave", f"{a_out}/robots/sharpa_wave")
    lnk = f"{out}/processed/taco/assets"
    os.makedirs(os.path.dirname(lnk), exist_ok=True)
    if not os.path.lexists(lnk):
        os.symlink("../../taco_retarget/processed/taco/assets", lnk)
    objs, missing = set(), []
    for c in clips:
        s, d = f"{base}/{c}", f"{out}/processed/taco/sharpa_wave/bimanual/{c}"
        if not os.path.isdir(d):
            shutil.copytree(s, d)
        for f in re.findall(r'file="([^"]+)"', Path(f"{d}/scene.xml").read_text()):
            m = re.match(r"objects/([^/]+)/", f)
            if m:
                objs.add(m.group(1))
    for o in sorted(objs):
        if not os.path.isdir(f"{a_out}/objects/{o}"):
            shutil.copytree(f"{a_src}/objects/{o}", f"{a_out}/objects/{o}")
    for c in clips:
        d = f"{out}/processed/taco/sharpa_wave/bimanual/{c}"
        missing.extend(
            (c, f)
            for f in re.findall(r'file="([^"]+)"', Path(f"{d}/scene.xml").read_text())
            if not os.path.exists(f"{a_out}/{f}")
        )
        for k in ("right_object_mesh_dir", "left_object_mesh_dir"):
            v = json.loads(Path(f"{d}/task_info.json").read_text()).get(k)
            if v and not os.path.isdir(f"{out}/{v}"):
                missing.append((c, v))
    got = len([x for x in os.listdir(f"{out}/processed/taco/sharpa_wave/bimanual") if x in set(clips)])
    print(f"[PREP_SPIDER] clips={got}/{len(clips)} objects={len(objs)} missing_refs={len(missing)}", missing[:5])
    return 0 if got == len(clips) == 64 and not missing else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
