"""The README inside an RL bundle (rl_bundle.py). What it states comes from the bundle's meta (counts · trees ·
parts · gate) and the statement's own reader texts (run_inputs.READERS); the launch env from run_inputs.launch_env."""

from __future__ import annotations

from dexengine.placement.bundle import run_inputs as RI
from dexengine.placement.bundle.archive import ARCHIVE


def example_env(man: dict, b: dict) -> str:
    """One bundle's launch env (run_inputs.launch_env) with SEQ_DIR on the receiver's data root."""
    env = [
        kv.replace("SEQ_DIR=$PWD/", "SEQ_DIR=$DEXENGINE_MAIN_ROOT/", 1)
        for kv in RI.launch_env(dict(b, hand=man["hand"]), man)
    ]
    return " ".join(env + [f"DEXENGINE_SEQ_LIST=$BUNDLE/manifest/{man['hand']}/{b['list']}"])


def gib(n: int) -> str:
    """Bytes as GiB text."""
    return f"{n / 2**30:.2f} GiB"


def readme(meta: dict, man: dict, bundles: list[dict]) -> str:
    """The bundle README text (Markdown)."""
    h = meta["hand"]
    kinds = "".join(f"| {k} | {v['files']:,} | {gib(v['bytes'])} |\n" for k, v in meta["counts"]["kinds"].items())
    readers = "".join(f"| `{rd}` | {n:,} | {RI.READERS[rd]} |\n" for rd, n in meta["counts"]["readers"].items())
    scope = (
        "**every bundle** of the manifest"
        if meta["scope"] == "full"
        else f"**test bundle — {len(bundles)} of the manifest's {len(man['bundles'])} bundles** ("
        + ", ".join(f"`{b['id']}`" for b in bundles)
        + ")"
    )
    b0 = bundles[0]
    side_note = (
        "" if b0["side"] == "both" else " (one hand — with a variant cell, DEXENGINE_COMBO_VARIANT is in the env too)"
    )
    return (
        _head_section(h, man, meta, bundles, scope, kinds, readers)
        + _howto_section(meta, man, b0, side_note)
        + _NOT_INCLUDED
    )


def _head_section(h, man, meta, bundles, scope, kinds, readers) -> str:
    """Title, provenance and the contents tables."""
    return f"""# RL bundle — {h}

Processed data for RL training, judging and rendering of hand `{h}` (arm `{man["arm"]}`). Scope: {scope}.
Built by `dexengine/placement/bundle/rl_bundle.py` · \
code commit `{meta["git"]["commit"][:12]}` ({meta["git"]["branch"]}) ·
manifest fingerprint `{man["fingerprint"][:12]}` · created {meta["created"]}.

## Contents

{meta["files"]:,} files · {gib(meta["bytes"])} (uncompressed) · \
{meta["trees"]:,} trees (clips) · {len(bundles)} bundles.
What goes in is not chosen by this tool: it reads the statement of what the training entry reads,
`dexengine/placement/bundle/run_inputs.py` (per bundle `Stater.bundle` · the manifest ledgers `manifest_level` (a test
bundle: only the axis ledgers of the selected bundles) · the render `Stater.render`).

| kind | files | size |
|---|---:|---:|
{kinds}
| reader | files | meaning |
|---|---:|---|
{readers}
Per-file list = `FILES.tsv` (path · kind · reader · bytes). Every path is **relative to the data root**.
Manifest = `manifest/{h}/manifest.json` + `lists/<bundle>.txt` \
(training roster `DEXENGINE_SEQ_LIST`) · `.tsv` (clip table).

"""


def _howto_section(meta, man, b0, side_note) -> str:
    """Unpack, check and run instructions."""
    return f"""## Where to unpack

Pick one data root (the code checkout itself or another folder) and unpack under it as is. No symlinks.

```bash
BUNDLE=<this folder> ; ROOT=<data root>
cd "$BUNDLE" && sha256sum -c PARTS.sha256                              # 1. check the received parts
mkdir -p "$ROOT" && cat "$BUNDLE"/{ARCHIVE}.part-* | zstd -dc | tar -x -C "$ROOT"   # 2. unpack
export DEXENGINE_MAIN_ROOT="$ROOT"                                        # the code reads data from here
```

## Check

```bash
# 3. count {meta["files"]:,} · trees {meta["trees"]:,} · every sha256 → [RL_BUNDLE_CHECK] PASS
python3 -m dexengine.placement.bundle.rl_bundle --check "$BUNDLE" --root "$ROOT"
(cd "$ROOT" && sha256sum -c --quiet "$BUNDLE/SHA256SUMS")               # 3. the same check without the code
```

## Run

Boot smoke (bundle `{b0["id"]}`{side_note}) — the env is exactly what `run_inputs.launch_env` gives
(entry point `scripts/stage3_rl.sh`). Without `../IsaacLab` next to the code tree, set `ISAACLAB_DIR`.

```bash
{example_env(man, b0)} SMOKE=1 GPU=0 bash scripts/stage3_rl.sh <run name>
```
As a campaign: `dexengine/rl/campaign.py --manifests "$BUNDLE/manifest" …` (see its module doc). Judge with the same env
through `scripts/rl/judge_sidecar.sh`. The RL trainer draws no frames: robot videos come only from the record render
(`scripts/stage4_record.py`, one canonical look `dexengine.record.looks.look_d`).

"""


_NOT_INCLUDED = """## Not included

- Source datasets (`data/<ds>/raw`) · human hand keypoints (`mano`) · hand retargeting sources (`retarget`) · intake
  ledgers · intermediate outputs of stages 1–2 (the rest of base · placement judging · gate · IK pool · f0 staging) —
  inputs of a rebuild, never read by RL.
- The retargeting venv (`.venv`) — stages 1–2 only. RL uses the Isaac venv only.
- Code — from the git repository (the same commit as above gives the same statement).
- Checkpoints · training/judge logs · render videos — outputs of RL.
- The base contact sidecars that the policy-free PARK probe (`DEXENGINE_PARK_PROBE`) also reads
  (`precompute/<ds>/contact_mano_base`) — training, judging (checkpoint) and render do not read them.
- Render background materials (MDL) — NVIDIA content, downloaded from NVIDIA's public storage into the local look \
mirror (`python -m dexengine.record.looks.look_d --mirror`; values `dexengine.record.looks.look_assets`). The \
Willow desk of that mirror is in the bundle (reader `kit:desk`): every scene spawns it.
- Clips outside the manifest · other hands' data — one exception: the arm collider digest walks the URDFs of every
  hand registered with the arm, so those hands' mesh stages (`data/_stage/<hand>`, reader `entry:robot-identity`) are
  included.

A bundle carries third-party files under their own terms (THIRD_PARTY.md): the NVIDIA desk and materials, and the
hand and robot meshes fetched from their sources. It is for moving a run to your own machines; do not publish or share
it.
"""
