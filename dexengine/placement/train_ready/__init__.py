"""The single measurement owner of trainability — the gate that picks the clips RL may train on.

T1 object USD · T2 registry promotion · T3 articulated USD · T4 tree roles current · T5 tree setup statement current.

Why: grounding (stage 1 · 2a) can finish with clips adopted while training still dies at the entrance.
  T1  the objects the adopted clips use have no USD (`data/<ds>/usd/<obj_id>/object.usd`) — the spawner cannot bring
      them up.
  T2  `dexengine.core.layout.DATASETS[ds].status` is still `intake`, so `datasets.resolve()` dies.
Both block training regardless of "it was adopted", so an oracle checks the statement (registry status) against the
measurement (does the asset exist): a measurement checked against a statement, not a second copy of it.

What it measures: per dataset, the ADOPTED clips of the active hand's final ledgers `status_final_<ds>.tsv`
(RR.ledger_files) are the universe; each clip tree's `objects[].obj_id` must have every object asset:
  data/<ds>/usd/<id>/object.usd               spawning (T1)
  data/<ds>/meshes/<id>/surface_points.npz    surface point-cloud observation
  data/<ds>/meshes/<id>/keypoints_fps2.npy    object keypoint reward term — without it training dies at the entrance
  data/<ds>/usd/<asset id>/object_artic.usd   articulated spawning (T3), objects with `articulation` only; the joint
                                              must be the spec's canonical one (`artic_not_canonical`)
Promotion (T2) is for datasets where covered == adopted. For a shortfall the bake command is
    ../IsaacLab/isaaclab.sh -p dexengine/robots/objects/usd_bake.py --batch --dataset <ds> --ids_file <file> --headless

Usage (from the repo root):
    python3 -m dexengine.placement.train_ready                 # table
    python3 -m dexengine.placement.train_ready --ids-dir DIR   # missing obj_ids → DIR/<ds>.txt (bake input)
    python3 -m dexengine.placement.train_ready --json          # machine readable
    python3 -m dexengine.placement.train_ready --stale-out F   # T4·T5 re-judge list (ds, axis, clip, reason)
    python3 -m dexengine.placement.train_ready --datasets a,b  # any mode: measure only those datasets' ledgers

Scope (`--datasets`): a campaign that narrowed its datasets measures only their ledgers; none given = every ledger
file of the hand; unknown names die (RR.datasets_scope). Every run prints on stderr
`[TRAIN_READY] scope=<ds,…|all> checked=<n> skipped_out_of_scope=<ds:n,…|->` (ADOPTED ledger rows).

T5: an adopted tree whose stamped setup (`desk_support` / `aerial_support`) differs from what the current rule gives
on the same tree with the current base statement (coords.setup_stale) is stale — reason `T5:<diff>`. A baked desk
tree without the stamp is read as 0 discs (old generation). The cost is dominated by articulated D20 trees; the
input-fingerprint cache (`--t5-cache`) keys on the rule code bytes and every input's mtime and size.

T5-judge: every judge result (ledger_states.JUDGED_STATUSES, and every adopted row, tree or not) is compared by the
judge-input fingerprint (ledger column `judge_input`, owner judge/input_record.py) against the same function on the
current base and rules. Different → `T5-judge:<fields>`; no fingerprint → `T5-judge:unknown`.

T5-base: an adopted row with a tree is compared on the fingerprint's `base` field (JFP.base_part) — a base rebuilt
from changed inputs changes the reference the adoption was judged on. Different → `T5-base`; none → `T5-base:unknown`.

T5-desk: an adopted row with a tree whose desk fields the judge reads (JFP.DESK_READ) differ from the current base
tree (JFP.base_tree) is stale — `T5-desk:<fields>`. A field the current base has and the tree lacks is counted
(stderr `[T5-desk]`), not queued.

Modules: assets (asset table and row helpers) · trees (T4 · static desk · T5) · judge_fp (T5-base · T5-desk ·
T5-judge) · survey (the loop) · cli (arguments and output). Output strings are byte-equal with the production survey
(their readers parse them), so the runtime messages keep their Korean text.
"""
