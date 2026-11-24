"""Judge side of the augmentation cells — judge mode D (region dump) · L (list judge).

The judge does not import augmentation code. This module calls the Judge's preparation and stage functions **in the
canonical order** (_clip_index_cache → _clip_desk_mask → _clip_volume · gate dp_confirm → FULL track_full_multi → mesh
_guard_pick → pivot swap → desk_ok_final → _pen_cross_guard) and writes into the bundle folder
(paths.run_path("run_aug_cells", hand, campaign=, bundle=)):
  region/region_<clip>.npz                D — every volume cell of the canonical dz layer · per cell θ · d (after the
                                          pivot swap) · canonical placement · grid · judge identity · the f0 shadow
                                          key it read (f0_key)
  judged/augcells_<combo>__<chunk>.jsonl  L — one line per listed cell (no early stop · no pruning · no fallback) + a
                                          final done line
  ikc/<paths.ikc_name(clip, kNN, side)>   L — all-frame joints of the passing cells (not in the chunk's R/ikc — the
                                          status ledger counts those)
  cells/ is written by the augmentation side; L only reads it.
Generation check: the clip's judge input record must equal the adopted row's record in the final ledger
(input_record.diff_ruleset), and the canonical dz (placement file robot_z_off ÷ step) must be exactly one judge dz
layer — otherwise the run dies.
Values: θ · d · pivot come from Judge._cell_dcap · _pivot_swap (the same functions as adoption), unrounded — json repr
round-trips float64 losslessly (the augmentation side compares with float64 == after parsing).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from dexengine.core.errors import input_missing
from dexengine.placement.judge import canon_fit as CF
from dexengine.placement.judge import f0_key as FK
from dexengine.placement.judge.aug_cells_canon import TAG, AugCellsCanon, dump_mismatch

MODES = ("off", "D", "L")
STAGES = ("volume", "gate", "full", "mesh", "desk", "guard", "pass")
DONE_KEYS = ("clips", "cells", "pass")  # chunk identity = the lines' judge_commit
REPO = Path(__file__).resolve().parents[2]


def done_marker(mode: str, n_clips: int, n_cells: int, n_pass: int | None = None) -> str:
    """End line of a judge run; record.augment.cells reads the clip count back with done_clips."""
    tail = f" · pass {n_pass}" if n_pass is not None else ""
    return f"{TAG} mode={mode} DONE clips {n_clips} · cells {n_cells}{tail}"


def done_clips(text: str, mode: str) -> list[str]:
    """Clip counts of the done_marker lines of one mode in a log."""
    return re.findall(rf"^{re.escape(TAG)} mode={re.escape(mode)} DONE clips (\d+)", text, re.M)


def die(msg: str):
    raise SystemExit(f"{TAG} ✗ {msg}")


def gen_reason(rows: list, jfp: str, where: tuple) -> str | None:
    """Generation check — None when there is exactly one adopted row and diff_ruleset says same, else the reason
    (AugCells._check_gen dies on it; the augmentation generation hold stores it). where = (dataset, emb, hand, clip)."""
    from dexengine.placement.judge import input_record as JFP  # noqa: PLC0415

    clip = where[3]
    if len(rows) != 1 or rows[0].get("status") != "ADOPTED":
        return f"{clip}: final ledger rows {[r.get('status') for r in rows]} — need exactly one adopted row"
    verdict, comps = JFP.diff_ruleset(rows[0].get(JFP.COLUMN) or None, jfp, "ADOPTED", where)
    if verdict != "same":
        return (
            f"{clip}: the judge input differs from the canonical bake record ({verdict}: {','.join(comps) or '-'}) "
            "— another generation"
        )
    return None


def dirs(bundle) -> dict:
    b = Path(bundle)
    return {k: b / k for k in ("region", "cells", "judged", "ikc")}


def region_path(bundle, clip: str) -> Path:
    return dirs(bundle)["region"] / f"region_{clip}.npz"


def cells_path(bundle, clip: str) -> Path:
    return dirs(bundle)["cells"] / f"cells_{clip}.json"


def axis_combo(arm: str, hand: str, dataset: str, emb: str, files=None) -> tuple[str, Path, str]:
    """(ledger axis key, ledger file, combo) — one judge run (one dataset · emb · scene) has exactly one final-ledger
    axis, and its clips' canonical placement records · baked trees live in the combo of the variant **that axis used**
    (arctic = profile variant _artic → <arm>_<hand>_artic · variant owner RR.ledger_axis_variant). The judge's own
    J.combo (arm_hand) does not know the variant — looked up with it, a variant-axis clip would die with 'no placement
    file'. Zero or several axes → dies (no choice is made)."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.placement.judge import input_record as JFP  # noqa: PLC0415

    RR = JFP._rr()
    # ledger = the hand root under the data root (main_root) — run from a worktree, run_root() is the worktree and
    # would find no ledger
    files = RR.ledger_files(root=P.main_root()) if files is None else files
    hit = sorted(
        (RR.ledger_axis(f), Path(f)) for f in files if RR.ledger_axis_parts(RR.ledger_axis(f))[:2] == (dataset, emb)
    )
    if len(hit) != 1:
        die(f"final-ledger axes of ({dataset}, {emb}): {[k for k, _ in hit]} — there must be exactly one")
    ((key, f),) = hit
    return key, f, P.combo_id(arm, hand, RR.ledger_axis_variant(key))


def judged_path(bundle, combo: str, chunk: str) -> Path:
    return dirs(bundle)["judged"] / f"augcells_{combo}__{chunk}.jsonl"


def head_commit() -> str:
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if r.returncode or len(r.stdout.strip()) != 40:
        die(f"could not read the judge code commit ({REPO}) — every line needs a judge_commit")
    return r.stdout.strip()


def cell_key(c) -> tuple:
    """Cell (ky, kx, kyy, dz) — dz is the judge dz-layer value (in cells) as is."""
    if not isinstance(c, (list, tuple)) or len(c) != 4:
        die(f"cell {c!r} — not of the form [ky, kx, kyy, dz]")
    return int(c[0]), int(c[1]), int(c[2]), float(c[3])


def match_dz(dz_layers, step: float, robot_z_off: float) -> float:
    """Placement-file robot_z_off (the plan line's round(dz × step, 4)) → one judge dz layer. None or two → dies."""
    hit = [float(z) for z in dz_layers if abs(float(z) * step - robot_z_off) <= 1e-4]
    if len(hit) != 1:
        die(
            f"{len(hit)} dz layers match the canonical robot_z_off {robot_z_off} (layers {list(dz_layers)} × "
            f"step {step})"
        )
    return hit[0]


def canon_cell(cells, theta, d_abs, canon: dict, step: float, yaw_gap: float) -> int:
    """Index of the region cell equal to the canonical placement (baked tree) — distance ≤ half a cell · yaw ≤ half a
    yaw gap · canonical dz. Dies unless exactly one."""
    hit = [
        i
        for i, (c, t, d) in enumerate(zip(cells, theta, d_abs, strict=True))
        if float(c[3]) == float(canon["dz"])
        and abs((float(t) - canon["yaw_deg"] + 180.0) % 360.0 - 180.0) <= yaw_gap / 2.0
        and abs(float(d[0]) - canon["dx"]) <= step / 2.0
        and abs(float(d[1]) - canon["dy"]) <= step / 2.0
    ]
    if len(hit) != 1:
        die(f"{len(hit)} canonical cells (must be 1) — canon {canon} · cells {len(cells)}")
    return hit[0]


def done_line(clips: int, cells: int, npass: int) -> dict:
    """Final line of a chunk jsonl (keys DONE_KEYS)."""
    return {"done": True, "clips": int(clips), "cells": int(cells), "pass": int(npass)}


def read_cells(path: Path, clip: str, jfp: str) -> list[tuple]:
    """cells/cells_<clip>.json → cell list (first = canonical cell). Dies on another clip · judge_fp, duplicates or
    an empty list."""
    if not path.is_file():
        die(input_missing(f"{clip}: no cell list {path}"))
    cj = json.loads(path.read_text(encoding="utf-8"))
    if cj.get("clip") != clip or cj.get("judge_fp") != jfp:
        die(
            f"{clip}: cell list clip {cj.get('clip')!r} · judge_fp differ from this judge run ({path.name}) — "
            "another generation"
        )
    cells = [cell_key(c) for c in (cj.get("cells") or [])]
    if not cells or len(set(cells)) != len(cells):
        die(f"{clip}: cell list is empty or has duplicates ({len(cells)})")
    return cells


class AugCells(AugCellsCanon):
    """Modes D · L of one judge run — the Judge makes one at boot and run_clip calls clip() per clip."""

    def __init__(self, J, mode: str, bundle: str, chunk: str):
        a = J.a
        if mode not in MODES[1:]:
            die(f"mode {mode!r} — {MODES[1:]}")
        if not bundle:
            die(
                input_missing(
                    "no --aug-cells-dir — the caller gives the bundle folder (paths.run_path('run_aug_cells', …))"
                )
            )
        # --reuse-from · --certify · --arc-probe · --full-mode are not judge flags any more (the FULL stage is always
        # track), so only the batched gate is checked. --place-reselect combines with D · L (the runner passes the
        # source chunk's margin to that condition group — record.augment.cells_cond).
        if mode == "L" and not a.batch_cells > 1:
            die(f"mode L runs only the canonical path (batched gate · FULL track) — batch_cells={a.batch_cells}")
        if mode == "L" and not chunk:
            die("mode L needs --aug-cells-chunk (the chunk jsonl name)")
        self.J, self.mode, self.bundle = J, mode, Path(bundle)
        self.D = dirs(bundle)
        from dexengine.core import paths as P  # noqa: PLC0415

        self.axis, self.ledger, self.combo = axis_combo(a.arm, a.hand, a.dataset, a.emb)
        # canonical placement records = the ledger axis's combo — _canon · Judge._adopted_pivot (_pivot_swap) read there
        J.PL_DIR = P.placement_dir(self.combo, a.dataset)
        self.commit = head_commit()
        self.n_clips = self.n_cells = self.n_pass = 0
        self.fh = None
        if mode == "D":
            self.D["region"].mkdir(parents=True, exist_ok=True)
        else:
            self.D["judged"].mkdir(parents=True, exist_ok=True)
            self.D["ikc"].mkdir(parents=True, exist_ok=True)
            p = judged_path(bundle, J.combo, chunk)
            if p.exists():
                die(f"chunk file already exists {p} — a new name per chunk (exactly one done line)")
            self.fh = open(p, "x", encoding="utf-8", buffering=1)  # noqa: SIM115 — lives for the run (finish closes)
        print(
            f"{TAG} mode={mode} bundle={self.bundle} commit={self.commit[:9]}"
            + f" axis={self.axis} combo={self.combo} placement={J.PL_DIR}"
            + (f" chunk={chunk}" if chunk else ""),
            flush=True,
        )

    # ── shared preparation ──────────────────────────────────────────────────────────────────────────
    def _canon(self, ck) -> dict:
        """Canonical placement of the clip (adopted placement file + baked tree) and its dz layer."""
        J, a = self.J, self.J.a
        adp = J._adopted_path(ck.clip)
        if not adp.is_file():
            die(input_missing(f"{ck.clip}: no placement file {adp} — only canonically adopted clips are accepted"))
        zoff = float(json.loads(adp.read_text())["robot_z_off"])
        dz = match_dz(J.DZ, J.step, zoff)
        from dexengine.core import paths as P  # noqa: PLC0415

        ti_p = P.trees_dir(self.combo, a.dataset, a.emb) / P.tree_name(ck.clip) / "task_info.json"
        if not ti_p.is_file():
            die(input_missing(f"{ck.clip}: no baked canonical tree {ti_p}"))
        from dexengine.placement.judge import input_record as JFP  # noqa: PLC0415

        tp = JFP._coords().tree_placement(json.loads(ti_p.read_text()))
        if tp is None:
            die(input_missing(f"{ck.clip}: {ti_p} is not a baked tree (no placement record)"))
        return {
            "dx": float(tp[0]),
            "dy": float(tp[1]),
            "yaw_deg": float(tp[2]),
            "pivot": [float(v) for v in tp[3][:2]],
            "dz": dz,
            "robot_z_off": zoff,
            "src": str(adp),
        }

    def _check_gen(self, clip: str, jfp: str) -> None:
        """Generation check — dies on a gen_reason reason (clips the generation hold already held do not come)."""
        a = self.J.a
        from dexengine.placement import final_ledger as FL  # noqa: PLC0415
        from dexengine.placement.judge import input_record as JFP  # noqa: PLC0415

        rows = [r for r in FL.read_final_rows(self.ledger) if r.get("clip") == clip]  # one axis (axis_combo)
        if why := gen_reason(rows, jfp, (a.dataset, a.emb, JFP._rr().hand_dir(), clip)):
            die(why)

    def _volume(self, ck, cn: dict) -> None:
        J = self.J
        CF.index_cache(J, ck, cn, self._geom, die)  # rule A (canon_fit)
        J._clip_desk_mask(ck)
        self._canon_exempt(ck, cn)
        ok = J._clip_volume(ck)
        if not CF.volume_exempt(J, ck, cn, *self._canon_grid(ck, cn), TAG) and not ok:
            die(
                input_missing(
                    f"{ck.clip}: no volume — a canonically adopted clip with an empty region (check "
                    "generation · conditions)"
                )
            )

    def clip(self, ck, jfp: str) -> None:
        """One clip: generation check → canonical placement → volume → mode D dump or mode L list judge."""
        self._check_gen(ck.clip, jfp)
        self._f0k = FK.key_of(FK.judge_marker(self.J, ck.clip))  # f0 key before reading (region dump f0_key)
        cn = self._canon(ck)
        self._volume(ck, cn)
        if self.mode == "D":
            self._dump(ck, jfp, cn)
        else:
            self._judge_list(ck, jfp, cn)
        self.n_clips += 1

    # ── mode D ───────────────────────────────────────────────────────────────────────────────────
    def _dump(self, ck, jfp: str, cn: dict) -> None:
        J = self.J
        dz = cn["dz"]
        lay = next(z for z in J.DZ if float(z) == dz)
        ky, kx, kyy = np.nonzero(ck.V[lay])
        cells = [(int(u), int(v), int(w), dz) for u, v, w in zip(ky, kx, kyy, strict=True)]
        geo = [self._geom(ck, c) for c in cells]
        theta = np.array([g[0] for g in geo], np.float64)
        d_abs = np.array([g[2] for g in geo], np.float64).reshape(-1, 2)
        p2s = {(float(g[3][0]), float(g[3][1])) for g in geo}
        if len(p2s) != 1:
            die(f"{ck.clip}: the pivot differs between cells {sorted(p2s)[:2]}")
        pivot = np.array(next(iter(p2s)), np.float64)
        yaw_gap = float(np.min(np.diff(np.sort(np.unique(np.asarray(J.yaws, float)))))) if len(J.yaws) > 1 else 360.0
        i0 = canon_cell(cells, theta, d_abs, cn, J.step, yaw_gap)
        canon = {k: cn[k] for k in ("dx", "dy", "yaw_deg", "pivot", "dz")}
        grid = {
            "half_x_m": float(J.Wx * J.step),
            "half_y_m": float(J.Wy * J.step),
            "yaw_lo": float(np.min(J.yaws)),
            "yaw_hi": float(np.max(J.yaws)),
            "step": float(J.step),
        }
        out = region_path(self.bundle, ck.clip)
        tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.npz")
        np.savez(
            tmp,
            cells=np.array([c[:3] for c in cells], np.int32).reshape(-1, 3),
            dz=np.full(len(cells), dz, np.float64),
            theta_abs_deg=theta,
            d_abs=d_abs,
            pivot=pivot,
            canon=np.array(json.dumps(canon)),
            robot_z_off=np.float64(dz * J.step),
            scene_z=np.float64(J.Z_OFF),
            grid=np.array(json.dumps(grid)),
            judge_fp=np.array(jfp),
            judge_commit=np.array(self.commit),
            f0_key=np.array(self._f0k),
        )
        os.replace(tmp, out)
        FK.unchanged(FK.judge_marker(J, ck.clip), self._f0k)
        self.n_cells += len(cells)
        print(
            f"{TAG} mode=D {ck.clip} dz={dz:g} source={cn['src']} cells {len(cells)} · canonical cell #{i0} "
            f"{cells[i0]} → {out.name}",
            flush=True,
        )

    # ── mode L ───────────────────────────────────────────────────────────────────────────────────
    def _judge_list(self, ck, jfp: str, cn: dict) -> None:
        J, a = self.J, self.J.a
        cells = read_cells(cells_path(self.bundle, ck.clip), ck.clip, jfp)
        geo = [self._geom(ck, c) for c in cells]
        yaw_gap = float(np.min(np.diff(np.sort(np.unique(np.asarray(J.yaws, float)))))) if len(J.yaws) > 1 else 360.0
        if canon_cell(cells[:1], [geo[0][0]], [geo[0][2]], cn, J.step, yaw_gap) != 0:
            die(f"{ck.clip}: first listed cell {cells[0]} is not the canonical cell")
        rp = region_path(self.bundle, ck.clip)
        by: dict = {}
        if rp.is_file():  # output check that D and L use the same functions · inputs
            z = np.load(rp, allow_pickle=False)
            by = {
                (int(c[0]), int(c[1]), int(c[2]), float(dz_)): (float(t), float(d[0]), float(d[1]))
                for c, dz_, t, d in zip(z["cells"], z["dz"], z["theta_abs_deg"], z["d_abs"], strict=False)
            }
            bad = dump_mismatch(ck.clip, cells, geo, by)
            if bad:
                die(
                    f"{ck.clip}: θ · d of listed cells differ from the region dump {bad[:2]} (must be the same "
                    "functions)"
                )
        row = dict.fromkeys(range(len(cells)))
        live = []
        self._confirm_cells(cells, geo, J, ck, row, a, live)
        qs = {}
        # FULL in slices of the anchor batch size (--max-checks), one after another — no pruning
        self._track_live(live, a, J, row, geo, ck, qs)
        self._write_rows(cells, geo, row, qs, ck, J, jfp, by)
        n_ok = sum(1 for k in row if row[k][0] == "pass")
        self.n_cells += len(cells)
        self.n_pass += n_ok
        cnt = {s: sum(1 for k in row if row[k][0] == s) for s in STAGES}
        print(
            f"{TAG} mode=L {ck.clip} cells {len(cells)} · pass {n_ok} · "
            + " ".join(f"{s}={n}" for s, n in cnt.items() if n),
            flush=True,
        )

    def _confirm_cells(self, cells, geo, J, ck, row, a, live):
        """Per cell: inside the volume, then the DP confirm; survivors go to live, the rest get their row."""
        for k, (c, g) in enumerate(zip(cells, geo, strict=True)):
            lay = next((z for z in J.DZ if float(z) == c[3]), None)
            if (
                lay is None
                or not (
                    0 <= c[0] < ck.V[lay].shape[0] and 0 <= c[1] < ck.V[lay].shape[1] and 0 <= c[2] < ck.V[lay].shape[2]
                )
                or not ck.V[lay][c[0], c[1], c[2]]
            ):
                row[k] = ("volume", "cell outside the volume", {})
                continue
            J.n_chk += 1
            okg, metg = J.dp_confirm(
                ck.traj, ck.cpiv, g[0], g[1], lay * J.step, step_mult=float(a.stride * a.dp_stride)
            )
            if not okg:
                J._tally("sparse", metg)
                row[k] = ("gate", "gate", metg)
                continue
            live.append(
                (
                    k,
                    {
                        "piv": ck.cpiv,
                        "yaw": g[0],
                        "dcap": g[1].copy(),
                        "dzm": lay * J.step,
                        "dzi": lay,
                        "chk": k + 1,
                        "gateq": {s2: (J.GATEQ[s2].copy() if s2 in J.GATEQ else None) for s2 in J.sides},
                    },
                )
            )

    def _track_live(self, live, a, J, row, geo, ck, qs):
        """Full tracking of the live cells in groups of max_checks, under the guard prefetch; fills row and qs."""
        for i in range(0, len(live), max(1, int(a.max_checks))):
            grp = live[i : i + max(1, int(a.max_checks))]
            res = J.track_full_multi([it for _k, it in grp])
            gj = [
                (it["dzm"], q_, J.cand_place(it))
                for (_k, it), (ok_, _m, q_, _e) in zip(grp, res, strict=True)
                if ok_ and q_ is not None
            ]
            with J.guard_prefetch(gj):
                for (k, it), (ok_, met_, q_, _e) in zip(grp, res, strict=True):
                    if not ok_ or q_ is None:
                        J._tally("confirm", met_)
                        row[k] = ("full", "confirm", met_)
                        continue
                    J.LASTQ.clear()
                    J.LASTQ.update(q_)
                    if not J._guard_pick(it["dzm"], J.cand_place(it)):
                        row[k] = ("mesh", "mesh guard", met_)
                        continue
                    _yaw, _dc, d, p2 = geo[k]
                    if not self._desk_ok(ck, k, d, p2, it["yaw"]):
                        row[k] = ("desk", "final desk check (after the pivot swap)", met_)
                        continue
                    gok, gwhy = J._pen_cross_guard(it["dzm"], place=(d[0], d[1], it["yaw"], p2))
                    if not gok:
                        row[k] = ("guard", f"final guard {gwhy}", met_)
                        continue
                    row[k] = ("pass", "-", met_)
                    qs[k] = {s: np.asarray(q, np.float32) for s, q in J.LASTQ.items()}

    def _write_rows(self, cells, geo, row, qs, ck, J, jfp, by):
        """One output line per cell (stage, reason, metrics and the passing joints)."""
        from dexengine.core import paths as P  # noqa: PLC0415

        for k, (c, g) in enumerate(zip(cells, geo, strict=True)):
            stage, why, met = row[k]
            if stage == "pass":
                for side, q in qs[k].items():
                    np.savez(
                        self.D["ikc"] / P.ikc_name(ck.clip, f"k{k:02d}", side),
                        arm_joint_positions_all=q,
                        num_frames=np.int32(len(q)),
                    )
            td = by.get(c, (float(g[0]), float(g[2][0]), float(g[2][1])))  # the dump's values when it has the cell
            self.fh.write(
                json.dumps(
                    {
                        "clip": ck.clip,
                        "k": k,
                        "cell": [c[0], c[1], c[2], c[3]],
                        "theta_abs_deg": td[0],
                        "d_abs": [td[1], td[2]],
                        "pivot": [float(g[3][0]), float(g[3][1])],
                        "robot_z_off": float(c[3] * J.step),
                        "ok": stage == "pass",
                        "stage": stage,
                        "reason": why,
                        "met": {kk: str(v) for kk, v in (met or {}).items()},
                        "judge_fp": jfp,
                        "judge_commit": self.commit,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def finish(self) -> None:
        """End of the run: mode L writes the done line and closes the chunk; the DONE marker is printed."""
        if self.mode == "L":
            self.fh.write(json.dumps(done_line(self.n_clips, self.n_cells, self.n_pass)) + "\n")
            self.fh.close()
        print(done_marker(self.mode, self.n_clips, self.n_cells, self.n_pass if self.mode == "L" else None), flush=True)
