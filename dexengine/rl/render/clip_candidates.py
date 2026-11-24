"""Candidate rollouts of one clip: the best success and the best failure, each stashed as mp4 when it becomes the best,
then the chosen videos saved.
"""

from __future__ import annotations

import math
import os

import imageio


def _rollout_and_stash_candidates(
    out_dir,
    fps,
    cid,
    tag,
    cname,
    args_cli,
    rollouts_n,
    _perclip_rates,
    _run_one_rollout,
    _score_metrics,
    rank_metric,
    warmup_attempts,
    front_cam,
    manifest,
    best_idx,
    best_score,
    ego_best_path,
    front_best_path,
    n_f,
    n_s,
):
    """A candidate is encoded to mp4 under .stash/ when it becomes the best and only its path is kept: holding the
    frames of the current, best-success and best-failure rollouts (11 MB per 1920x1920 frame) exhausted host RAM.
    """
    _commit, _stash, best_chosen = _make_stash(out_dir, fps, cid)
    best_metrics, best_n, best_p, bf_chosen, bf_n, bf_p = _empty_candidates()
    attempt, best_chosen, best_idx, best_metrics, best_score, bf_chosen, bf_idx, bf_score, n_f, n_s = (
        _keep_best_failure_video(
            cid,
            tag,
            cname,
            args_cli,
            rollouts_n,
            _perclip_rates,
            _run_one_rollout,
            _score_metrics,
            _stash,
            best_n,
            best_p,
            best_score,
            bf_n,
            bf_p,
            n_s,
            n_f,
            best_chosen,
            best_idx,
            best_metrics,
            bf_chosen,
        )
    )

    _save_chosen_videos(
        out_dir,
        cid,
        tag,
        cname,
        rank_metric,
        warmup_attempts,
        front_cam,
        manifest,
        best_idx,
        best_score,
        ego_best_path,
        front_best_path,
        n_f,
        n_s,
        _commit,
        attempt,
        best_chosen,
        best_metrics,
        best_n,
        best_p,
        bf_chosen,
        bf_idx,
        bf_p,
        bf_score,
    )


def _empty_candidates():
    """Empty best-success and best-failure candidate holders."""
    bf_chosen = False
    best_p = {}
    bf_p = {}
    best_n = {}
    bf_n = {}
    best_metrics = None
    return best_metrics, best_n, best_p, bf_chosen, bf_n, bf_p


def _save_chosen_videos(
    out_dir,
    cid,
    tag,
    cname,
    rank_metric,
    warmup_attempts,
    front_cam,
    manifest,
    best_idx,
    best_score,
    ego_best_path,
    front_best_path,
    n_f,
    n_s,
    _commit,
    attempt,
    best_chosen,
    best_metrics,
    best_n,
    best_p,
    bf_chosen,
    bf_idx,
    bf_p,
    bf_score,
):
    """Move the chosen best (and failure) videos into place and write the manifest row."""
    saved_best = False
    if best_chosen:
        _commit(best_p.get("ego"), ego_best_path)
        _commit(best_p.get("front"), front_best_path)
        _commit(best_p.get("side"), os.path.join(out_dir, f"{cid:02d}_{tag}_side_best.mp4"))
        _commit(best_p.get("side_front"), os.path.join(out_dir, f"{cid:02d}_{tag}_side_front_best.mp4"))
        saved_best = True
        print(
            f"  [Render] clip {cid:02d} '{tag}': BEST-SUCCESS saved "
            f"(idx {best_idx + 1}/{attempt}, {n_s} succ, "
            f"{rank_metric}={best_score:.4f}, "
            f"ego {best_n.get('ego', 0)} frames, front {best_n.get('front', 0)})"
        )
    elif bf_chosen:
        _commit(bf_p.get("ego"), os.path.join(out_dir, f"{cid:02d}_{tag}_ego_failure_best.mp4"))
        _commit(bf_p.get("front"), os.path.join(out_dir, f"{cid:02d}_{tag}_front_failure_best.mp4"))
        _commit(bf_p.get("side"), os.path.join(out_dir, f"{cid:02d}_{tag}_side_failure_best.mp4"))
        _commit(bf_p.get("side_front"), os.path.join(out_dir, f"{cid:02d}_{tag}_side_front_failure_best.mp4"))
        print(
            f"  [Render] clip {cid:02d} '{tag}': NO SUCCESS in {attempt} attempts (retry cap) — "
            f"saved BEST-RANKED FAILURE instead (idx {bf_idx + 1}, {rank_metric}={bf_score:.4f})"
        )
    else:
        print(f"  [Render] clip {cid:02d} '{tag}': NO SUCCESS in {attempt} eval attempts — saving nothing")

    manifest[cid] = {
        "name": cname,
        "warmup": warmup_attempts,
        "attempts": n_s + n_f,
        "succ": n_s,
        "fail": n_f,
        "saved_best": saved_best,
        "best_idx": best_idx + 1,
        "best_score": best_score if best_idx >= 0 else None,
        "best_metrics": best_metrics,
        "has_front": front_cam is not None,
        "rank_metric": rank_metric,
        "mode": "best_success",
    }


def _make_stash(out_dir, fps, cid):
    """Stash helpers: a candidate is encoded once it becomes the best."""
    _stash_dir = os.path.join(out_dir, ".stash")

    def _stash(frames, name, _cid=cid):
        """frames -> temporary mp4 path (None when empty); the frame list can be dropped after the call."""
        if not frames:
            return None
        os.makedirs(_stash_dir, exist_ok=True)
        _p = os.path.join(_stash_dir, f"{_cid:02d}_{name}.mp4")
        imageio.mimsave(_p, frames, fps=fps)
        return _p

    def _commit(src, dst):
        """Move a temporary file to its final name (same filesystem, so the move is atomic)."""
        if src and os.path.exists(src):
            os.replace(src, dst)
            return True
        return False

    # "A candidate was chosen" is tracked by these flags, not by the paths: a view's path is None when that view
    # captured nothing (ego under RENDER_FRONT_ONLY).
    best_chosen = False
    return _commit, _stash, best_chosen


def _keep_best_failure_video(
    cid,
    tag,
    cname,
    args_cli,
    rollouts_n,
    _perclip_rates,
    _run_one_rollout,
    _score_metrics,
    _stash,
    best_n,
    best_p,
    best_score,
    bf_n,
    bf_p,
    n_s,
    n_f,
    best_chosen,
    best_idx,
    best_metrics,
    bf_chosen,
):
    """All-fail fallback: the best-ranked failure is kept as well, so a 0 % policy still yields one video."""
    bf_score = float("-inf")
    bf_idx = -1
    # All-fail retry: when a whole batch fails, batches of the same size are added (up to
    # videos_allfail_retry_batches more) until the first batch that holds a success.
    _retry_b = max(0, int(getattr(args_cli, "videos_allfail_retry_batches", 4) or 0))
    _total_cap = rollouts_n * (1 + _retry_b)
    # Eval-informed budget: a clip below 3 % completion runs one batch only (its failure video shows why it fails);
    # otherwise the cap is max(retry cap, min(ceil(4.6 / rate), 40)): ceil(4.6 / rate) rollouts give a >= 99 % chance
    # of one success at that rate, and the 40 bound keeps the render inside the no-progress watchdog.
    if _perclip_rates is not None:
        _pc = next((v for k, v in _perclip_rates.items() if k in cname), None)
        if _pc is not None:
            if _pc < 0.03:
                _total_cap = rollouts_n
            else:
                _total_cap = max(_total_cap, min(int(math.ceil(4.6 / max(_pc, 1e-6))), 40))
            print(f"  [Render] clip {cid:02d} '{tag}': eval rate {_pc:.2%} -> rollout cap {_total_cap}")
    return _rollout_candidates(
        cid,
        tag,
        rollouts_n,
        _run_one_rollout,
        _score_metrics,
        _stash,
        best_n,
        best_p,
        best_score,
        bf_n,
        bf_p,
        n_s,
        n_f,
        best_chosen,
        best_idx,
        best_metrics,
        bf_chosen,
        _total_cap,
        bf_idx,
        bf_score,
    )


def _rollout_candidates(
    cid,
    tag,
    rollouts_n,
    _run_one_rollout,
    _score_metrics,
    _stash,
    best_n,
    best_p,
    best_score,
    bf_n,
    bf_p,
    n_s,
    n_f,
    best_chosen,
    best_idx,
    best_metrics,
    bf_chosen,
    _total_cap,
    bf_idx,
    bf_score,
):
    """Roll the candidates out, keeping the best success and the best failure."""
    _batch_end = rollouts_n
    attempt = 0
    while attempt < _batch_end:
        is_failure, ego_fs, front_fs, metrics = _run_one_rollout()
        if is_failure:
            n_f += 1
            score = _score_metrics(metrics)
            # A candidate needs frames from some view, not from ego specifically.
            if score > bf_score and (ego_fs or front_fs):
                bf_score = score
                bf_idx = attempt
                for _v, _fr in (
                    ("ego", ego_fs),
                    ("front", front_fs),
                    ("side", metrics.get("_side_fs")),
                    ("side_front", metrics.get("_sidef_fs")),
                ):
                    if bf_p.get(_v):
                        try:
                            os.remove(bf_p[_v])  # drop the previous candidate's file
                        except OSError as e:
                            print(f"[CLIP_MODES] could not remove {bf_p[_v]}: {e}", flush=True)
                    bf_p[_v] = _stash(_fr, f"bf_{_v}")
                    bf_n[_v] = len(_fr) if _fr else 0
                bf_chosen = True
        else:
            n_s += 1
            score = _score_metrics(metrics)
            if score > best_score and (ego_fs or front_fs):
                best_score = score
                best_idx = attempt
                for _v, _fr in (
                    ("ego", ego_fs),
                    ("front", front_fs),
                    ("side", metrics.get("_side_fs")),
                    ("side_front", metrics.get("_sidef_fs")),
                ):
                    if best_p.get(_v):
                        try:
                            os.remove(best_p[_v])
                        except OSError as e:
                            print(f"[CLIP_MODES] could not remove {best_p[_v]}: {e}", flush=True)
                    best_p[_v] = _stash(_fr, f"best_{_v}")
                    best_n[_v] = len(_fr) if _fr else 0
                best_chosen = True
                best_metrics = metrics
        attempt += 1
        if attempt == _batch_end and n_s == 0 and _batch_end < _total_cap:
            _batch_end = min(_batch_end + rollouts_n, _total_cap)
            print(
                f"  [Render] clip {cid:02d} '{tag}': all {attempt} attempts failed — extending batch (cap {_total_cap})"
            )
    return attempt, best_chosen, best_idx, best_metrics, best_score, bf_chosen, bf_idx, bf_score, n_f, n_s
