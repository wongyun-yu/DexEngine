"""The per-clip render modes of the per-clip video render, and its manifest.

kinematic replay  robot and objects are teleported to the reference at every frame (no policy, no physics step).
best success      ``rollouts_n`` rollouts (extended on all-fail); the best-ranked success is kept, else the
                  best-ranked failure.
legacy            the first success and the first failure among ``max_attempts`` rollouts are kept.
"""

from __future__ import annotations

import os

import imageio

from dexengine.rl.render.clip_candidates import _rollout_and_stash_candidates


def _best_success_clip(
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
):
    """Run the clip's rollouts and save the best-ranked success (or the best-ranked failure) per view."""
    ego_best_path = os.path.join(out_dir, f"{cid:02d}_{tag}_ego_best.mp4")
    front_best_path = os.path.join(out_dir, f"{cid:02d}_{tag}_front_best.mp4")
    n_s = n_f = 0
    best_score = float("-inf")
    best_idx = -1
    _rollout_and_stash_candidates(
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
    )


def _legacy_clip(out_dir, fps, cid, tag, cname, max_attempts, _run_one_rollout, warmup_attempts, front_cam, manifest):
    """Save the first success and the first failure of the clip, then finish ``max_attempts`` for the counts."""
    ego_s_path = os.path.join(out_dir, f"{cid:02d}_{tag}_ego_success.mp4")
    ego_f_path = os.path.join(out_dir, f"{cid:02d}_{tag}_ego_failure.mp4")
    front_s_path = os.path.join(out_dir, f"{cid:02d}_{tag}_front_success.mp4")
    front_f_path = os.path.join(out_dir, f"{cid:02d}_{tag}_front_failure.mp4")

    have_s = have_f = False
    n_s = n_f = 0

    for attempt in range(max_attempts):
        if have_s and have_f:
            break
        is_failure, ego_fs, front_fs, _ = _run_one_rollout()
        if is_failure:
            n_f += 1
        else:
            n_s += 1

        # Saving is gated on frames from any view, not on the ego list (empty under RENDER_FRONT_ONLY).
        if not is_failure and not have_s and (ego_fs or front_fs):
            if ego_fs:
                imageio.mimsave(ego_s_path, ego_fs, fps=fps)
            if front_fs:
                imageio.mimsave(front_s_path, front_fs, fps=fps)
            have_s = True
            print(
                f"  [Render] clip {cid:02d} '{tag}': SUCCESS saved "
                f"(ego {len(ego_fs)} frames, front {len(front_fs)}) "
                f"on eval attempt {attempt + 1}"
            )
        elif is_failure and not have_f and (ego_fs or front_fs):
            if ego_fs:
                imageio.mimsave(ego_f_path, ego_fs, fps=fps)
            if front_fs:
                imageio.mimsave(front_f_path, front_fs, fps=fps)
            have_f = True
            print(
                f"  [Render] clip {cid:02d} '{tag}': FAILURE saved "
                f"(ego {len(ego_fs)} frames, front {len(front_fs)}) "
                f"on eval attempt {attempt + 1}"
            )

    # Once both videos are saved the remaining attempts still run, so the manifest counts all max_attempts
    # rollouts.
    remaining = max_attempts - (n_s + n_f)
    for _ in range(remaining):
        is_failure, _, _, _ = _run_one_rollout()
        if is_failure:
            n_f += 1
        else:
            n_s += 1

    manifest[cid] = {
        "name": cname,
        "warmup": warmup_attempts,
        "attempts": n_s + n_f,
        "succ": n_s,
        "fail": n_f,
        "saved_s": have_s,
        "saved_f": have_f,
        "has_front": front_cam is not None,
        "mode": "legacy",
    }
    if not have_s:
        print(f"  [Render] clip {cid:02d}: NO SUCCESS in {n_s + n_f} eval attempts")
    if not have_f and n_f == 0:
        print(f"  [Render] clip {cid:02d}: 100% SUCCESS — no failure video to save")
    elif not have_f:
        print(f"  [Render] clip {cid:02d}: FAILURES occurred ({n_f}) but no frames captured")


def _write_manifest(
    out_dir,
    checkpoint_path,
    warmup_attempts,
    rollouts_n,
    select_best_success,
    rank_metric,
    max_episode_steps,
    fps,
    manifest,
):
    """Write ``manifest.txt`` (one row per rendered clip) and return its path."""
    mpath = os.path.join(out_dir, "manifest.txt")
    with open(mpath, "w") as f:
        f.write("# Per-clip video render manifest\n")
        f.write(f"# Checkpoint: {checkpoint_path}\n")
        f.write(
            f"# warmup_attempts={warmup_attempts}, eval_rollouts={rollouts_n}, "
            f"select_best_success={int(select_best_success)}, rank_metric={rank_metric}\n"
        )
        f.write(f"# max_episode_steps={max_episode_steps}, fps={fps}\n\n")
        if select_best_success:
            f.write(
                f"{'cid':<5}{'name':<55}{'warmup':<8}{'evals':<8}{'succ':<8}{'fail':<8}"
                f"{'best_idx':<10}{'best_score':<14}{'saved':<10}\n"
            )
            for cid in sorted(manifest.keys()):
                m = manifest[cid]
                saved = "ego_best" if m.get("saved_best") else "(none)"
                if m.get("saved_best") and m.get("has_front"):
                    saved = "ego+front_best"
                score_s = f"{m['best_score']:.4f}" if m.get("best_score") is not None else "n/a"
                bidx = m.get("best_idx", -1)
                bidx_s = f"{bidx}/{rollouts_n}" if bidx > 0 else "-"
                f.write(
                    f"{cid:<5}{str(m.get('name', ''))[:53]:<55}{m.get('warmup', 0):<8}"
                    f"{m.get('attempts', 0):<8}{m.get('succ', 0):<8}{m.get('fail', 0):<8}"
                    f"{bidx_s:<10}{score_s:<14}{saved:<10}\n"
                )
        else:
            f.write(f"{'cid':<5}{'name':<55}{'warmup':<8}{'evals':<8}{'succ':<8}{'fail':<8}{'saved':<15}\n")
            for cid in sorted(manifest.keys()):
                m = manifest[cid]
                if m.get("mode") == "kinematic_replay":
                    f.write(
                        f"{cid:<5}{str(m.get('name', ''))[:53]:<55}{'kinrepl':<8}{'-':<8}{'-':<8}{'-':<8}"
                        f"{('front+ego' if m.get('has_front') else 'ego'):<15}\n"
                    )
                    continue
                saved = []
                if m.get("saved_s"):
                    saved.append("ego_s")
                    if m["has_front"]:
                        saved.append("front_s")
                if m.get("saved_f"):
                    saved.append("ego_f")
                    if m["has_front"]:
                        saved.append("front_f")
                elif m["fail"] == 0:
                    saved.append("(100%_succ)")
                f.write(
                    f"{cid:<5}{str(m.get('name', ''))[:53]:<55}{m.get('warmup', 0):<8}"
                    f"{m.get('attempts', 0):<8}{m.get('succ', 0):<8}{m.get('fail', 0):<8}"
                    f"{('+'.join(saved) or 'none'):<15}\n"
                )
    print(f"  [Render] Manifest written: {mpath}")
    return mpath
