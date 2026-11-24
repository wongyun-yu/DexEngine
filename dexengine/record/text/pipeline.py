"""Text pipeline (clip instructions + phase texts), run separately from recording. One call = one pass (tick); every
stage is idempotent, so a later pass continues where an interrupted one stopped.

Root = run layer run_collect_text/<track> (hand independent). Config <root>/track.json (explicit, no defaults — a
missing or unknown key dies):
  pairs          target dataset<TAB>clip list (the final datasets' clips are added when frames is on)
  data_root      clip tree root data/
  frames · views · frame_size     frames on (phases.TEXT_FRAMES), views, size; renders are read from the final
                 datasets under the data root (and their aug twins)
  model · redo_model              first model, model for the redo (rejections ∪ confidence low)
  text_only      false = wait for the render; true = clips without a render get text from labels + reference phases
  budget_usd     spent (usage, api.folder_usd) + pending (submitted, not received) + new estimate above this: no submit
  submit_until   no new submit after this time (UTC ISO; fetching continues)
  min_batch      submit only with at least this many new clips (1 from 6 hours before submit_until)
  retext         true = reload the final datasets after the table changed (store.retext)
  redo_with_why  put the rejection reasons into the redo requests (retry_why)
  clip_usd_cap   halt when a received round costs more per batch clip (a round may carry its own clip_usd_cap)
  round_cap      cap on new clips per round (0 = none, order.cap)
  via · workers  submit path batch | sync (synchronous, raw.jsonl of the same folder) · concurrent sync requests
Pass: (1) per round status -> when ended fetch -> validate -> (main round) build and submit the redo; (2) merge (all
validated rounds, oldest first) -> <root>/final/texts.jsonl (final table) -> split_final: per dataset the data layer
collect_text/texts.jsonl (+ texts.source.json = source table, sha, rows); (3) retext; (4) new round: targets - table
- in flight - unresolved -> sources (--frames-only unless text_only) -> build -> key / time / budget gates -> submit.
Without a key, or with --dry, the network stages (status, fetch, submit) are skipped. Withdraw: --withdraw <round>
--why <reason> (unsubmitted rounds only); void: --void <round> --why <reason> (submitted rounds); their clips are
built again. A key or credit error (api.KEY_NEEDED_RE) writes <root>/STOP with the reason and exits 3.
End line: targets = table + in flight + waiting for render + unresolved + no tree + ready + after the cap, and money.
Track state (config keys, state file, STOP): track.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from dexengine.core import config
from dexengine.core.repo import REPO_ROOT
from dexengine.record.text import api as CTA
from dexengine.record.text import generate as CTX
from dexengine.record.text import order as ORD
from dexengine.record.text import store as CSTORE
from dexengine.record.text import text_table as TT
from dexengine.record.text.api import MARK, die, read_jsonl
from dexengine.record.text.track import halt, load, net_ok, now_utc, ns, save, track_root

ERR_CAP = 2  # batch errors per round that halt the pass (advance)
SPLIT_FILE, SPLIT_SOURCE = "texts.jsonl", "texts.source.json"  # data layer collect_text output and its source record


def advance(root: Path, cfg: dict, st: dict, net: str | None) -> None:
    """(1) Advance the rounds: submitted -> fetched -> validated -> (main) build and submit the redo. Two gates: batch
    errors >= ERR_CAP cancel the remaining batches and halt; a received round costing more than clip_usd_cap per batch
    clip halts (the probe request is a separate fixed cost)."""
    for r in st["rounds"]:
        d = Path(r["dir"])
        if r["state"] == "submitted":
            if net:
                continue
            CTA.cmd_status(ns(dir=d))
            bs = json.loads((d / "batches.json").read_text())
            if why := CTA.error_gate(r["name"], bs, ERR_CAP):
                halt(root, why)
                save(root, st)
                return
            if all(b.get("processing_status") == "ended" for b in bs):
                CTA.cmd_fetch(ns(dir=d))
                r["state"] = "fetched"
                usd, fixed, nb = CTA.round_usd(d)
                r.update(usd_batch=round(usd, 4), usd_fixed=round(fixed, 4))
                cap = float(r.get("clip_usd_cap", cfg["clip_usd_cap"]))  # a round's own approved cap wins
                if nb and usd / nb > cap:
                    halt(root, f"{r['name']} batch cost per clip ${usd / nb:.4f} > ${cap}")
        if r["state"] == "fetched":
            CTX.cmd_validate(ns(dir=d))
            r["state"] = "validated"
        if r["state"] == "validated" and r["kind"] == "main" and not r.get("redo"):
            out = next(
                o for o in (d.parent / f"{d.name}_redo{'' if k < 2 else k}" for k in range(1, 99)) if not o.exists()
            )  # a voided earlier redo folder stays; next name (_redo2 ...)
            CTX.cmd_redo(ns(src=d, model=cfg["redo_model"], out=out, with_why=bool(cfg["redo_with_why"])))
            r["redo"] = str(out) if (out / "requests.jsonl").is_file() else "none"
            if r["redo"] != "none":
                st["rounds"].append(
                    {
                        "name": out.name,
                        "dir": str(out),
                        "kind": "redo",
                        "state": "built",
                        "built": now_utc().isoformat(),
                    }
                )
        if r["state"] == "built":
            submit(root, cfg, st, r, net)
        save(root, st)


def withdraw(root: Path, name: str, why: str) -> int:
    """Withdraw a round that was never submitted (built): state withdrawn, WITHDRAWN file in the round folder; it is
    never submitted, even when a key appears."""
    cfg, st = load(root)
    r = next((x for x in st["rounds"] if x["name"] == name), None)
    if r is None or r["state"] != "built" or not why.strip():
        die(f"cannot withdraw — round {name} state {r and r['state']} (built only) · reason {why!r} (not empty)")
    at = now_utc().isoformat()
    (Path(r["dir"]) / "WITHDRAWN").write_text(json.dumps({"why": why, "at": at}, ensure_ascii=False) + "\n")
    r.update(state="withdrawn", withdrawn=at, why=why)
    save(root, st)
    print(f"{MARK} track {name} withdrawn — {why}", flush=True)
    return 0


def void(root: Path, name: str, why: str) -> int:
    """Void a submitted round (e.g. submitted in a form the model rejects): submitted / fetched / validated only;
    state void, VOID file (reason, time). A void round leaves in-flight, pending money, merge and redo, and its clips
    are built again. A void redo clears its parent's redo mark (the parent builds a new redo)."""
    cfg, st = load(root)
    r = next((x for x in st["rounds"] if x["name"] == name), None)
    if r is None or r["state"] not in ("submitted", "fetched", "validated") or not why.strip():
        die(
            f"cannot void — round {name} state {r and r['state']} (submitted · fetched · validated only) · "
            f"reason {why!r}"
        )
    at = now_utc().isoformat()
    (Path(r["dir"]) / "VOID").write_text(
        json.dumps({"why": why, "at": at, "was": r["state"]}, ensure_ascii=False) + "\n"
    )
    r.update(state="void", voided=at, why_void=why)
    for p in st["rounds"]:
        if p.get("redo") == r["dir"]:
            p.pop("redo")
    save(root, st)
    print(f"{MARK} track {name} void — {why}", flush=True)
    return 0


def spent(st: dict) -> dict:
    """Money spent per round and in total, and pending (est_usd of submitted rounds)."""
    by = {r["name"]: round(CTA.folder_usd(Path(r["dir"])), 4) for r in st["rounds"]}
    pending = round(sum(float(r.get("est_usd") or 0) for r in st["rounds"] if r["state"] == "submitted"), 4)
    return {"by_round": by, "total": round(sum(by.values()), 4), "pending": pending}


def target_pairs(cfg: dict, root: Path) -> list[tuple[str, str]]:
    """Target (dataset, clip) in order = the pairs file ∪ clips present in the final datasets (order.final_targets,
    read every pass, when frames is on). The same clip under another dataset dies."""
    rows = list(
        dict.fromkeys(tuple(x.split("\t")[:2]) for x in Path(cfg["pairs"]).read_text().splitlines() if x.strip())
    )
    if cfg["frames"]:
        from dexengine.record.text import finals as FIN

        pk = {TT.clip_key(c): d for d, c in rows}
        ft = ORD.final_targets(FIN.orig_finals())
        if bad := [(d, c, pk[c]) for d, c in ft if pk.get(c, d) != d]:
            die(f"{len(bad)} final clips under another dataset than in the target list — {bad[:3]}")
        rows += [(d, c) for d, c in ft if c not in pk]
    return rows


def submit(root: Path, cfg: dict, st: dict, r: dict, net: str | None) -> None:
    """Submit a built round: every record must be a target (else it dies); then the key, time and budget gates
    (a round that does not pass them stays built and the reason is printed)."""
    d = Path(r["dir"])
    stray = sorted({(x["dataset"], x["clip"]) for x in read_jsonl(d / "records.jsonl")} - set(target_pairs(cfg, root)))
    if stray:
        die(f"track {r['name']}: {len(stray)} records outside the targets — {stray[:3]}")
    until = dt.datetime.fromisoformat(cfg["submit_until"])
    if net:
        print(f"{MARK} track {r['name']} submit skipped — {net}", flush=True)
        return
    if now_utc() > until:
        print(f"{MARK} track {r['name']} submit skipped — past {cfg['submit_until']}", flush=True)
        return
    est = json.loads((d / "estimate.json").read_text())["models"][
        json.loads((d / "manifest.json").read_text())["model"]
    ]["batch_usd"]
    sp = spent(st)
    if sp["total"] + sp["pending"] + est > float(cfg["budget_usd"]):
        print(
            f"{MARK} track ALERT {r['name']} submit stopped — spent ${sp['total']:.2f} + pending "
            f"${sp['pending']:.2f} + estimate ${est:.2f} > budget ${cfg['budget_usd']}",
            flush=True,
        )
        return
    if cfg["via"] == "sync":  # synchronous into raw.jsonl of the same folder; no batch, fetched next pass
        CTA.cmd_run(ns(dir=d, limit=10**9, workers=int(cfg["workers"]), usd_cap=None))
    elif cfg["via"] == "batch":
        CTA.cmd_submit(ns(dir=d, force=False))
    else:
        die(f"track via={cfg['via']!r} — batch | sync only")
    r.update(state="submitted", submitted=now_utc().isoformat(), est_usd=est)


def merge(root: Path, cfg: dict, st: dict) -> Path | None:
    """All validated rounds (later wins) -> one final table (rows outside the targets are left out)."""
    dirs = [r["dir"] for r in st["rounds"] if r["state"] == "validated"]
    if not dirs:
        return None
    out = root / "final" / "texts.jsonl"
    tg = root / "final" / "targets.tsv"  # targets = pairs ∪ final clips (target_pairs), rewritten every pass
    tg.parent.mkdir(parents=True, exist_ok=True)
    tg.write_text("".join(f"{d}\t{c}\n" for d, c in target_pairs(cfg, root)))
    CTX.cmd_merge(ns(dir=dirs, out=out, pairs=str(tg)))
    return out


def split_final(final: Path | None) -> dict[str, int]:
    """(2) The final table -> per dataset the data layer collect_text SPLIT_FILE (a split of the table, with
    SPLIT_SOURCE = source table path, sha, rows); same sha = skipped; a dataset without rows loses its output. No
    table yet (no validated round) = nothing touched. Only outputs this table wrote are touched: an output without a
    source record, or written from another table, dies. A row of an unregistered dataset dies. Returns {dataset:
    rows}."""
    import hashlib

    from dexengine.core import paths as P

    if final is None:
        return {}
    rows, me = read_jsonl(final), str(final.resolve())
    sha = hashlib.sha256(final.read_bytes()).hexdigest()
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)
    if bad := sorted(set(by) - set(P.data_layout.DATASETS)):
        die(f"datasets {bad} of the final table are not registered (data_layout.DATASETS)")
    for ds in sorted(P.data_layout.DATASETS):
        d = P.layer_path("collect_text", ds)
        src = d / SPLIT_SOURCE
        if (d / SPLIT_FILE).exists() or src.exists():
            owner = json.loads(src.read_text()).get("source") if src.is_file() else None
            if owner != me:
                die(
                    f"{d}: text output of table {owner} != this track {me} — two writers for one place "
                    "(neither deleted nor overwritten)"
                )
        if ds not in by:
            for f in (d / SPLIT_FILE, src):
                f.unlink(missing_ok=True)
            continue
        if src.is_file() and json.loads(src.read_text()).get("sha256") == sha:
            continue
        d.mkdir(parents=True, exist_ok=True)
        for f, text in (
            (d / SPLIT_FILE, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in by[ds])),
            (
                src,
                json.dumps(
                    {"source": me, "sha256": sha, "rows": len(by[ds]), "at": now_utc().isoformat()},
                    ensure_ascii=False,
                    indent=1,
                ),
            ),
        ):
            tmp = f.with_name(f.name + ".tmp")
            tmp.write_text(text)
            tmp.replace(f)
    n = {ds: len(v) for ds, v in sorted(by.items())}
    print(f"{MARK} split datasets {len(n)} · rows {sum(n.values())} = table {len(rows)} {n}", flush=True)
    if sum(n.values()) != len(rows):
        die("split count mismatch")
    return n


def unresolved(root: Path, keys: set, prompt: str, models: set) -> set:
    """Clips left out of new rounds (the attempt cap): rejected (last rejection in merge_rejects.jsonl) with the
    current prompt (prompt_sha) and a current model. Clips rejected under an older prompt or another model are built
    once more; batch-error rejections (no prompt_sha) stay unresolved."""
    return {
        TT.clip_key(r["clip"])
        for r in read_jsonl(Path(root) / "final" / "merge_rejects.jsonl")
        if r.get("prompt_sha") is None or (r.get("prompt_sha") == prompt and r.get("model") in models)
    } & keys


def _round_records(todo, work, cfg, root, cnt, recs):
    """Run the sources CLI on this round's pairs: its records are appended, missing-tree and frame-waiting counts go
    into cnt."""
    if todo:
        argv = [
            sys.executable,
            "-m",
            "dexengine.record.text.sources",
            "--pairs",
            str(work / "pairs.tsv"),
            "--data-root",
            cfg["data_root"],
            "--out",
            str(work / "records.jsonl"),
            *(["--with-frames"] if cfg["frames"] else []),
            "--views",
            cfg["views"],
            "--frame-size",
            cfg["frame_size"],
            "--frames-dir",
            str(root / "frames"),
        ]
        if cfg["frames"]:  # renders = the final datasets
            from dexengine.record.text import finals as FIN

            for g in FIN.orig_finals():
                argv += ["--dataset-glob", str(g)]
        if not cfg["text_only"]:
            argv.append("--frames-only")
        rc = subprocess.run(argv, cwd=str(REPO_ROOT)).returncode
        if rc:
            die(f"sources rc {rc}")
        recs += read_jsonl(work / "records.jsonl")
        cnt["no_tree"] = sum(1 for x in (work / "missing.tsv").read_text().splitlines() if x.strip())
        fm = work / "frames_missing.tsv"
        cnt["waiting"] = (
            sum(1 for x in fm.read_text().splitlines() if x.strip()) if fm.is_file() and not cfg["text_only"] else 0
        )
    return recs


def new_round(root: Path, cfg: dict, st: dict, net: str | None) -> dict:
    """(4) New round: targets - table - in flight - unresolved -> sources -> build -> submit. Returns the counts of
    the end line."""
    pairs = target_pairs(cfg, root)
    keys = {TT.clip_key(c) for _, c in pairs}
    if len(keys) != len(pairs):
        die(f"the target list has {len(pairs) - len(keys)} clip names under two datasets (key = clip name)")
    final = root / "final" / "texts.jsonl"
    done = {TT.clip_key(r["clip"]) for r in read_jsonl(final)} & keys
    inflight = set()
    for r in st["rounds"]:
        if r["state"] not in ("validated", "withdrawn", "void"):  # clips of withdrawn / void rounds are built again
            inflight |= {TT.clip_key(x["clip"]) for x in read_jsonl(Path(r["dir"]) / "records.jsonl")}
    inflight = (inflight & keys) - done
    unres = unresolved(root, keys, CTX.prompt_sha(), {cfg["model"], cfg["redo_model"]}) - done - inflight
    todo = [(d, c) for d, c in pairs if TT.clip_key(c) not in done | inflight | unres]
    n0, todo = len(todo), ORD.cap(todo, cfg["round_cap"])
    k = len(st["rounds"])
    work = root / "work" / f"w{k:03d}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "pairs.tsv").write_text("".join(f"{d}\t{c}\n" for d, c in todo))
    cnt = {
        "target": len(pairs),
        "table": len(done),
        "inflight": len(inflight),
        "unresolved": len(unres),
        "waiting": 0,
        "no_tree": 0,
        "ready": 0,
        "built": 0,
        "capped": n0 - len(todo),
    }
    recs = []
    recs = _round_records(todo, work, cfg, root, cnt, recs)
    cnt["ready"] = len(recs)
    until = dt.datetime.fromisoformat(cfg["submit_until"])
    need = 1 if until - now_utc() < dt.timedelta(hours=6) else int(cfg["min_batch"])
    if len(recs) < need:
        print(f"{MARK} track new clips {len(recs)} < {need} — nothing built this pass", flush=True)
        return cnt
    out = root / "rounds" / f"r{k:03d}"
    CTX.build_dir(recs, out, cfg["model"], extra={"track_round": k})
    r = {"name": out.name, "dir": str(out), "kind": "main", "state": "built", "built": now_utc().isoformat()}
    st["rounds"].append(r)
    save(root, st)
    submit(root, cfg, st, r, net)
    save(root, st)
    cnt["built"] = len(recs)
    return cnt


def tick(root: Path, dry: bool = False) -> int:
    """One pass; exit 3 with a STOP file when a key or credits are needed."""
    try:
        return _tick(Path(root), dry)
    except CTA.KeyNeeded as e:
        halt(Path(root), f"key needed — {e.code} · key file {CTA.KEY_FILE} (mode 600) or env ANTHROPIC_API_KEY")
        return 3


def _tick(root: Path, dry: bool) -> int:
    cfg, st = load(root)
    net = net_ok(dry)
    if net:
        print(f"{MARK} track network stages skipped — {net}", flush=True)
    advance(root, cfg, st, net)
    final = merge(root, cfg, st)
    split_final(final)
    CSTORE.tick_line(root, cfg, final, target_pairs(cfg, root))
    CSTORE.retext(root, cfg, st, final, save)
    cnt = new_round(root, cfg, st, net)
    sp = spent(st)
    parts = (
        cnt["table"]
        + cnt["inflight"]
        + cnt["waiting"]
        + cnt["unresolved"]
        + cnt["no_tree"]
        + cnt["ready"]
        + cnt["capped"]
    )
    print(
        f"{MARK} track targets {cnt['target']} = table {cnt['table']} + in flight {cnt['inflight']} + waiting for "
        f"render {cnt['waiting']} + unresolved {cnt['unresolved']} + no tree {cnt['no_tree']} + ready "
        f"{cnt['ready']} (built now {cnt['built']}) + after cap {cnt['capped']} · spent ${sp['total']:.2f} + "
        f"pending ${sp['pending']:.2f} / budget ${cfg['budget_usd']}",
        flush=True,
    )
    if parts != cnt["target"]:
        die(f"track count mismatch {parts} != targets {cnt['target']}")
    return 0


# ───────────────────────── CLI ─────────────────────────
def track_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments of one pipeline call (also the `track` sub-command of generate)."""
    ap.add_argument("--track", required=True, help="track name — root = run layer run_collect_text/<name>")
    ap.add_argument("--dry", action="store_true", help="without the network stages (status, fetch, submit)")
    ap.add_argument(
        "--withdraw", default="", help="instead of a pass: withdraw one unsubmitted (built) round (--why required)"
    )
    ap.add_argument("--print-root", action="store_true", help="print the track root only")
    ap.add_argument(
        "--void",
        default="",
        help="instead of a pass: void one submitted round (--why required; its clips are built again)",
    )
    ap.add_argument("--why", default="", help="reason of --withdraw / --void (not empty; kept in WITHDRAWN / VOID)")
    return ap


def run_track(a) -> int:
    """Run parsed track arguments: print the root, withdraw, void, or one pass."""
    root = track_root(a.track)
    if a.print_root:
        print(root)
        return 0
    if a.withdraw and a.void:
        die("--withdraw and --void are exclusive")
    return (
        withdraw(root, a.withdraw, a.why) if a.withdraw else void(root, a.void, a.why) if a.void else tick(root, a.dry)
    )


def main(argv=None) -> int:
    """CLI of the text stage: one pipeline pass (or withdraw / void / print-root)."""
    ap = track_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    config.add_data_root_arg(ap)
    a = ap.parse_args(argv)
    config.apply_data_root(a)
    return run_track(a)


if __name__ == "__main__":
    sys.exit(main())
