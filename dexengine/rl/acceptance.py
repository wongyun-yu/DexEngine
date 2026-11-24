"""RL campaign results: the per-hand clip-owner ledger (reader) and the result tables of a campaign (collect).

owner ledger  records/hands/<hand>/tables/rl_clip_owner.tsv under an owner root (ledger_path; the canonical
              copy is canon_ledger_path): one row per unique (axis, clip) of the hand, its state (STATES) and, when
              canon, the checkpoint to infer it with (ckpt_rel; best · last copy the owner bundle's judge record).
              read_ledger is its one reader (another head raises); use_rows is the consumers' choosing entry:
              {checkpoint: canon clips} of requested clips, every row re-read (canon · checkpoint file there). The
              data a checkpoint meets is proven where it is loaded ([CANON_USE], dexengine.rl.checkpoints).
collect       every bundle of a hand's campaign manifest → run_rl_results (per bundle: run clips · primaries · dups
              · state · park_last3 · the finals · cell · clip table · pass10 k/n · best · last iter:AGG · judge
              NE×epc), run_rl_results_clips (per unique clip: primary and dup bundles · best park_last3 of the done
              bundles with it · pass10 1|0|- from clip_ten) and run_rl_results_ds (unique clips · mean of the
              clips' best values · pass10 k/n from ten_total). Rows = manifest bundles (a set match); clip rows =
              the manifest target (a dup never counts twice).
"""

from __future__ import annotations

import collections
import json
import os
import re
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError, input_missing
from dexengine.core.paths import clip_name
from dexengine.rl import cell_files as CF
from dexengine.rl import cell_state as CH
from dexengine.rl import cells as CL
from dexengine.rl import root_guard as HR
from dexengine.rl import run_queue as CQ
from dexengine.rl.judge import record as CX

TAG = "[CLIP_OWNER]"
SCHEMA = 3
FILE = "rl_clip_owner.tsv"
OLD_HEAD = (
    "axis",
    "clip",
    "dataset",
    "bundle",
    "iter",
    "kind",
    "tag",
    "ckpt",
    "complete",
    "total",
    "rate_pct",
    "bundles",
    "waiting",
)
HEAD = OLD_HEAD + (
    "state",
    "gen",
    "campaign",
    "run",
    "ckpt_rel",
    "best",
    "last",
    "old",
    "turn",
    "static_add",
    "train_gen",
    "check_gen",
)
#: display word of each ledger state (the ledger stores the keys; the campaign pages carry these words, so they are
#: stored values: canon = canonical · pending · waiting for retraining · held · dropped)
STATES = {
    "canon": "canonical",
    "pending": "pending",
    "regen_wait": "waiting for retraining",
    "held": "held",
    "dropped": "dropped",
}


def die(msg: str) -> None:
    """Raise SystemExit with the ledger's tag."""
    raise SystemExit(f"{TAG} {msg}")


# ── owner ledger (reader)
# ──────────────────────────────────────────────────────────────────────────────────────────────


def ledger_path(hand: str, root: Path) -> Path:
    """The hand's owner-ledger file under an owner root (module doc)."""
    return P.run_path("exp_table", hand, root=root) / FILE


def canon_ledger_path(hand: str) -> Path:
    """The canonical owner ledger: ledger_path under paths.owner_root()."""
    return ledger_path(hand, P.owner_root())


def read_ledger(path: Path) -> list[dict]:
    """The owner ledger's rows; another head (an older schema) or a row of another width raises SystemExit."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != HEAD:
        die(f"{path}: head ≠ schema {SCHEMA} {HEAD} — an old ledger (retry after the next tick)")
    out = []
    for ln in lines[1:]:
        c = ln.split("\t")
        if len(c) != len(HEAD) or c[HEAD.index("state")] not in STATES:
            die(f"{path}: row {ln[:80]!r} — columns {len(HEAD)} · state ∈ {tuple(STATES)}")
        out.append(dict(zip(HEAD, c, strict=True)))
    return out


def use_rows(path: Path, clips: list[tuple[str, str]], data_root: Path) -> dict[tuple, list[str]]:
    """{(campaign, bundle, run, ckpt_rel): [clips]} for the requested (axis, clip)s: every row canon and its
    checkpoint file at data_root/ckpt_rel (module doc), else SystemExit (the ledger is stale). Prints one [CANON_USE]
    choosing line (checkpoints c · clips n/n)."""
    rows = {(r["axis"], r["clip"]): r for r in read_ledger(path)}
    out: dict[tuple, list[str]] = collections.defaultdict(list)
    bad, gap = [], []  # gap[i]: bad[i] is a missing row or checkpoint (an input-missing exit when shown)
    for k in clips:
        r = rows.get(k)
        if r is None or r["state"] != "canon":
            bad.append(f"{k[0]}:{k[1]} {'not in the ledger' if r is None else STATES[r['state']]}")
            gap.append(r is None)
            continue
        ck = Path(data_root) / r["ckpt_rel"]
        if not ck.is_file():
            bad.append(f"{k[1]} checkpoint missing {r['ckpt_rel']}")
            gap.append(True)
            continue
        out[(r["campaign"], r["bundle"], r["run"], r["ckpt_rel"])].append(k[1])
    if bad:
        msg = f"ledger stale — retry after the next tick: {len(bad)} {bad[:4]}"
        die(input_missing(msg) if any(gap[:4]) else msg)
    print(f"[CANON_USE] picked checkpoints {len(out)} · clips {sum(map(len, out.values()))}/{len(clips)}", flush=True)
    return dict(out)


# ── per-clip aggregation of a campaign's results
# ───────────────────────────────────────────────────────────────────────
class Slots:
    """The slot states one aggregation reads (clip_best · clip_ten): turned = {(bundle id, axis, clip)} of
    old-generation slots (out of every value and OR) · closed = {(bundle id, axis, clip)} of yielded slots of closed
    bundles (out of the same, never a primary) · wait = {(axis, clip)} whose every slot is turned (the only clips
    allowed
    no current primary bundle) · spare = {(bundle id, axis, clip)} of lineage slots that are live but never the clip's
    primary · source = where the states came from (a marker word; required)."""

    def __init__(self, turned=(), wait=(), source: str = "", closed=(), spare=()):
        if not source:
            raise ValueError("Slots: source (where the slot states were read) is missing")
        self.turned, self.wait, self.source = frozenset(turned), frozenset(wait), source
        self.closed, self.spare = frozenset(closed), frozenset(spare)

    def live(self, bid: str, axis: str, clip: str) -> bool:
        k = (bid, axis, clip_name(clip))
        return k not in self.turned and k not in self.closed


def slots_not_read(owner: str, expires: str) -> Slots:
    """Slots of a path that does not read the TURNOVER declarations: every slot is live; the source word names
    the owner and the expiry of that state."""
    return Slots(source=f"declarations not read · owner {owner} · expires {expires}")


def clip_weighted(xs: list[tuple[int, str, float | None]]) -> tuple[float | None, int]:
    """(clip-weighted mean park_last3 over the done entries with a value, their clips) of [(clips, state,
    value)]."""
    dn = [(n, v) for n, st, v in xs if st == "done" and v is not None]
    return (sum(n * v for n, v in dn) / sum(n for n, _v in dn) if dn else None), sum(n for n, _v in dn)


def clip_best(bundles: list[dict], res: dict[str, dict], slots: Slots) -> dict[tuple[str, str], dict]:
    """{(axis, clip): dataset · primary bundle · dup bundles · done bundles · best (value, bundle) or None}, every
    unique clip once ((axis, clip) is the manifest's key): its value is the best park_last3 among the done bundles that
    trained it on its current generation, primary or dup. `res` = {bundle id: bundle_result}. Turned slots are dropped
    first (Slots); then a clip without exactly one current primary bundle raises; none is allowed only for a clip in
    slots.wait (out of this table: no value)."""
    per: dict[tuple[str, str], list] = collections.defaultdict(list)
    for b in bundles:
        for c in b["clips"]:
            if slots.live(b["id"], b["axis"], c):
                spare = (b["id"], b["axis"], clip_name(c)) in slots.spare  # read like a dup (never the primary)
                per[(b["axis"], c)].append((b["id"], c in b["dups"] or spare, b["dataset"]))
    out = {}
    for k, where in sorted(per.items()):
        prim = [bid for bid, d, _ds in where if not d]
        if len(prim) != 1:
            raise ValueError(f"{k[0]}:{k[1]}: primary bundles {prim} — one primary bundle per clip")
        done = [
            (res[bid]["value"], bid)
            for bid, _d, _ds in where
            if res[bid]["state"] == "done" and res[bid]["value"] is not None
        ]
        out[k] = {
            "dataset": where[0][2],
            "primary": prim[0],
            "dups": [bid for bid, d, _ds in where if d],
            "done": len(done),
            "best": max(done) if done else None,
        }
    zero = sorted({(b["axis"], c) for b in bundles for c in b["clips"]} - set(out) - slots.wait)
    if zero:
        raise ValueError(
            f"clips without a current slot {len(zero)} {zero[:3]} — only clips waiting for retraining or "
            f"dropped (slots.wait) may have 0"
        )
    return out


def clip_mean(cb: dict, dataset: str | None = None) -> tuple[float | None, int]:
    """clip_weighted over the unique clips of `clip_best` (one dataset, or all): each clip weight 1 at its best."""
    return clip_weighted(
        [
            (1, "done" if v["best"] else "-", v["best"][0] if v["best"] else None)
            for v in cb.values()
            if dataset is None or v["dataset"] == dataset
        ]
    )


def clip_ten(bundles: list[dict], res: dict[str, dict], slots: Slots) -> dict[tuple[str, str], dict]:
    """{(axis, clip): dataset · judged [bundle ids whose ten_pct names it] · passed (bool)} over the done bundles
    with a ten_pct (per clip [complete, total, pass]), every unique clip once: it passes when any bundle that trained it
    on
    its current generation (primary or dup) passes it (an OR over bundles, never a sum; a turned slot is not in the OR).
    A ten_pct naming a clip its bundle does not train raises; names are compared as clip names (paths.clip_name)."""
    out: dict[tuple[str, str], dict] = {}
    for b in bundles:
        r = res[b["id"]]
        t = r.get("ten_pct") if r["state"] == "done" else None
        if not t:
            continue
        stray = sorted({clip_name(c) for c in t["clips"]} - {clip_name(c) for c in b["clips"]})
        if stray:
            raise ValueError(f"{b['id']}: ten_pct clips outside the bundle {stray[:3]}")
        for c, v in t["clips"].items():  # the judge may name trees (_pl_rt) — one clip namespace
            if not slots.live(b["id"], b["axis"], c):
                continue
            e = out.setdefault((b["axis"], clip_name(c)), {"dataset": b["dataset"], "judged": [], "passed": False})
            e["judged"].append(b["id"])
            e["passed"] = e["passed"] or bool(v[2])
    return out


def ten_text(t: dict | None) -> str:
    """A ten_pct or ten_total as the tables write it: "k/n", "-" = not measured (never 0)."""
    return "-" if not t else f"{t['k']}/{t['n']}"


def ten_total(ct: dict, dataset: str | None = None) -> dict | None:
    """{k passed unique clips, n judged unique clips, bundles} of `clip_ten` (one dataset, or all); None = none
    judged (not measured, never 0)."""
    xs = [v for v in ct.values() if dataset is None or v["dataset"] == dataset]
    return (
        {"k": sum(v["passed"] for v in xs), "n": len(xs), "bundles": len({b for v in xs for b in v["judged"]})}
        if xs
        else None
    )


# ── result tables (collect)
# ────────────────────────────────────────────────────────────────────────────────────────────
def manifest_of(hand: str, campaign: str, root=None) -> dict:
    """The campaign's frozen manifest copy (cells.manifest_dir), the one the loop launched from."""
    f = CL.manifest_dir(CL.campaign_dir(hand, campaign, root)) / "manifest.json"
    if not f.is_file():
        raise InputMissingError(f"[cells] no manifest copy in the campaign cell: {f}")
    return json.loads(f.read_text(encoding="utf-8"))


def bundle_of(man: dict, bid: str) -> dict:
    """Bundle `bid` of a manifest, with the manifest's hand."""
    b = next((b for b in man["bundles"] if b["id"] == bid), None)
    if b is None:
        raise InputMissingError(f"[cells] no bundle {bid} in the manifest")
    return dict(b, hand=man["hand"])


def judged(run: str, sdir: Path) -> dict:
    """{'value': park_last3 or None, 'finals': [(iter, ckpt, agg)] of OK f rows, 'p10': {iter: [k, n]} of those rows'
    pass10 column (absent column or '-' = not in the map)} from the sidecar's table."""
    tsv = CH.judge_files(run, sdir)[1]
    lines = tsv.read_text(encoding="utf-8").splitlines() if tsv.is_file() else []
    rows = [ln.split("\t") for ln in lines[1:]]
    fin = sorted((int(r[2]), r[3], float(r[5])) for r in rows if len(r) > 8 and r[1] == "f" and r[8] == "OK")
    pc = lines[0].split("\t").index("pass10") if lines and "pass10" in lines[0].split("\t") else None
    p10 = {
        int(r[2]): [int(x) for x in r[pc].split("/")]
        for r in rows
        if pc is not None and len(r) > pc and r[1] == "f" and r[8] == "OK" and re.fullmatch(r"\d+/\d+", r[pc])
    }
    v = CH.judge_verdict(run, sdir)
    val = float(CH.judge_files(run, sdir)[0].read_text(encoding="utf-8").split()[1]) if v and v[0] == "DONE" else None
    return {"value": val, "finals": fin, "p10": p10}


def dataset_key(b: dict) -> str:
    """The dataset cell a bundle's result rows go to: its one dataset, or data_layout.RL_MIXED when it mixes."""
    ds = CF.bundle_datasets(b)
    return next(iter(ds)) if len(ds) == 1 else P.data_layout.RL_MIXED


def bundle_result(hand: str, campaign: str, b: dict, root=None) -> dict:
    """One bundle as its cell says: run ('-' before launch) · state (cell_state.bundle_state; queued without a launch
    record) · value and finals (judged) · finalized · cell · ten_pct · special · special_pending · best · last · judge
    · extras_sha (judge.record.judge_extras, the record itself, finalized or not; None = not measured · '-' = no
    file)."""
    sdir = CL.bundle_dir(hand, campaign, b, root)
    rec = CH.cell_file(sdir, "launch")
    run = json.loads(rec.read_text(encoding="utf-8"))["run"] if rec.is_file() else "-"
    st = CH.bundle_state(sdir, run) if rec.is_file() else "queued"
    j, x = (
        (judged(run, sdir) if rec.is_file() else {"value": None, "finals": [], "p10": {}}),
        CX.judge_extras(run, sdir),
    )
    return {
        "run": run,
        "state": st,
        "value": j["value"],
        "finals": j["finals"],
        "p10": j["p10"],
        "cell": sdir,
        "ten_pct": x.ten_pct,
        "best": x.best,
        "last": x.last,
        "judge": x.judge,
        "special": x.special,
        "special_pending": x.special_pending,
        "extras_sha": x.sha256,
        "finalized": CH.cell_file(sdir, "finalized").is_file(),
    }


def collect(hand: str, campaign: str, root=None) -> tuple[Path, Path, int]:
    """Write run_rl_results, run_rl_results_clips and run_rl_results_ds (module doc) → (bundle table, dataset table,
    rows). A dup is trained in its bundle and counted there (bundle rows: run clips · primaries · dups); per clip and
    per dataset every clip counts once, at its best value (clip_best · clip_mean)."""
    bds, cb, ct, ds_rows, out, out_ds, rows = _collect_result_rows(hand, campaign, root)
    for key, sts in sorted(bds.items()):
        mean, clips_done = clip_mean(cb, key)
        c = collections.Counter(sts)
        ds_rows.append(
            (
                key,
                len(sts),
                sum(v["dataset"] == key for v in cb.values()),
                c["done"],
                c["failed"],
                len(sts) - c["done"] - c["failed"],
                "-" if mean is None else f"{mean:.2f}",
                clips_done,
                ten_text(ten_total(ct, key)),
            )
        )
    _write(
        out_ds,
        "dataset\tbundles\tclips_unique\tdone\tfailed\tother\tclip_best_park_last3\tclips_done\tpass10\n",
        ds_rows,
    )
    return out, out_ds, len(rows)


def _collect_result_rows(hand, campaign, root):
    """Result rows of every run of this hand/campaign, read from the manifest and the run folders."""
    man = manifest_of(hand, campaign, root)
    out = P.run_path("run_rl_results", hand, root=root, campaign=campaign)
    out_ds = P.run_path("run_rl_results_ds", hand, root=root, campaign=campaign)
    out_cl = P.run_path("run_rl_results_clips", hand, root=root, campaign=campaign)
    HR.assert_writable(out, hand, root=root)
    base = Path(root if root is not None else P.WT)
    rows, res, bds = [], {}, collections.defaultdict(list)
    for b in man["bundles"]:
        b = dict(b, hand=hand)
        r = res[b["id"]] = bundle_result(hand, campaign, b, root)
        key = dataset_key(b)
        rows.append(
            (
                b["id"],
                key,
                b["n"],
                CQ.collect_state(b, campaign, r["state"], r["run"]),
                int(r["finalized"]),
                "-" if r["value"] is None else f"{r['value']:.2f}",
                ",".join(f"{i}:{a:.2f}" for i, _c, a in r["finals"]) or "-",
                r["run"],
                str(r["cell"].relative_to(base)),
                b["n_primary"],
                len(b["dups"]),
                str((r["cell"] / CH.CELL_FILES["clips"]).relative_to(base)),
                ten_text(r["ten_pct"]),
                CX.pick_label(r["best"]),
                CX.pick_label(r["last"]),
                CX.judge_label(r["judge"]),
            )
        )
        bds[key].append(rows[-1][3])
    if {r[0] for r in rows} != {b["id"] for b in man["bundles"]} or len(rows) != len(man["bundles"]):
        raise ValueError("collected rows ≠ manifest bundles")
    sl = slots_not_read("rl_source", "when the collect reads TURNOVER")  # collect does not read TURNOVER
    cb = clip_best(man["bundles"], res, sl)
    target = sum(a["target"] for a in man["axes"])
    if len(cb) != target:
        raise ValueError(f"unique clips {len(cb)} ≠ target {target}")
    ct = clip_ten(man["bundles"], res, sl)  # 10 % completion per unique clip (OR over bundles)
    _write(
        out,
        "bundle\tdataset\tclips\tstate\tfinalized\tpark_last3\tfinals\trun\tcell\tprimary\tdups"
        "\tclips_table\tpass10\tbest\tlast\tjudge\n",
        rows,
    )
    _write(
        out_cl,
        "clip\tdataset\tprimary_bundle\tdup_bundles\tbundles_done\tbest_bundle_park_last3\tbest_bundle\taxis\tpass10\n",
        [
            (
                c,
                v["dataset"],
                v["primary"],
                ",".join(v["dups"]) or "-",
                v["done"],
                "-" if v["best"] is None else f"{v['best'][0]:.2f}",
                "-" if v["best"] is None else v["best"][1],
                ax,
                int(ct[(ax, c)]["passed"]) if (ax, c) in ct else "-",
            )
            for (ax, c), v in cb.items()
        ],
    )
    ds_rows = []
    return bds, cb, ct, ds_rows, out, out_ds, rows


def _write(path: Path, head: str, rows: list[tuple]) -> None:
    """Write a TSV atomically (temp file + replace)."""
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(head + "".join("\t".join(map(str, r)) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)
