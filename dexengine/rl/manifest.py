"""RL campaign manifest — one hand's whole training universe as bundles (the input of dexengine/rl/campaign.py).

Universe = every folded ledger axis of the hand (ADOPTED) minus the named exclusions below. Bundles = the selector's
rule (placement.train_set.select.partition: run-key pools · bundle_pack.PACK_RULE, with dups) — no bundle is built
here. Written: <out>/<hand>/manifest.json · lists/<bundle>.txt (the run's clips — DEXENGINE_SEQ_LIST) ·
lists/<bundle>.tsv (manifest_rules.clip_table — frozen into the run's launch record). Per bundle: id · clips · n ·
n_primary · dups {clip: primary bundle} · keys · families · run key/group · class · objects · launch inputs; per hand:
every ledger's sha256 (manifest_rules.manifest_fingerprint), the exclusions with their clips, the rule, cohesion
totals, the git commit.
R0 (a) at build: primaries == ADOPTED − excluded per axis, each once; any other occurrence of a clip is a recorded dup
  of its primary bundle (same run key), never twice in one bundle; all ≤ n — else it refuses to write.
  (b) `--verify <manifest>` (run before launching) re-hashes every ledger, re-derives the universe and re-measures
  what the bundles were built from — run key, object key and family per clip, and the per-axis tree fingerprint
  (manifest_rules.axis_fp: a tree re-baked under an unchanged ledger) — rc 1 on any difference.
A hand with no folded ledger dies. Small pools (< SMALL_POOL clips) are bundles like any other, never merged
  across datasets (flag `small`). Each bundle records `merge_key` and its scene departures (`scene_kinds`).
Plugin rules (`SiteRules`): a plugin may narrow the universe (an early round's clip list), cut partial-retrain
  lineages and name its generator. The release rules do none of that; a plugin's manifest writer passes its own. A
  manifest records its generator, and `verify` must run with the rules that built it.

Usage:
  DEXENGINE_ARM=rby1 python -m dexengine.rl.manifest --hand <hand> --out <dir> [--n 64] [--block 8]
      [--exclude-optional keypoint_glitch,unclassified] [--datasets <ds> …] [--data-root DIR]
  DEXENGINE_ARM=rby1 python -m dexengine.rl.manifest --verify <dir>/<hand>/manifest.json
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import subprocess
import sys
from pathlib import Path

from dexengine.core import config
from dexengine.core import datasets as RR
from dexengine.core import paths as P
from dexengine.placement import train_ready_util as TRU
from dexengine.placement.train_set import bundle_pack as BP
from dexengine.placement.train_set import select as ST
from dexengine.rl import manifest_rules as BR
from dexengine.rl.manifest_axes import SMALL_POOL, bundle_axis, sha256, tree_facts

SCHEMA = 3  # 3: object/family packing — dups · keys · families · list tables (refuses <= 2)
#: Stored values (the manifest's exclusion reasons keep their recorded text).
OUT_OF_SCOPE = (
    "outside the branch manifest scope (--datasets) — written as a per-clip exclusion so --verify derives "
    "the same universe"
)
#: Clip-level exclusions a caller may name with --exclude-optional — OFF by default. Each reads its producer's
#: statement in the BASE layer task_info: keypoint_glitch = the flag dexengine/data/filtering.py writes,
#: unclassified = case.case of the case taxonomy. Candidates are always listed in the manifest (applied true/false).
#: The reasons are stored values.
OPTIONAL_EXCLUDES = {
    "keypoint_glitch": (
        "source MANO keypoint glitch — keypoint_glitch in base quality.flags (producer reference_check)",
        lambda ti: "keypoint_glitch" in ((ti.get("quality") or {}).get("flags") or []),
    ),
    "unclassified": (
        "case unclassified — base case.case == UNCLASSIFIED",
        lambda ti: (ti.get("case") or {}).get("case") == "UNCLASSIFIED",
    ),
}
#: Campaign order of the run-group classes (easiest evidence first). `klass` derives the class from the group.
CLASS_ORDER = ("base", "idle", "uni", "shared", "aerial", "artic")
#: The manifest key of its data generation (absent = canonical).
GEN_KEY = "generation"


class SiteRules:
    """The release rules: the whole ADOPTED universe, no lineage, the data root's generation."""

    generator = "dexengine/rl/manifest.py"

    def narrow(self, hand: str, srcs: list[dict]) -> list[dict]:
        return srcs

    def pinned(self, man: dict, now: dict) -> list[str]:
        return []

    def gate(self, hand: str, lineage, srcs: list[dict], optional: set[str]) -> tuple:
        if lineage:
            raise SystemExit("[manifest] a lineage (partial retrain) manifest needs the site rules that cut it")
        return lineage, {}

    def lineage_excluded(self, lineage, axis: str, adopted: list[str], ex: dict) -> dict:
        return {}

    def lineage_record(self, lineage, hand: str, srcs: list[dict], bundles: list[dict]):
        return None

    def gen_fields(self, lineage) -> tuple:
        """(bundle-id tail, manifest keys): the data root's generation (data_layout.data_generation) — its name tails
        every id and its record is GEN_KEY; canonical: (None, {})."""
        g = P.data_layout.data_generation(str(P.main_root()))
        return (g and g["name"]), ({GEN_KEY: g} if g else {})


RELEASE = SiteRules()


def git_state() -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(P.WT), *a], capture_output=True, text=True, timeout=120).stdout.strip()

    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "tree": str(P.WT),
        "dirty": bool(g("status", "--porcelain", "--untracked-files=no")),
    }


def sources(hand: str, site: SiteRules = RELEASE) -> list[dict]:
    """The hand's universe sources: every folded ledger axis (none → dies; no frozen-roster fallback), narrowed
    by the plugin rules."""
    files = RR.ledger_files("*", hand, root=P.main_root())
    if not files:
        raise SystemExit(
            f"[manifest] {hand}: no folded ledger — fold the stage-2 chunk ledgers first (stage2_ground.sh --fold, or "
            f"DEXENGINE_HAND={hand} python3 dexengine/placement/status_fold.py --datasets <ds>)"
        )
    out = []
    for f in files:
        ax = RR.ledger_axis(f)
        ds, emb, _v = RR.ledger_axis_parts(ax)
        var = RR.ledger_axis_variant(ax)
        out.append(
            {
                "axis": ax,
                "dataset": ds,
                "emb": emb,
                "variant": var,
                "combo": RR.combo(hand, var),
                "adopted": TRU._adopted(str(f)),
                "source": f"ledger:{f}",
                "files": [{"path": str(f), "sha256": sha256(f)}],
            }
        )
    return site.narrow(hand, out)


def exclusions(src: dict, hand: str, optional: set[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """(excluded clip → reason, optional rule → candidate clips; key "_unreadable" = clips with no base task_info).
    Dataset exclusions (RR.campaign_excluded · --datasets) cover the whole axis. An optional rule on an unreadable
    clip dies."""
    out = RR.campaign_excluded()
    if src.get("out_of_scope") or src["dataset"] in out:
        why = f"dataset:{src['dataset']} — {out.get(src['dataset']) or OUT_OF_SCOPE}"
        return dict.fromkeys(src["adopted"], why), {}
    base = P.layer_path("base", src["dataset"], hand=RR.hand_dir(hand), emb=src["emb"])
    cand: dict[str, list[str]] = collections.defaultdict(list)
    for c in src["adopted"]:
        f = base / c / "task_info.json"
        if not f.is_file():
            cand["_unreadable"].append(c)
            continue
        ti = json.loads(f.read_text(encoding="utf-8"))
        for name, (_why, pred) in OPTIONAL_EXCLUDES.items():
            if pred(ti):
                cand[name].append(c)
    if optional and cand.get("_unreadable"):
        raise SystemExit(
            f"[manifest] {src['axis']}: {len(cand['_unreadable'])} clips without a base task_info — the "
            f"optional exclusions {sorted(optional)} cannot be decided (e.g. {cand['_unreadable'][:2]})"
        )
    out = {c: f"optional:{n} — {OPTIONAL_EXCLUDES[n][0]}" for n in sorted(optional) for c in cand.get(n, [])}
    return out, dict(cand)


def check_r0(src: dict, target: list[str], rows: list[dict], n: int) -> None:
    """R0 (a) for one axis (manifest_rules.r0_problems) — dies on any problem."""
    if bad := BR.r0_problems(target, rows, n):
        raise SystemExit(f"[R0] {src['axis']}: " + " · ".join(bad))


def build(
    hand: str,
    out: Path,
    n: int,
    block: int,
    optional: set[str],
    scope: set | None = None,
    lineage: dict | None = None,
    site: SiteRules = RELEASE,
) -> dict:
    """Build and write one hand's manifest under `out` (module doc); refuses a non-empty lists folder."""
    axes, bundles, cands, excl, gf, hdir, lin, lineage, rule, srcs, stamp, tot = _build_bundles(
        hand, out, n, block, optional, scope, lineage, site
    )
    man = dict(
        schema=SCHEMA,
        hand=hand,
        arm=RR.active_arm(),
        generated=stamp,
        generator=site.generator,
        git=git_state(),
        data_root=str(P.main_root()),
        mode="ledger",
        rule=rule,
        fingerprint=BR.manifest_fingerprint(srcs, BR.selection(n, block, optional, scope, lineage)),
        axes=axes,
        totals=tot,
        exclusions={
            "datasets": RR.campaign_excluded(),
            "optional_applied": sorted(optional),
            "optional_candidates": {
                k: {"reason": OPTIONAL_EXCLUDES[k][0], "applied": k in optional, "clips": cands.get(k, [])}
                for k in OPTIONAL_EXCLUDES
            },
            "optional_unreadable": cands.get("_unreadable", []),
            "clips": excl,
        },
        bundles=bundles,
        **({"lineage": lin} if lin else {}),
        **gf,
    )
    (hdir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def _build_bundles(hand, out, n, block, optional, scope, lineage, site):
    """Sources, axes, lineage and bundles of the manifest build."""
    if unknown := optional - set(OPTIONAL_EXCLUDES):
        raise SystemExit(
            f"[manifest] unknown optional exclusions {sorted(unknown)} — known {sorted(OPTIONAL_EXCLUDES)}"
        )
    srcs = [dict(s, out_of_scope=scope is not None and s["dataset"] not in scope) for s in sources(hand, site)]
    hdir = out / hand
    lists = hdir / "lists"
    if lists.exists() and any(lists.iterdir()):
        raise SystemExit(f"[manifest] {lists} is not empty — a frozen manifest is never overwritten (give a new --out)")
    rule = BR.rule_record(n, block, SMALL_POOL, CLASS_ORDER)
    lists.mkdir(parents=True, exist_ok=True)
    bundles, axes, excl, cands = [], [], [], {}
    gen, gf = site.gen_fields(lineage)
    lineage, defer = site.gate(hand, lineage, srcs, optional)
    for s in srcs:
        ex, cand = exclusions(s, hand, optional)
        ex.update(defer.get(s["axis"], {}))
        if lineage:
            ex.update(site.lineage_excluded(lineage, s["axis"], s["adopted"], ex))
        target = [c for c in s["adopted"] if c not in ex]
        rows, tree_fp = bundle_axis(s, target, hand, n, block, lists, gen)
        check_r0(s, target, rows, n)
        bundles += rows
        excl += [{"axis": s["axis"], "clip": c, "reason": why} for c, why in sorted(ex.items())]
        for k, v in cand.items():
            cands.setdefault(k, []).extend({"axis": s["axis"], "clip": c} for c in v)
        axes.append(
            {
                "axis": s["axis"],
                "dataset": s["dataset"],
                "emb": s["emb"],
                "variant": s["variant"],
                "combo": s["combo"],
                "source": s["source"],
                "files": s["files"],
                "adopted": len(s["adopted"]),
                "excluded": len(ex),
                "target": len(target),
                "bundles": len(rows),
                "tree_sha256": tree_fp,
                "obj_key": RR.DATASET_PROFILE[s["dataset"]].obj_key if target else None,
            }
        )
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    bundles.sort(key=lambda b: (order[b["klass"]], -b["pool"], b["id"]))
    tot = {
        "adopted": sum(a["adopted"] for a in axes),
        "excluded": len(excl),
        "target": sum(a["target"] for a in axes),
        "bundled": sum(b["n_primary"] for b in bundles),
        "dup_slots": sum(len(b["dups"]) for b in bundles),
        "slots": sum(b["n"] for b in bundles),
        "bundles": len(bundles),
        "cohesion": BR.cohesion(bundles),
        "by_class": {
            k: {
                "bundles": sum(b["klass"] == k for b in bundles),
                "clips": sum(b["n_primary"] for b in bundles if b["klass"] == k),
                "dups": sum(len(b["dups"]) for b in bundles if b["klass"] == k),
            }
            for k in CLASS_ORDER
        },
    }
    if tot["bundled"] != tot["target"] or tot["adopted"] - tot["excluded"] != tot["target"]:
        raise SystemExit(f"[R0] {hand}: totals disagree {tot}")
    lin = lineage and site.lineage_record(lineage, hand, srcs, bundles)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    return axes, bundles, cands, excl, gf, hdir, lin, lineage, rule, srcs, stamp, tot


def remeasure(man: dict) -> list[str]:
    """Re-measure what the bundles were built from, on the trees as they are NOW (an oracle — the manifest states,
    this measures): each clip's run key and object key, and each axis' tree fingerprint."""
    bad = []
    hd = RR.hand_dir(man["hand"])
    by_axis: dict[str, list[dict]] = collections.defaultdict(list)
    for b in man["bundles"]:
        by_axis[b["axis"]].append(b)
    for a in man["axes"]:
        bs = by_axis.get(a["axis"], [])
        prim = [c for b in bs for c in b["clips"] if c not in b["dups"]]
        if not prim:
            continue
        root = P.trees_dir(a["combo"], a["dataset"], a["emb"])
        fp, _kinds = tree_facts(root, prim)  # over primaries — the set the build fingerprinted
        if fp != a.get("tree_sha256"):
            bad.append(f"{a['axis']}: tree fingerprint differs (same ledger, trees baked again)")
        pools = ST.run_pools(prim, a["combo"], a["dataset"], a["emb"])
        rk_of = {c: rk for rk, cs in pools.items() for c in cs}
        family, _near = ST.key_shape(a["dataset"])
        keys: dict[str, str] = {}
        for cs in pools.values():
            keys.update(BP.rekey({c: ST.target_object(c, a["dataset"], a["emb"], hd) for c in cs}))
        for b in bs:
            if (got := sorted({rk_of[c] for c in b["clips"]})) != [b["run_key"]]:
                bad.append(f"{b['id']}: run key now {got} ≠ manifest {b['run_key']}")
            if (
                dict(collections.Counter(keys[c] for c in b["clips"])) != b["objects"]
                or {c: keys[c] for c in b["clips"]} != b["keys"]
            ):
                bad.append(f"{b['id']}: objects differ from the manifest")
            if {k: family(k) for k in b["families"]} != b["families"]:
                bad.append(f"{b['id']}: shape families differ from the manifest (surface points changed)")
    return bad


def verify(path: Path, site: SiteRules = RELEASE) -> list[str]:
    """R0 (b): the manifest still describes the data — every source re-hashed, universe re-derived, list files equal,
    and the build inputs re-measured on the trees (remeasure). [] = the same."""
    man = json.loads(path.read_text(encoding="utf-8"))
    if man.get("schema") != SCHEMA:
        return [f"schema {man.get('schema')} ≠ {SCHEMA} — an old manifest is rebuilt (no tree fingerprint or run key)"]
    want_rule = BR.rule_record(man["rule"]["n"], man["rule"]["block"], SMALL_POOL, CLASS_ORDER)
    bad = [] if man["rule"] == want_rule else [f"bundling rule record {man['rule']} ≠ code {want_rule} — rebuild"]
    now = {s["axis"]: s for s in sources(man["hand"], site)}
    bad += site.pinned(man, now)
    for a in man["axes"]:
        s = now.get(a["axis"])
        if s is None:
            bad.append(f"{a['axis']}: source is gone")
            continue
        if [f["sha256"] for f in s["files"]] != [f["sha256"] for f in a["files"]]:
            bad.append(f"{a['axis']}: source sha256 differs (the data generation moved)")
    bad += [f"{ax}: new axis not in the manifest" for ax in sorted(set(now) - {a["axis"] for a in man["axes"]})]
    excl = {(e["axis"], e["clip"]) for e in man["exclusions"]["clips"]}
    want = {(ax, c) for ax, s in now.items() for c in s["adopted"]} - excl
    got = [(b["axis"], c) for b in man["bundles"] for c in b["clips"] if c not in b["dups"]]
    if set(got) != want or len(got) != len(set(got)):
        bad.append(
            f"bundle primaries ≠ sources − exclusions: missing {len(want - set(got))} · outside "
            f"{len(set(got) - want)} · unrecorded dups {len(got) - len(set(got))}"
        )
    by_axis: dict[str, list[dict]] = collections.defaultdict(list)
    for b in man["bundles"]:
        by_axis[b["axis"]].append(b)
    for ax, rows in sorted(by_axis.items()):
        bad += [
            f"[R0] {ax}: {x}"
            for x in BR.r0_problems([c for r in rows for c in r["clips"] if c not in r["dups"]], rows, man["rule"]["n"])
        ]
        bad += [f"[pack] {ax}: {x}" for x in BR.pool_violations(rows, man["rule"]["n"])]
    for b in man["bundles"]:
        f, t = path.parent / b["list"], path.parent / b["table"]
        if not f.is_file() or f.read_text(encoding="utf-8").split() != b["clips"]:
            bad.append(f"{b['id']}: clip list file differs from the manifest ({f})")
        if not t.is_file() or t.read_text(encoding="utf-8") != BR.clip_table(b):
            bad.append(f"{b['id']}: clip table differs from the manifest ({t})")
    return bad + remeasure(man)


def summary(man: dict, out: Path) -> list[str]:
    """The two summary lines a build prints."""
    t = man["totals"]
    return [
        f"[manifest] {man['hand']}: sources {t['adopted']} − excluded {t['excluded']} = target {t['target']} · "
        f"bundles {t['bundles']} · bundled clips {t['bundled']} + dups {t['dup_slots']} · same object≥{BP.CO_MIN} "
        f"{t['cohesion']['same']} · same or similar≥{BP.CO_MIN} {t['cohesion']['similar']} / "
        f"{t['cohesion']['clips']} · fingerprint {man['fingerprint'][:12]} → {out / man['hand'] / 'manifest.json'}",
        "[manifest] by class "
        + " · ".join(f"{k} {v['bundles']} bundles/{v['clips']}" for k, v in t["by_class"].items()),
    ]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m dexengine.rl.manifest", description=__doc__.split("\n")[0])
    ap.add_argument("--hand", default="")
    ap.add_argument(
        "--out", type=Path, default=None, help="output root (required): <out>/<hand>/manifest.json · lists/"
    )
    ap.add_argument("--n", type=int, default=BR.N_DEFAULT, help=f"bundle size (default = the cap {BR.N_DEFAULT})")
    ap.add_argument("--block", type=int, default=BR.BLOCK_DEFAULT, help=f"object block (default {BR.BLOCK_DEFAULT})")
    ap.add_argument(
        "--exclude-optional",
        default="",
        help=f"optional exclusions (comma-separated, default none): {', '.join(OPTIONAL_EXCLUDES)}",
    )
    ap.add_argument("--datasets", nargs="+", choices=RR.ALL, default=None, help="scope (default all datasets)")
    ap.add_argument("--verify", type=Path, default=None, help="R0 (b): does the manifest match the data now (rc 0/1)")
    config.add_data_root_arg(ap)
    return ap


def run_verify(path: Path, site: SiteRules = RELEASE) -> int:
    bad = verify(path, site)
    for b in bad[:20]:
        print(f"[verify] ✗ {b}")
    print(f"[verify] {path}: {'same' if not bad else f'differs {len(bad)}'}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = parser()
    a = ap.parse_args(argv)
    config.apply_data_root(a)
    if a.verify:
        return run_verify(a.verify)
    if not a.hand or a.out is None:
        ap.error("--hand and --out are required (no default)")
    man = build(
        a.hand,
        a.out,
        a.n,
        a.block,
        {x for x in a.exclude_optional.split(",") if x},
        set(a.datasets) if a.datasets else None,
    )
    print("\n".join(summary(man, a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
