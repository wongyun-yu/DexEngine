"""Rules over campaign manifest rows (no data, light imports): the bundling rule record and defaults, the set
check over one axis' rows, the packer's post-conditions, the cohesion count, the content fingerprint and the axis tree
fingerprint.

The per-clip table of a
bundle row is dexengine.rl.cell_files (clip_rows · table_text) — the same table a finalized cell carries.
"""

from __future__ import annotations

import collections
import hashlib

from dexengine.placement.train_set import bundle_pack as BP
from dexengine.rl import cell_files as CF

#: The bundling defaults (the manifest CLI's defaults). A build at them adds nothing to the fingerprint.
N_DEFAULT, BLOCK_DEFAULT = 64, 8


def clip_table(b: dict) -> str:
    """A manifest bundle row's per-clip table (lists/<id>.tsv; frozen into the run's launch record)."""
    return CF.table_text(CF.clip_rows(b))


def _pack_cohesion(keys: dict[str, str], fam: dict[str, str], bundles: list[dict]) -> dict:
    """Per primary placement (each clip once, in its primary bundle): does its bundle hold >= CO_MIN clips of the same
    key (`same`) / of the same family (`similar` — same key included), counting dups as companions."""
    same = sim = tot = 0
    for b in bundles:
        run = list(b["clips"]) + list(b.get("dups") or {})
        kc = collections.Counter(keys[c] for c in run)
        fc = collections.Counter(fam[keys[c]] for c in run)
        for c in b["clips"]:
            tot += 1
            same += kc[keys[c]] >= BP.CO_MIN
            sim += fc[fam[keys[c]]] >= BP.CO_MIN
    return {"clips": tot, "same": same, "similar": sim}


def cohesion(bundles: list[dict]) -> dict:
    """_pack_cohesion over manifest rows (primaries; dups count as companions) — same key / same family."""
    tot = collections.Counter()
    for b in bundles:
        prim = [c for c in b["clips"] if c not in b["dups"]]
        tot.update(_pack_cohesion(b["keys"], dict(b["families"]), [{"clips": prim, "dups": b["dups"]}]))
    return dict(tot)


def r0_problems(target: list[str], rows: list[dict], n: int) -> list[str]:
    """R0 (a) over one axis' manifest rows: primaries == target (set difference both ways), each clip primary once;
    every other occurrence of a clip is a recorded dup whose origin is its primary bundle of the same run key, never
    twice in one bundle; n == len(clips) <= n."""
    got = [c for r in rows for c in r["clips"] if c not in r["dups"]]
    miss, extra = sorted(set(target) - set(got)), sorted(set(got) - set(target))
    home = {c: r for r in rows for c in r["clips"] if c not in r["dups"]}
    twice = [c for c, k in collections.Counter(got).items() if k > 1]
    checks = [
        ("missing", miss),
        ("outside", extra),
        ("unrecorded dup", twice),
        (
            "dup with a wrong origin",
            [
                f"{r['id']}:{c}"
                for r in rows
                for c, o in r["dups"].items()
                if c not in r["clips"] or c not in home or home[c]["id"] != o or home[c]["run_key"] != r["run_key"]
            ],
        ),
        ("twice in one bundle", [r["id"] for r in rows if len(r["clips"]) != len(set(r["clips"]))]),
        ("over n", [r["id"] for r in rows if r["n"] > n or r["n"] != len(r["clips"])]),
    ]
    return [f"{name} {len(v)} {v[:3]}" for name, v in checks if v]


def rule_record(n: int, block: int, small_pool: int, class_order) -> dict:
    """The rule a campaign manifest was built under (bundle_pack's constants + the manifest's) — `verify` compares the
    record with the code's (a rule change means a rebuild · n over N_DEFAULT dies)."""
    if n > N_DEFAULT:
        raise SystemExit(f"[PACK] bundle size n={n} > cap {N_DEFAULT} — no bundle over the cap")
    return {
        "version": BP.PACK_RULE,
        "n": n,
        "block": block,
        "rekey_min": BP.REKEY_MIN,
        "co_min": BP.CO_MIN,
        "fam_size_edges": list(BP.FAM_SIZE_EDGES),
        "fam_thin": BP.FAM_THIN,
        "nb_extra_max": BP.NB_EXTRA_MAX,
        "small_pool": small_pool,
        "class_order": list(class_order),
    }


def selection(n: int, block: int, optional: set[str], scope: set | None, lineage: dict | None) -> dict[str, str]:
    """What a manifest build takes from its ledgers' universe — each entry "" at the full build: lineage (gen ·
    declaration sha) · scope (the datasets) · optional (the optional exclusions) · n · block."""
    return {
        "lineage": f"{lineage['gen']}:{lineage['turnover_sha']}" if lineage else "",
        "scope": "" if scope is None else "datasets:" + ",".join(sorted(scope)),
        "optional": ",".join(sorted(optional)),
        "n": "" if n == N_DEFAULT else str(n),
        "block": "" if block == BLOCK_DEFAULT else str(block),
    }


def manifest_fingerprint(srcs: list[dict], sel: dict[str, str]) -> str:
    """A campaign manifest's content fingerprint: every source ledger's sha256 (manifest.sources), then each non-empty
    `selection` entry. The full build adds none — the ledgers' alone."""
    body = "".join(f"{s['axis']}:{f['sha256']}\n" for s in srcs for f in s["files"])
    body += "".join(f"{k}={v}\n" for k, v in sorted(sel.items()) if v)
    return hashlib.sha256(body.encode()).hexdigest()


def pool_violations(rows: list[dict], n: int) -> list[str]:
    """bundle_pack.violations over each run-key pool of one axis' manifest rows (the packer's post-conditions
    re-checked on what was written — block minimum · small keys whole · dup origins · size)."""
    out = []
    for rk in sorted({r["run_key"] for r in rows}):
        pr = [r for r in rows if r["run_key"] == rk]
        idx = {r["id"]: j for j, r in enumerate(pr)}
        keys = {c: r["keys"][c] for r in pr for c in r["clips"] if c not in r["dups"]}
        packed = [
            {
                "clips": [c for c in r["clips"] if c not in r["dups"]],
                "dups": {c: idx.get(o, -1) for c, o in r["dups"].items()},
            }
            for r in pr
        ]
        out += [f"{rk}: {v}" for v in BP.violations(keys, packed, n)]
    return out


def axis_fp(pairs) -> str:
    """The axis tree fingerprint: sha256 over `clip sha` lines in sorted clip order (pairs = (clip, sha))."""
    return hashlib.sha256("".join(f"{c} {s}\n" for c, s in sorted(pairs)).encode()).hexdigest()
