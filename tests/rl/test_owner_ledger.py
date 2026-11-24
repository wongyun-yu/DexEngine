"""The clip-owner ledger writer (dexengine.rl.owner_ledger) on synthetic finalized campaign cells: owner rule and
tie-break, a dup clip, pending bundles, and the round trip into the readers recording uses (read_ledger, use_rows,
record.sources canon). A ledger with plugin columns is never replaced."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dexengine.core.hashing import sha256_file
from dexengine.rl import acceptance as A
from dexengine.rl import cells as CL
from dexengine.rl import owner_ledger as OL

REPO = Path(__file__).resolve().parents[2]
HAND, CAMP = "wuji2", "c1"
HEAD = "time\tkind\titer\tckpt\tgpu\tagg_pct\tbatch_pct\tn_real\tstatus\tlog_or_reason\tpass10\n"


def _bundle(bid, clips, dups=None):
    keys = dict.fromkeys(clips, "k")
    return {
        "id": bid,
        "axis": "taco",
        "dataset": "taco",
        "clips": clips,
        "dups": dups or {},
        "n": len(clips),
        "n_primary": len(clips) - len(dups or {}),
        "keys": keys,
        "families": {"k": "f"},
    }


def _cell(root, b, judged=None):
    """A bundle cell; judged = {iter: {clip: (complete, total)}} makes it a finalized, judged cell."""
    sdir = CL.bundle_dir(HAND, CAMP, b, root)
    sdir.mkdir(parents=True)
    run, lid = f"R_{b['id']}_s42", "L1"
    (sdir / "launch.json").write_text(json.dumps({"run": run, "launch_id": lid}))
    if judged is None:
        return sdir
    (sdir / "train.rc").write_text("0\n")
    (sdir / f"{run}_judge.rc").write_text(f"DONE 50.00 pass10=1/1\nlaunch={lid}\n")
    rows, kept = [], []
    for it, per in sorted(judged.items()):
        ck = f"agent_{it * 32}.pt"
        rows.append(f"t\tf\t{it}\t{ck}\t0\t50.00\t-\t1\tOK\tlog\t1/1\n")
        kept.append(f"skrl/{run}/checkpoints/{ck}")
        (sdir / kept[-1]).parent.mkdir(parents=True, exist_ok=True)
        (sdir / kept[-1]).write_bytes(b"pt")
        (sdir / f"{run}_jf{it:04d}_clips.json").write_text(json.dumps({"clips": {c: list(v) for c, v in per.items()}}))
    (sdir / f"{run}_judge.tsv").write_text(HEAD + "".join(rows))
    last = max(judged)
    ten = {c: [v[0], v[1], int(v[0] > 0)] for c, v in judged[last].items()}
    (sdir / f"{run}_judge_extras.json").write_text(
        json.dumps(
            {
                "ten_pct": {"k": sum(v[2] for v in ten.values()), "n": len(ten), "clips": ten},
                "special": [],
                "special_pending": [],
            }
        )
    )
    (sdir / "FINALIZED.json").write_text(json.dumps({"kept": kept}))
    return sdir


@pytest.fixture
def campaign(data_root):
    a = _bundle("taco.both.01", ["c1", "c2"])
    b = _bundle("taco.both.02", ["c3", "c1"], dups={"c1": "taco.both.01"})
    p = _bundle("taco.both.03", ["c4"])
    man = {"hand": HAND, "bundles": [a, b, p]}
    mdir = CL.manifest_dir(CL.campaign_dir(HAND, CAMP, data_root))
    mdir.mkdir(parents=True)
    (mdir / "manifest.json").write_text(json.dumps(man))
    _cell(data_root, a, {1400: {"c1": (2, 4), "c2": (4, 4)}, 1600: {"c1": (3, 4), "c2": (4, 4)}})
    _cell(data_root, b, {1600: {"c3": (1, 4), "c1": (3, 4)}})  # ties a's 1600 on c1
    _cell(data_root, p)  # launched, not judged
    return data_root


def test_owner_rule_dup_and_pending(campaign):
    rows, n = OL.rows_of(HAND, [CAMP], campaign)
    by = {r["clip"]: r for r in rows}
    assert n == {"clips": 4, "canon": 3, "pending": 1, "checkpoints": 2}
    assert by["c1"]["bundle"] == "taco.both.01" and by["c1"]["iter"] == 1600  # tie at 3/4: bundle id first
    assert by["c1"]["bundles"] == "taco.both.01,taco.both.02" and by["c4"]["waiting"] == "taco.both.03"
    assert by["c2"]["iter"] == 1600 and by["c2"]["complete"] == 4  # tie 4/4: later iteration
    assert by["c4"]["state"] == "pending" and by["c4"]["ckpt_rel"] == "-"
    assert all(r[c] == "-" for r in rows for c in OL.SITE_COLS)


def test_round_trip_into_the_record_readers(campaign, tmp_path):
    owner = tmp_path / "owner"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "dexengine.rl.owner_ledger",
            "--hand",
            HAND,
            "--campaigns",
            CAMP,
            "--owner-root",
            str(owner),
            "--data-root",
            str(campaign),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO)},
    )
    assert r.returncode == 0, r.stderr[-600:]
    assert f"{A.TAG} {HAND} clips 4 · canon 3 · pending 1 · checkpoints 2 →" in r.stdout
    f = A.ledger_path(HAND, owner)
    assert len(A.read_ledger(f)) == 4
    use = A.use_rows(f, [("taco", "c1"), ("taco", "c2"), ("taco", "c3")], campaign)
    assert sum(map(len, use.values())) == 3 and all((campaign / k[3]).is_file() for k in use)
    with pytest.raises(SystemExit):
        A.use_rows(f, [("taco", "c4")], campaign)  # pending is not usable
    from dexengine.record import sources

    assert sources.canon(HAND, f)[1] == sha256_file(f)


def test_a_site_ledger_is_not_replaced(campaign, tmp_path):
    rows, _n = OL.rows_of(HAND, [CAMP], campaign)
    owner = tmp_path / "owner"
    f = OL.write(HAND, rows, owner)
    OL.write(HAND, rows, owner)  # its own output: replaced
    site = [dict(r, state="regen_wait") if r["clip"] == "c4" else r for r in rows]
    f.write_text("\t".join(A.HEAD) + "\n" + "".join("\t".join(str(r[k]) for k in A.HEAD) + "\n" for r in site))
    before = f.read_bytes()
    with pytest.raises(SystemExit) as e:
        OL.write(HAND, rows, owner)
    assert e.value.code == 3 and f.read_bytes() == before
    f.write_text(
        "\t".join(A.HEAD)
        + "\n"
        + "".join("\t".join(str(dict(r, gen="0123456789ab")[k]) for k in A.HEAD) + "\n" for r in rows)
    )
    assert OL.foreign(f) == ["site columns filled ['gen']"]
