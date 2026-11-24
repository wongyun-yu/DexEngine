"""Every trainer argv the launcher writes (the golden dumps in launch_cases/) parses with the trainer's own parser
(``dexengine.rl.args.build_parser``); the only flags it leaves are Isaac Lab's AppLauncher flags."""

from __future__ import annotations

from pathlib import Path

import pytest

from dexengine.rl.args import build_parser
from dexengine.rl.launch.argv import ENTRY

CASES = sorted((Path(__file__).parent / "launch_cases").glob("*.case"))
APP_LAUNCHER_FLAGS = {"--headless", "--enable_cameras"}


def trainer_argvs(case: Path) -> list[list[str]]:
    """The argv blocks (one token per line, ended by ENV:: lines or a blank line) of a case's dump section."""
    dump = case.read_text(encoding="utf-8").split("## dump", 1)[1].splitlines()[1:]
    out, cur = [], []
    for ln in dump + [""]:
        if ln.startswith("## "):
            break
        if ln.startswith("ENV::") or not ln.strip():
            if cur:
                out.append(cur)
            cur = []
            continue
        cur.append(ln)
    return [a for a in out + ([cur] if cur else []) if a[0].endswith(ENTRY.rsplit("/", 1)[-1])]


@pytest.mark.parametrize("case", CASES, ids=[c.stem for c in CASES])
def test_trainer_argv_parses(case):
    argvs = trainer_argvs(case)
    assert argvs, "no trainer argv in the dump"
    for argv in argvs:
        _ns, unknown = build_parser().parse_known_args(argv[1:])
        assert set(unknown) <= APP_LAUNCHER_FLAGS, unknown
