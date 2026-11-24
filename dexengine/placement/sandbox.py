"""Isolated stage-2 run (`scripts/stage2_ground.sh --sandbox DIR`).

The judge, the bake and the gate run exactly as in a canonical run, but the layers the chain **writes**
(`layout.SANDBOX_LAYERS`: trees, precompute, gate meta, contact derive, f0 shadows, views) resolve under DIR. The input
layers (base, retarget, MANO, reach maps, contact sources, wrist caches) are still read from the data root. The value
travels to every child process as env `DEXENGINE_SANDBOX_ROOT` (`layout.SANDBOX_ENV`), which the layer owner
(`paths.layer_path` / `paths.output_root`) reads.

    python -m dexengine.placement.sandbox init <dir|""> <tag> <dry 0|1>   prints the sandbox root ("" = canonical run)
    python -m dexengine.placement.sandbox prove <data root> <write path>…  prints the [SANDBOX] marker

Rules: a `--sandbox` value that differs from an inherited `DEXENGINE_SANDBOX_ROOT` dies (one input, two values); a test
tag (regress · canary — `rules.test_tag_kind`) runs only sandboxed; in a sandboxed run every write path must lie under
the root, checked before anything is written. Exit code 2 on any violation.
"""

from __future__ import annotations

import os
import sys

from dexengine.core import layout as DL

TAG = "[SANDBOX]"


def resolve(want: str, tag: str, dry: bool, env: dict | None = None, cwd: str | None = None) -> str:
    """The sandbox root of this run ("" = canonical). `want` = the --sandbox value (relative = under `cwd`); it is
    created. `env` = the inherited environment (default os.environ)."""
    env = os.environ if env is None else env
    inherited = env.get(DL.SANDBOX_ENV, "")
    root = inherited
    if want:
        p = want if os.path.isabs(want) else os.path.join(cwd or os.getcwd(), want)
        if inherited and os.path.realpath(inherited) != os.path.realpath(p):
            raise SystemExit(f"[stage2] ✗ --sandbox {p} differs from the inherited {DL.SANDBOX_ENV}={inherited}")
        try:
            os.makedirs(p, exist_ok=True)
        except OSError as e:
            raise SystemExit(f"[stage2] ✗ cannot create the sandbox root {p}: {e}") from None
        root = os.path.realpath(p)
    from dexengine.data import rules as RR  # noqa: PLC0415 — owner of the test-tag classes

    kind = RR.test_tag_kind(tag)
    if kind and not root and not dry:
        raise SystemExit(
            f"[stage2] ✗ test tag {tag} ({kind}) runs only sandboxed (--sandbox <dir>) — it must not write "
            "the canonical trees, caches, gate tables or f0 shadows"
        )
    return root


def prove(root: str, data_root: str, paths) -> str:
    """The marker line of this run; in a sandboxed run every write path must lie under `root` (else SystemExit)."""
    if not root:
        return f"{TAG} none — writes go to the canonical layers (data root {data_root})"
    base = os.path.realpath(root) + os.sep
    for p in paths:
        if not os.path.realpath(p).startswith(base):
            raise SystemExit(
                f"[stage2] ✗ sandboxed run, but the write path {p} lies outside the sandbox root {root} "
                "— nothing is written"
            )
    return f"{TAG} root={root} · write layers {','.join(DL.SANDBOX_LAYERS)} · input layers = data root {data_root}"


def main(argv: list[str]) -> int:
    if argv[:1] == ["init"] and len(argv) == 4:
        print(resolve(argv[1], argv[2], argv[3] == "1"))
        return 0
    if argv[:1] == ["prove"] and len(argv) >= 2:
        print(prove(os.environ.get(DL.SANDBOX_ENV, ""), argv[1], argv[2:]))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            sys.exit(2)
        raise
