"""Pieces of the trainer entry `dexengine/rl/cli.py`.

PRE-KIT modules (`inputs`, `robot`, `seqdir`) run before AppLauncher and import only the standard library and the
dependency-free dexengine registries (no numpy, torch or Isaac Lab; `dexengine.rl.preapp` checks this).
`preflight` (stdlib at import) runs the scene-input checks of the env build before Kit when the dry run asks for it,
and the process ends there.
POST-KIT module (`cameras`) is imported inside `create_env`, after Kit has started.
"""
