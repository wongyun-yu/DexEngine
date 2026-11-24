"""Pieces of the trainer entry `dexengine/rl/cli.py`.

PRE-KIT modules (`inputs`, `robot`, `seqdir`) run before AppLauncher and import only the standard library and the
dependency-free dexengine registries (no numpy, torch or Isaac Lab; `dexengine.rl.preapp` checks this).
POST-KIT module (`cameras`) is imported inside `create_env`, after Kit has started.
"""
