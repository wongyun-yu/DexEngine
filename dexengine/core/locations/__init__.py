"""The path layer behind `dexengine.core.paths` (which re-exports every name).

modules   owning modules (layout, qpos layout, registries, readers, converters), hand rig folders
names     clip / tree names, suffixes, clip ids, derived clips, IK-pool names
roots     code checkout, data root, run roots, cluster trees, staging roots, IK pool, retargeting runtime
hands     hand / arm ids, spec dirs, combos, rig hashes, per-hand records
layers    per-dataset data layers and entry-layer roots
trees     tree folder identity, sync folders, canonical counts, `ClipPaths`
runs      run layers, RL bundle / chunk ids, clip lists, the stage banner
cli_what  the `--what` answers; cli = the command line
"""
