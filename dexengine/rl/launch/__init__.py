"""RL launcher (paper Sec. 3.4): recipe (configs/rl/recipe.yaml) + request -> trainer processes.

The public names (`Request`, `from_env`, `resolve`, `run`, `load_dump`, `write_dump`, `table_values`) load lazily
from `dexengine.rl.launch.api`, so importing a leaf such as `dexengine.rl.launch.hand_recipe` stays cheap.
"""

from __future__ import annotations

_API = ("Launch", "RecipeError", "Request", "from_env", "load_dump", "resolve", "run", "table_values", "write_dump")


def __getattr__(name):
    if name in _API:
        from dexengine.rl.launch import api

        return getattr(api, name)
    raise AttributeError(name)
