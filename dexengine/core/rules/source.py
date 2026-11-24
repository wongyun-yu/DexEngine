"""Read a value from its owning source file without importing it: Python constants (ast), JSON keys, CLI flag
defaults and shell variable defaults. Rule thresholds are references `path:symbol` resolved here."""

from __future__ import annotations

import os
import pathlib
import re

from dexengine.core import paths as _paths_mod
from dexengine.core.repo import REPO_ROOT

REPO = str(REPO_ROOT)
#: References `dexengine/core/datasets.py:<name>` name this registry. Its code lives in the modules below,
#: read in order.
SELF_PATH = "dexengine/core/datasets.py"
#: A reference into the operations layer: ``site:<path in that folder>:<symbol>`` names operations code (a queue, a
#: retry loop) that a rule card cites. The folder (SITE_LAYER, next to the code tree) is optional: where it is
#: present the reference resolves like any other; a tree without it reports SITE_ABSENT for it instead of
#: "missing".
SITE_PREFIX = "site:"
SITE_LAYER = "ops"
SITE_ABSENT = "(deployment layer, not in this tree)"
_SELF_FILES = tuple(
    f"dexengine/core/rules/{m}.py"
    for m in ("source", "profiles", "identity", "outputs", "ledger", "ruleset", "rule", "thresholds")
)


def _json_key(f: str, name: str):
    """Find one key in a JSON document (nested included) — (found, value)."""
    import json as _json  # noqa: PLC0415

    try:
        with open(f, encoding="utf-8") as fh:
            doc = _json.load(fh)
    except (OSError, ValueError):
        return False, None

    def dig(o):
        if isinstance(o, dict):
            if name in o:
                return True, o[name]
            for v in o.values():
                hit, val = dig(v)
                if hit:
                    return True, val
        elif isinstance(o, list):
            for v in o:
                hit, val = dig(v)
                if hit:
                    return True, val
        return False, None

    return dig(doc)


def _yaml_key(f: str, name: str):
    """A key of a YAML document (nested sections included) — (exists, value)."""
    import yaml as _yaml  # noqa: PLC0415

    try:
        with open(f, encoding="utf-8") as fh:
            doc = _yaml.safe_load(fh)
    except (OSError, _yaml.YAMLError):
        return False, None
    for sec in (doc or {}).values():
        if isinstance(sec, dict) and name in sec:
            return True, sec[name]
    return (True, doc[name]) if isinstance(doc, dict) and name in doc else (False, None)


def _flag_default(f: str, path: str, name: str):
    """The **declared default** of `--flag` — from argparse (.py) or a shell script. An unknown extension raises."""
    import ast  # noqa: PLC0415

    with open(f, encoding="utf-8") as fh:
        txt = fh.read()
    if path.endswith(".py"):
        try:
            tree = ast.parse(txt)
        except SyntaxError:
            return False, None
        for nd in ast.walk(tree):
            if not (isinstance(nd, ast.Call) and getattr(nd.func, "attr", "") == "add_argument"):
                continue
            if not (nd.args and isinstance(nd.args[0], ast.Constant) and nd.args[0].value == name):
                continue
            for kw in nd.keywords:
                if kw.arg == "default":
                    try:
                        return True, ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        return True, None
            return True, None  # declared without a default (required etc.)
        return False, None
    if path.endswith((".sh", ".bash", ".env")):  # shell: the line that reads `--x <value>`
        import re as _re  # noqa: PLC0415

        m = _re.search(_re.escape(name) + r"[= ]+([^\s\\\"']+)", txt)
        return (True, m.group(1)) if m else (False, None)
    raise ValueError(f"dexengine.core.rules: no reader for flag references in this extension — {path}:{name}")


def _num_or_str(v):
    """A declared default (string) → a number when it parses as one. An empty string ("off") or anything else stays
    as is."""
    if isinstance(v, str) and v.strip():
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _env_default_expr(node):
    """`os.environ.get(NAME, LIT)` · `os.getenv(NAME, LIT)`, possibly wrapped in `float(...)`·`int(...)`·`... or X`
    → (NAME, default). None for any other shape. The value is the **declared default** — an env set at run time
    wins over it."""
    import ast  # noqa: PLC0415

    conv = None
    while True:
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and node.values:
            node = node.values[0]
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("float", "int", "str")
            and len(node.args) == 1
        ):
            conv = conv or node.func.id
            node = node.args[0]
            continue
        break
    if not isinstance(node, ast.Call) or len(node.args) != 2:
        return None
    fn = node.func
    is_get = (
        isinstance(fn, ast.Attribute)
        and fn.attr == "get"
        and isinstance(fn.value, ast.Attribute)
        and fn.value.attr == "environ"
    )
    is_getenv = isinstance(fn, ast.Attribute) and fn.attr == "getenv"
    if not (is_get or is_getenv):
        return None
    k, d = node.args
    if not (isinstance(k, ast.Constant) and isinstance(k.value, str) and isinstance(d, ast.Constant)):
        return None
    v = d.value
    if conv == "float":
        v = float(v)
    elif conv == "int":
        v = int(v)
    else:
        v = _num_or_str(v)
    return k.value, v


def _var_default(f: str, path: str, var: str):
    """A `$<name>` reference = the **variable default** the file declares — (found, value).
    .py: an `os.environ.get("<name>", default)` call (inside a function is fine). Shell: `${<name>:-default}` or
    `<name>=${N:-default}`. Two or more defaults in one file that **differ** give None (conflict) — verify reports it
    as 'value not readable'. An unknown extension raises."""
    txt = pathlib.Path(f).read_text(encoding="utf-8")
    vals = []
    if path.endswith(".py"):
        import ast  # noqa: PLC0415

        try:
            tree = ast.parse(txt)
        except SyntaxError:
            return False, None
        for nd in ast.walk(tree):
            if isinstance(nd, ast.Call):
                ed = _env_default_expr(nd)
                if ed is not None and ed[0] == var:
                    vals.append(ed[1])
    elif path.endswith((".sh", ".bash", ".env")):
        pat = re.escape(var)
        vals = [_num_or_str(m) for m in re.findall(r"\$\{" + pat + r":-([^}]*)\}", txt)]
        vals += [_num_or_str(m) for m in re.findall(r"(?m)(?:^|[\s;])" + pat + r"=\$\{[^}:]+:-([^}]*)\}", txt)]
    else:
        raise ValueError(f"dexengine.core.rules: no reader for variable defaults in this extension — {path}:${var}")
    if not vals:
        return False, None
    return True, (vals[0] if all(v == vals[0] for v in vals) else None)


def ref_env(ref: str) -> str:
    """The env name that can **override the referenced value at run time** ("" if none), so a page can say
    "default · overridable by env"."""
    path, _, name = ref.rpartition(":")
    if name.startswith("$"):
        return name[1:]
    if not path.endswith(".py") or name.startswith("--") or path.startswith("@"):
        return ""
    import ast  # noqa: PLC0415

    f = os.path.join(REPO, path)
    try:
        tree = ast.parse(pathlib.Path(f).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            ed = _env_default_expr(node.value)
            return ed[0] if ed else ""
    return ""


def _assign_value(node):
    """(True, value) of an assignment: its literal, the declared default of an env-overridable constant, the
    literal keys of a dict, else None."""
    import ast  # noqa: PLC0415

    try:
        return True, ast.literal_eval(node.value)
    except (ValueError, SyntaxError):
        # `float(os.environ.get("DEXENGINE_X", "20"))` — an env-overridable constant reads its
        # **declared default**. Returning None would hide env-overridable judge thresholds from
        # the rule table.
        _ed = _env_default_expr(node.value)
        if _ed is not None:
            return True, _ed[1]
        # A non-literal value may still have **literal keys** (e.g. search_policy.DZ_CELLS, whose
        # values are the aliases _ONE/_THREE/_WIDE). A scope (scope_from) needs only the keys, so
        # return them — None would make the dz dataset scope an **empty tuple**.
        if isinstance(node.value, ast.Dict):
            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if keys:
                return True, dict.fromkeys(keys)
        return True, None


def _py_symbol(f, name):
    """(exists, value|None) of a top-level symbol of a .py file, read from its source."""
    import ast  # noqa: PLC0415

    try:
        with open(f, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except SyntaxError:
        return False, None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True, None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return _assign_value(node)
                if isinstance(t, ast.Tuple):  # `A, B, C = 3.0, 10.0, 10.0` is an ownership declaration too
                    for k, el in enumerate(t.elts):
                        if isinstance(el, ast.Name) and el.id == name:
                            try:
                                return True, ast.literal_eval(node.value).__getitem__(k)
                            except Exception:  # noqa: BLE001
                                return True, None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            try:
                return True, ast.literal_eval(node.value) if node.value is not None else None
            except (ValueError, SyntaxError):
                return True, None
    return False, None


def _source_file(path: str) -> str:
    """The file a reference path names: `@<run layer>/<name>` = a place built by the run-layer table (per-hand
    directories — no literal output paths; only hand-agnostic layers such as exp_shared: a hand layer makes run_path
    demand a hand and die, a reference does not pick a hand) · `site:<path>` = an operations-layer file · else a path
    of the code tree."""
    if path.startswith("@"):
        layer, _sep, rest = path[1:].partition("/")
        return str(_paths().run_path(layer) / rest)
    if path.startswith(SITE_PREFIX):
        return os.path.join(REPO, SITE_LAYER, path[len(SITE_PREFIX) :])
    return os.path.join(REPO, path)


def _missing(path: str):
    """(exists, value) of a reference whose file is absent: a `site:` reference in a tree without the operations
    layer is SITE_ABSENT, any other one is missing."""
    if path.startswith(SITE_PREFIX) and not os.path.isdir(os.path.join(REPO, SITE_LAYER)):
        return True, SITE_ABSENT
    return False, None


def _source_symbol(path: str, name: str):
    """Read a top-level symbol of the owning file **from its source** — (exists, value|None).

    The module is not executed: the owning modules drag in heavy dependencies such as mujoco·trimesh (absent from the
    system python), and executing them would tie the registry to those dependencies. A constant is read down to its
    value with ast.literal_eval; a function or class is only checked for existence.
    The value is **not copied** here — it is read from the owning file every time."""
    if path == SELF_PATH:
        for sub in _SELF_FILES:
            hit = _source_symbol(sub, name)
            if hit[0]:
                return hit
        return False, None

    f = _source_file(path)
    if not os.path.isfile(f):
        return _missing(path)

    # A reference has three **kinds** — Python top-level symbol · JSON key · CLI flag default. All three are read:
    # returning `(False, None)` for any of them would be the **wrong diagnosis** "no such symbol".
    # An unknown kind is not reported as missing; it **raises**.
    if path.endswith(".json"):
        return _json_key(f, name)
    if path.endswith((".yaml", ".yml")):
        return _yaml_key(f, name)
    if name.startswith("--"):  # CLI flag = the default that file declares
        return _flag_default(f, path, name)
    if name.startswith("$"):  # the **declared default** of an env / shell variable
        return _var_default(f, path, name[1:])
    if not path.endswith(".py"):
        raise ValueError(
            f"dexengine.core.rules: no reader for symbol references in this extension — {path}:{name} "
            "(reads only .py top-level symbols · .json keys · '--flag')"
        )
    return _py_symbol(f, name)


def _paths():
    """dexengine.core.paths loaded once **from this file's tree** (a module-global cache, not a name cache: a fake
    repo's copy of this registry has its own globals, so it never picks up the real paths module)."""
    return _paths_mod


def _sym(ref: str):
    path, name = ref.rsplit(":", 1)
    ok, val = _source_symbol(path, name)
    return val, ok
