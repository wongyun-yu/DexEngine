"""Compare data the way every release gate does (from-raw, empty-root assets): one digest per comparison unit.

Units (fixed here): .usd/.usda/.usdc one for the layer's text export (pxr) · .npz one per array (`<name>`:
dtype, shape, bytes) · .npy and .pkl one for the loaded content
(pickle bytes are not stable run to run for the same content) · .json one for the parsed document — or, when the gate
names its sample clips and the document has a top-level `clips` object, one per sample clip (`clips.<clip>`): the
rest of such a document (summaries) is fitted over the whole dataset and a sample cannot reproduce it · other files
one for the bytes. Two files are equal when their unit sets and every unit digest are equal; the public reference is
a list of those digests (path · unit · sha256), so no derived file has to be shipped to check against it.

Pass rule: every unit is digested twice — its full
value and its verdict view — and a file fails only when a verdict view differs (or a file or unit is on one side
only). The verdict view drops the stamps by dexengine.data.tree_content's definition (a version key, an fp key, a
date-time or absolute-path string, a task_info's top-level provenance) and every floating value (float arrays keep
their dtype and shape; float scalars count as one value), so what must agree is everything else: the verdicts
(adopted · rejected · reason), labels, ids, counts, flags, structure and text. A unit whose full value differs but
whose verdict view agrees is reported as numeric (`[COMPARE] numeric …`, with the max |Δ| per array when both roots
are at hand), never failed: there is no threshold. The rows of configs/release/compare_rules.tsv (rule · files ·
match · why) are dropped from both views. A run prints the machine it ran on (`[COMPARE_ENV]`: CPU, its vector
extensions, glibc, the BLAS core), because some steps round differently on another CPU. A reference digest is
`<full>:<verdict>`; an older reference with one digest is taken as both.

    python -m dexengine.release.compare --a <root> (--b <root> | --ref <digest list>) --paths <list> [--clips <list>]
    python -m dexengine.release.compare --a <root> --paths <list> [--clips <list>] --write-ref <digest list>

With --ref the files compared are those of --paths and those the reference lists, so a file made on one side only fails
either way.

End marker `[COMPARE] files=… equal=… differ=… only_a=… only_b=…`; exit 1 when anything differs.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import pickle
import platform
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexengine.core.repo import CONFIGS_DIR
from dexengine.data import tree_content as TC  # the stamp definition (owner)

RULES = CONFIGS_DIR / "release" / "compare_rules.tsv"
HEAD = ("rule", "files", "match", "why")
KINDS = ("json_key", "line", "file", "usd_meta")
USD_SUFFIXES = (".usd", ".usda", ".usdc")
USD_META = ("documentation",)  # root-layer metadata a usd_meta row may clear
TAG = "[COMPARE]"
ENV_TAG = "[COMPARE_ENV]"
WHOLE = "-"  # unit name of a file compared as one unit
MISSING = "missing"  # digest of a sample clip a clip-keyed document does not hold
VSEP = ":"  # a unit digest = <full>:<verdict view>


@dataclass(frozen=True)
class Rule:
    rule: str
    files: str
    match: str
    why: str

    def applies(self, rel: str) -> bool:
        return fnmatch.fnmatchcase(rel, self.files)


def load_rules(path: str | os.PathLike = RULES) -> list[Rule]:
    """The rows of the rules file; a row of another form dies (no silent rule)."""
    out, head = [], None
    for n, ln in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("\t")
        if head is None:
            if tuple(f) != HEAD:
                raise SystemExit(f"{TAG} ✗ {path}:{n}: header must be {' · '.join(HEAD)}")
            head = f
            continue
        if len(f) != len(HEAD) or not all(x.strip() for x in f):
            raise SystemExit(f"{TAG} ✗ {path}:{n}: needs {len(HEAD)} non-empty columns (the reason is required)")
        r = Rule(*f)
        if r.rule not in KINDS:
            raise SystemExit(f"{TAG} ✗ {path}:{n}: rule {r.rule!r} is not one of {', '.join(KINDS)}")
        if r.rule == "file" and r.match != "-":
            raise SystemExit(f"{TAG} ✗ {path}:{n}: a file rule takes match '-'")
        if r.rule == "line":
            re.compile(r.match)
        if r.rule == "usd_meta" and r.match not in USD_META:
            raise SystemExit(f"{TAG} ✗ {path}:{n}: usd_meta clears one of {', '.join(USD_META)}, not {r.match!r}")
        out.append(r)
    if head is None:
        raise SystemExit(f"{TAG} ✗ {path}: no header row")
    return out


def _as_array(h, v):
    """numpy scalars and torch tensors (pickled hand fits hold them) as arrays; a tensor keeps its own tag."""
    if isinstance(v, np.generic):
        return np.asarray(v)
    if type(v).__module__.partition(".")[0] == "torch" and hasattr(v, "detach"):
        h.update(b"P")
        return v.detach().cpu().numpy()
    return v


def _scalar_bytes(v) -> bytes | None:
    """Tagged bytes of a scalar value; None when `v` is not one."""
    if v is None:
        return b"N"
    if isinstance(v, bool):
        return b"B1" if v else b"B0"
    if isinstance(v, int):
        return b"I" + str(v).encode() + b";"
    if isinstance(v, float):
        return b"Fnan" if v != v else b"F" + struct.pack("<d", v)
    if isinstance(v, (str, bytes)):
        b = v.encode("utf-8") if isinstance(v, str) else v
        return (b"S" if isinstance(v, str) else b"Y") + str(len(b)).encode() + b":" + b
    return None


def _feed(h, v, verdict: bool = False) -> None:
    """Feed one loaded value into `h` in a canonical, type-tagged form (dict keys sorted; NaN as one value). The
    verdict view (`verdict`) feeds a floating array's dtype and shape only and a float scalar as one tag."""
    v = _as_array(h, v)
    if isinstance(v, np.ndarray):
        h.update(b"A" + v.dtype.str.encode() + repr(v.shape).encode())
        if v.dtype != object:
            if not (verdict and v.dtype.kind in "fc"):
                h.update(np.ascontiguousarray(v).tobytes())
            return
        v = v.ravel().tolist()
        h.update(b"O")
    if verdict and isinstance(v, float):
        h.update(b"F")
        return
    sb = _scalar_bytes(v)
    if sb is not None:
        h.update(sb)
    elif isinstance(v, (list, tuple)):
        h.update((b"L" if isinstance(v, list) else b"T") + str(len(v)).encode() + b":")
        for x in v:
            _feed(h, x, verdict)
    elif isinstance(v, dict):
        h.update(b"D" + str(len(v)).encode() + b":")
        for k in sorted(v, key=repr):
            _feed(h, k, verdict)
            _feed(h, v[k], verdict)
    else:
        raise TypeError(f"{TAG} no canonical form for a {type(v).__name__} value")


def digest_value(v, verdict: bool = False) -> str:
    h = hashlib.sha256()
    _feed(h, v, verdict)
    return h.hexdigest()


def unstamp(v, top: bool = True):
    """`v` without its stamps, by the owner's definition (dexengine.data.tree_content): version and fp keys,
    date-time and absolute-path strings, and a document's top-level provenance. Records stay (they hold verdicts)."""
    if isinstance(v, dict):
        return {
            k: unstamp(x, False)
            for k, x in v.items()
            if not ((top and k == TC.SOURCE_TOP) or TC.is_version_key(k) or TC.is_fp_key(k) or TC.is_stamp_value(x))
        }
    if isinstance(v, list):
        return [unstamp(x, False) for x in v if not TC.is_stamp_value(x)]
    return v


def unit_digest(v) -> str:
    """`<full>:<verdict view>` of one loaded value."""
    return digest_value(v) + VSEP + digest_value(unstamp(v), verdict=True)


def _drop(doc, segs: list[str]) -> None:
    """Remove the value at the dotted path `segs` (a `*` segment = every key or index)."""
    if not segs:
        return
    seg, rest = segs[0], segs[1:]
    if isinstance(doc, dict):
        keys = list(doc) if seg == "*" else ([seg] if seg in doc else [])
    elif isinstance(doc, list):
        keys = list(range(len(doc))) if seg == "*" else ([int(seg)] if seg.isdigit() and int(seg) < len(doc) else [])
    else:
        return
    for k in reversed(keys):
        if rest:
            _drop(doc[k], rest)
        else:
            del doc[k]


def _npz(path: Path) -> dict[str, str]:
    with np.load(path, allow_pickle=True) as z:
        return {k: unit_digest(z[k]) for k in z.files}


def _pkl(path: Path) -> dict[str, str]:
    with open(path, "rb") as fh:
        return {WHOLE: unit_digest(pickle.load(fh))}


def _json(path: Path, mine: list[Rule], clips: frozenset[str] | None) -> dict[str, str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    for r in mine:
        if r.rule == "json_key":
            _drop(doc, r.match.split("."))
    if clips is not None and isinstance(doc, dict) and isinstance(doc.get("clips"), dict):
        got = doc["clips"]
        return {f"clips.{c}": unit_digest(got[c]) if c in got else MISSING for c in sorted(clips)}
    return {WHOLE: unit_digest(doc)}


def _bytes(path: Path, mine: list[Rule]) -> dict[str, str]:
    data = path.read_bytes()
    drops = [re.compile(r.match.encode()) for r in mine if r.rule == "line"]
    if drops:
        data = b"".join(x for x in data.splitlines(keepends=True) if not any(d.search(x) for d in drops))
    return {WHOLE: hashlib.sha256(data).hexdigest()}


def _usd(path: Path, mine: list[Rule]) -> dict[str, str]:
    """A USD layer by content: opened anonymously (the file is never written), the usd_meta rows' root-layer metadata
    cleared, the text export digested. Needs pxr (the Isaac interpreter); without it a USD file dies."""
    try:
        from pxr import Sdf  # noqa: PLC0415 — only USD files need it
    except ImportError:
        raise SystemExit(
            f"{TAG} ✗ {path}: a USD file needs pxr — run the comparison with the Isaac interpreter"
        ) from None
    layer = Sdf.Layer.OpenAsAnonymous(str(path))
    if layer is None:
        raise SystemExit(f"{TAG} ✗ {path}: not a USD layer")
    for r in mine:
        if r.rule == "usd_meta":
            setattr(layer, r.match, "")  # the root-layer field the row names (load_rules allows USD_META only)
    return {WHOLE: hashlib.sha256(layer.ExportToString().encode()).hexdigest()}


def units(rel: str, path: Path, rules: list[Rule], clips: frozenset[str] | None = None) -> dict[str, str]:
    """The comparison units of one file → their digests ({} when a `file` rule skips the file)."""
    mine = [r for r in rules if r.applies(rel)]
    if any(r.rule == "file" for r in mine):
        return {}
    by_suffix = {
        ".npz": lambda: _npz(path),
        ".npy": lambda: {WHOLE: unit_digest(np.load(path, allow_pickle=True))},
        ".pkl": lambda: _pkl(path),
        ".json": lambda: _json(path, mine, clips),
        **dict.fromkeys(USD_SUFFIXES, lambda: _usd(path, mine)),
    }
    return by_suffix.get(path.suffix, lambda: _bytes(path, mine))()


def _verdict(d: str | None) -> str | None:
    return None if d is None else d.rpartition(VSEP)[2]


def diff_units(ua: dict[str, str], ub: dict[str, str]) -> tuple[str, list[str]]:
    """('' or the first units whose verdict view differs, the units whose full value only differs)."""
    keys = sorted(ua.keys() | ub.keys())
    bad = [k for k in keys if _verdict(ua.get(k)) != _verdict(ub.get(k))]
    num = [k for k in keys if k not in bad and ua.get(k) != ub.get(k)]
    return (f"units differ: {bad[:5]}" if bad else ""), num


def max_abs(a: Path, b: Path, unit_names: list[str]) -> dict[str, float]:
    """Max |Δ| of the floating arrays `unit_names` of two npz files (report only)."""
    out = {}
    with np.load(a, allow_pickle=True) as za, np.load(b, allow_pickle=True) as zb:
        for k in unit_names:
            x, y = za[k], zb[k]
            if x.dtype.kind in "fc" and x.shape == y.shape:
                out[k] = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
    return out


def root_units(root: Path, rels: list[str], rules: list[Rule], clips: frozenset[str] | None = None) -> dict:
    """{rel: {unit: digest}} of the files present under `root` (absent files are left out)."""
    return {rel: units(rel, root / rel, rules, clips) for rel in sorted(set(rels)) if (root / rel).is_file()}


def compare_units(ua: dict, ub: dict, rels: list[str]) -> dict:
    """Per relative path: equal, differ (with the finding) or present on one side only."""
    res = {"equal": [], "differ": {}, "numeric": {}, "only_a": [], "only_b": []}
    for rel in sorted(set(rels)):
        ha, hb = rel in ua, rel in ub
        if not (ha and hb):
            if ha or hb:
                res["only_a" if ha else "only_b"].append(rel)
            else:
                res["differ"][rel] = "missing on both sides"
            continue
        why, num = diff_units(ua[rel], ub[rel])
        if why:
            res["differ"][rel] = why
        else:
            res["equal"].append(rel)
            if num:
                res["numeric"][rel] = dict.fromkeys(num)
    return res


def compare_roots(a: Path, b: Path, rels: list[str], rules: list[Rule] | None = None, clips=None) -> dict:
    rules = load_rules() if rules is None else rules
    res = compare_units(root_units(a, rels, rules, clips), root_units(b, rels, rules, clips), rels)
    for rel, num in res["numeric"].items():
        if rel.endswith(".npz"):
            num.update(max_abs(a / rel, b / rel, list(num)))
    return res


def write_ref(out: Path, u: dict) -> int:
    """The public reference: one `path<TAB>unit<TAB>sha256` line per unit."""
    lines = [f"{rel}\t{k}\t{d}" for rel in sorted(u) for k, d in sorted(u[rel].items())]
    out.write_text("".join(x + "\n" for x in lines))
    return len(lines)


def read_ref(path: Path) -> dict:
    u: dict[str, dict[str, str]] = {}
    for n, ln in enumerate(path.read_text().splitlines(), 1):
        if not ln.strip():
            continue
        f = ln.split("\t")
        if len(f) != 3:
            raise SystemExit(f"{TAG} ✗ {path}:{n}: a reference line is path<TAB>unit<TAB>sha256")
        u.setdefault(f[0], {})[f[1]] = f[2]
    return u


def run_environment() -> dict:
    """The machine a run used: CPU model, its AVX-512 / AVX2 / FMA flags, glibc, the BLAS core in use."""
    cpu, flags = platform.processor(), set()
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name") and not cpu.strip():
                cpu = ln.split(":", 1)[1].strip()
            if ln.startswith("flags"):
                flags = set(ln.split(":", 1)[1].split())
                if cpu.strip():
                    break
    except OSError:
        pass
    try:
        import threadpoolctl  # noqa: PLC0415 — optional reader of the loaded BLAS

        blas = [
            f"{i.get('internal_api')}:{i.get('architecture', '?')}:{i.get('version', '?')}"
            for i in threadpoolctl.threadpool_info()
            if i.get("user_api") == "blas"
        ]
    except ImportError:
        blas = ["threadpoolctl missing"]
    return {
        "cpu": cpu,
        "vector": sorted(f for f in flags if f in ("avx2", "fma", "avx512f", "avx512dq", "avx512vl", "avx512bw")),
        "glibc": os.confstr("CS_GNU_LIBC_VERSION") if hasattr(os, "confstr") else "?",
        "blas": blas,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def report(res: dict) -> str:
    n = len(res["equal"]) + len(res["differ"]) + len(res["only_a"]) + len(res["only_b"])
    return (
        f"{TAG} files={n} equal={len(res['equal'])} (numeric only {len(res.get('numeric', {}))}) "
        f"differ={len(res['differ'])} only_a={len(res['only_a'])} only_b={len(res['only_b'])}"
    )


def _list(p: Path | None) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip()] if p else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dexengine.release.compare", description=__doc__.split("\n")[0])
    ap.add_argument("--a", type=Path, required=True, help="the rebuilt root")
    ap.add_argument("--b", type=Path, default=None, help="the reference root")
    ap.add_argument("--ref", type=Path, default=None, help="the reference digest list")
    ap.add_argument("--write-ref", type=Path, default=None, help="write the digest list of --a here (no comparison)")
    ap.add_argument("--paths", type=Path, required=True, help="file of data-root relative paths, one per line")
    ap.add_argument("--clips", type=Path, default=None, help="sample clips: clip-keyed documents compare these only")
    ap.add_argument("--json", type=Path, default=None, help="write the per-file result here")
    a = ap.parse_args(argv)
    if sum(x is not None for x in (a.b, a.ref, a.write_ref)) != 1:
        ap.error("give exactly one of --b, --ref, --write-ref")
    print(f"{ENV_TAG} {json.dumps(run_environment(), sort_keys=True)}", flush=True)
    rels = _list(a.paths)
    if not rels:
        raise SystemExit(f"{TAG} ✗ {a.paths} lists no path")
    clips = frozenset(_list(a.clips)) if a.clips else None
    rules = load_rules()
    if a.b is not None:
        res = compare_roots(a.a, a.b, rels, rules, clips)
    else:
        ua = root_units(a.a, rels, rules, clips)
        if a.write_ref:
            print(f"{TAG} reference {a.write_ref}: {write_ref(a.write_ref, ua)} units of {len(ua)} files", flush=True)
            return 0 if len(ua) == len(set(rels)) else 1
        ref = read_ref(a.ref)
        # Both directions: a file the reference lists but this run's list lacks is checked too (absent → only_b).
        both = sorted(set(rels) | set(ref))
        res = compare_units(root_units(a.a, both, rules, clips), ref, both)
    for rel, why in sorted(res["differ"].items())[:20]:
        print(f"{TAG} differ {rel}: {why}", flush=True)
    for rel, num in sorted(res["numeric"].items()):
        shown = ", ".join(k if v is None else f"{k} max|Δ| {v:.3g}" for k, v in sorted(num.items()))
        print(f"{TAG} numeric {rel}: {shown}", flush=True)
    for side in ("only_a", "only_b"):
        for rel in res[side][:20]:
            print(f"{TAG} {side} {rel}", flush=True)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(report(res), flush=True)
    return 1 if res["differ"] or res["only_a"] or res["only_b"] else 0


if __name__ == "__main__":
    sys.exit(main())
