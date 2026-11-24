"""Log markers that loops outside the package read (a site's collect loops): one owner for each line's form,
read by the emitters and by the readers alike, so a reader never keeps a copy of the text.

Standard library only and no package import at module level: a reader imports this file by path with the system
python (`importlib.util.spec_from_file_location`), and shell readers source its assignments:
  python3 <tree>/dexengine/core/log_markers.py --sh    MARK_<NAME>='<head>' and RX_<NAME>='<grep -E pattern>' lines

A marker = a fixed head the emitter's line carries, a formatter that builds the whole line (the emitters call it),
and a reader pattern (`rx`, Python re with named groups; `ere` = the same pattern for grep -E). Readers use
`found(text)` / `parse(line)` (Python) or RX_<NAME> (shell). The patterns use only syntax both engines read alike:
literals, bracket classes, groups, alternation, ?, *, +, {m,}, ^, $.

  EXT_PREPARE_DONE  the aug cells ext 'prepare' stage finished with judge targets (absent = nothing to judge)
  JUDGE_FP          the judge started (its condition fingerprint line)
  JUDGE_LOG         the path of a judge's log in the ext log (a reader reads JUDGE_FP from that file)
  JOBS_OK           a launcher wrote its job table (a lane without it was reaped before it started)
  PROBE · PROBE_RC  per-spec probe verdicts of a cluster lane and the probe's summary rc
  COLLECT_PLACE_OK  a piece tar was placed and verified on the receiving machine (the remote copy may go)
  INFRA             failure words of infrastructure (not the clip's fault): no clip is counted against
"""

from __future__ import annotations

import re
import sys


class Marker:
    """name = the shell name suffix (MARK_<name> · RX_<name>) · head = the fixed text every line of this marker
    carries · rx = the reader pattern (Python re, named groups). A plain class: a dataclass needs the module in
    sys.modules, which a load by path does not give."""

    __slots__ = ("head", "name", "rx")

    def __init__(self, name: str, head: str, rx: str):
        self.name, self.head, self.rx = name, head, rx

    @property
    def ere(self) -> str:
        """The reader pattern for grep -E (named groups become plain groups)."""
        return re.sub(r"\(\?P<\w+>", "(", self.rx)

    def found(self, text: str) -> bool:
        return re.search(self.rx, text, re.M) is not None

    def parse(self, line: str) -> dict | None:
        m = re.search(self.rx, line)
        return None if m is None else m.groupdict()


EXT_PREPARE_DONE = Marker("EXT_PREPARE_DONE", "[EXT_STAGE] prepare done", r"\[EXT_STAGE\] prepare done( |$)")
JUDGE_FP = Marker("JUDGE_FP", "[JUDGE_FP] schema=", r"^\[JUDGE_FP\] schema=(?P<schema>[^ ]+)")
JUDGE_LOG = Marker("JUDGE_LOG", "[JUDGE_LOG]", r"\[JUDGE_LOG\] (?P<log>[^ ]*judge\.log)")
JOBS_OK = Marker("JOBS_OK", "jobs OK", r"jobs OK (?P<n>[0-9]+)( |$)")
PROBE = Marker(
    "PROBE",
    "probe",
    r"\] probe (?P<verdict>[A-Z]+) (?P<spec>[^ ]+) job=(?P<job>[^ ]+)( alive=(?P<alive>[01]))?",
)
PROBE_RC = Marker("PROBE_RC", "PROBE rc=", r"\] PROBE rc=(?P<rc>-?[0-9]+) (?P<counts>\{[^}]*\})")
COLLECT_PLACE_OK = Marker(
    "COLLECT_PLACE_OK",
    "[COLLECT_PLACE] OK",
    r"^\[COLLECT_PLACE\] OK (?P<tar>[^ ]+/pieces/(?P<piece>[^/ ]+)\.tar) sha=(?P<sha>[0-9a-f]{12,})",
)

#: session kinds (the session launcher's classification of a dead session) and their text in its death line
KIND_TEXT = {
    "boot": "boot crash (crash before the hook)",
    "pre": "error before the hook (deterministic, no retry)",
    "fail": "died during the session",
    "marks": "canonical background, lighting or look not applied (canonical defect)",
    "stall": "stall (no progress, session group reaped)",
    "crash": "native crash (signal death after the hook, inside Kit)",
}
#: the kinds worth a new session (infrastructure, not the clip): a new attempt may pass
RETRY_KINDS = ("boot", "stall", "crash")
#: (log text, label) of boot failures the launcher recognises in a session log
BOOT_SIGNS = (
    ("Crash detected", "app crash (breakpad)"),
    ("CUDA being in bad state", "CUDA bad state"),
    ("Segmentation fault", "segfault"),
    ("CUDA error", "CUDA error"),
    ("out of memory", "out of memory"),
)
#: runtime words of infrastructure failures (python, CUDA, the kernel, the cluster)
RUNTIME_INFRA = (
    "No such file",
    "SIGSEGV",
    "OutOfMemory",
    "Killed",
    "timed out",
    "core dumped",
    "Fatal Python",
    "NCCL",
    "Xid",
)


def _alt(words) -> str:
    return "|".join(re.sub(r"([][.*+?^$(){}|\\])", r"\\\1", w) for w in words)


INFRA = Marker(
    "INFRA",
    "",
    "("
    + _alt([KIND_TEXT[k] for k in RETRY_KINDS] + [s for pair in BOOT_SIGNS for s in pair] + list(RUNTIME_INFRA))
    + ")",
)
MARKERS = (EXT_PREPARE_DONE, JUDGE_FP, JUDGE_LOG, JOBS_OK, PROBE, PROBE_RC, COLLECT_PLACE_OK, INFRA)


def ext_prepare_done(clips: int, groups: int, plan) -> str:
    return f"{EXT_PREPARE_DONE.head} · clips {clips} · condition groups {groups} · plan {plan}"


def judge_fp(schema, rest: str) -> str:
    return f"{JUDGE_FP.head}{schema} {rest}"


def judge_log(path) -> str:
    return f"{JUDGE_LOG.head} {path}"


def jobs_ok(tag: str, n: int, rest: str) -> str:
    return f"{tag} {JOBS_OK.head} {n} {rest}"


def probe(tag: str, verdict: str, spec: str, fields: dict, tail: str = "") -> str:
    """`fields` in order — job first (the reader pattern needs it), alive (0|1) when the lane measures it; `tail` =
    free text after them."""
    line = (
        (f"{tag} " if tag else "") + f"{PROBE.head} {verdict} {spec} " + " ".join(f"{k}={v}" for k, v in fields.items())
    )
    return f"{line} {tail}" if tail else line


def probe_rc(tag: str, rc: int, counts) -> str:
    """`counts` = {verdict: n} (printed as the dict) or the caller's own text of it (a JSON object)."""
    return f"{tag} {PROBE_RC.head}{rc} {counts}"


def collect_place_ok(tar, sha: str, rest: str) -> str:
    return f"{COLLECT_PLACE_OK.head} {tar} sha={sha[:12]} · {rest}"


def _sq(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def sh_lines() -> list[str]:
    out = []
    for m in MARKERS:
        out += [f"MARK_{m.name}={_sq(m.head)}"] if m.head else []
        out.append(f"RX_{m.name}={_sq(m.ere)}")
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--sh"]:
        print("\n".join(sh_lines()))
        return 0
    print("usage: log_markers.py --sh", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
