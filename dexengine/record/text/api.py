"""Messages API layer of the text stage: transport (REST: synchronous run, batch submit/status/fetch), prices,
cost estimate, jsonl I/O. Standard library only (no SDK). Key: when KEY_FILE exists (mode 600, written by hand)
its value goes into env ANTHROPIC_API_KEY (load_key dies on another mode or an empty file); otherwise the env value
is used. The key is never printed or passed as an argument. Owner of the [COLLECT_TEXT] marker.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MARK = "[COLLECT_TEXT]"
#: API key file (written by hand, mode 600); env ANTHROPIC_API_KEY is the other source. A missing key stops the
#: text pipeline with both sources named.
KEY_FILE = Path.home() / ".config" / "dexengine" / "anthropic_api_key"
API = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
MAX_BATCH_REQ, MAX_BATCH_BYTES = 100_000, 250 * 1024 * 1024  # API limits 100,000 requests / 256 MB (6 MB margin)
#: Prices (USD per million tokens), from platform.claude.com/docs/en/about-claude/pricing. Batch = 0.5 x base.
#: Cache read = base input x cache_hit. The newer tokenizer yields about 30 % more tokens for the same text.
PRICES = {
    "claude-opus-5-5": {"inp": 4.0, "out": 20.0, "cache_hit": 0.05},
    "claude-sonnet-5-5": {"inp": 2.0, "out": 10.0, "cache_hit": 0.05},
    "claude-haiku-5-5": {"inp": 0.10, "out": 0.50, "cache_hit": 0.10},
    "claude-fable-5-1": {"inp": 10.0, "out": 50.0, "cache_hit": 0.025},
}
BATCH_FACTOR = 0.5
CHARS_PER_TOKEN = 3.0  # conservative for mixed English / JSON (newer tokenizer); real values come from usage
OUT_TOKENS_EST = 780  # assumed output tokens per clip (instruction + 4 paraphrases + short + detailed + fields
#                                + per phase 1 instruction + 2 paraphrases, 5 phases), calibrated on measured usage


def die(msg: str) -> None:
    """Print a FAIL line and exit with code 2."""
    print(f"{MARK} FAIL {msg}", flush=True)
    raise SystemExit(2)


#: Key or credit errors (response body): retrying does not help — the track halts (STOP) until a key is provided
#: (KEY_FILE or env ANTHROPIC_API_KEY). Other 4xx die.
KEY_NEEDED_RE = re.compile(
    r"credit balance|billing|purchase credits|authentication_error|invalid x-api-key|"
    r"permission_error|api key",
    re.I,
)


class KeyNeeded(SystemExit):
    """Key or credit problem (KEY_NEEDED_RE); the request is not repeated (pipeline.tick halts)."""


def read_jsonl(p: Path) -> list[dict]:
    """Rows of a jsonl file ([] when missing)."""
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(p: Path, rows) -> None:
    """Write rows atomically (temp file + replace)."""
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def image_tokens(w: int, h: int) -> int:
    """Image tokens = ceil(w/28) x ceil(h/28), capped at 4,784 (vision docs; no downscaling within 2,576 px)."""
    return min(4784, math.ceil(w / 28) * math.ceil(h / 28))


# ───────────────────────── cost estimate ─────────────────────────
def estimate(reqs: list[dict]) -> dict:
    """Token estimate from character counts of the requests: system (cached), per-clip input, assumed output;
    batch and synchronous USD per model."""
    if not reqs:
        return {}
    p0 = reqs[0]["params"]
    sys_tok = (len(p0["system"][0]["text"]) + len(json.dumps(p0["output_config"]))) / CHARS_PER_TOKEN
    user_tok, img_tok, n_img = 0.0, 0, 0
    for r in reqs:
        c = r["params"]["messages"][0]["content"]
        if isinstance(c, str):
            user_tok += len(c) / CHARS_PER_TOKEN
            continue
        for b in c:
            if b["type"] == "text":
                user_tok += len(b["text"]) / CHARS_PER_TOKEN
        for f in r.get("frames", []):
            img_tok += image_tokens(f["w"], f["h"])
            n_img += 1
    user_tok += img_tok
    n = len(reqs)
    out = {}
    for m, pr in PRICES.items():
        cached = (sys_tok * pr["inp"] * 1.25 + (n - 1) * sys_tok * pr["inp"] * pr["cache_hit"]) / 1e6  # write + reads
        uncached = n * sys_tok * pr["inp"] / 1e6
        body = (user_tok * pr["inp"] + n * OUT_TOKENS_EST * pr["out"]) / 1e6
        out[m] = {
            "sync_usd": round(body + cached, 2),
            "batch_usd": round((body + cached) * BATCH_FACTOR, 2),
            "batch_usd_no_cache": round((body + uncached) * BATCH_FACTOR, 2),
        }
    return {
        "n": n,
        "sys_tokens": round(sys_tok),
        "user_tokens_mean": round(user_tok / n),
        "images": n_img,
        "image_tokens_mean": round(img_tok / n),
        "out_tokens_assumed": OUT_TOKENS_EST,
        "models": out,
    }


def row_usd(row: dict, model: str) -> float:
    """Actual cost of one result row: usage x PRICES (cache write 1.25 x, cache read cache_hit x), times
    BATCH_FACTOR for batch rows (via sync / probe rows are synchronous prices). Failed or expired rows cost 0."""
    res = row.get("result") or {}
    if res.get("type") != "succeeded":
        return 0.0
    u, pr = res["message"].get("usage") or {}, PRICES[model]
    usd = (
        int(u.get("input_tokens") or 0) * pr["inp"]
        + int(u.get("cache_creation_input_tokens") or 0) * pr["inp"] * 1.25
        + int(u.get("cache_read_input_tokens") or 0) * pr["inp"] * pr["cache_hit"]
        + int(u.get("output_tokens") or 0) * pr["out"]
    ) / 1e6
    return usd * (1.0 if row.get("via") in ("sync", "probe") else BATCH_FACTOR)


def round_usd(d: Path) -> tuple[float, float, int]:
    """(batch cost, synchronous fixed cost = probe and sync rows, succeeded batch clips) of one round; the cost gate
    uses the batch cost per clip."""
    model = json.loads((Path(d) / "manifest.json").read_text())["model"]
    raw = read_jsonl(Path(d) / "raw.jsonl")
    sync = [r for r in raw if r.get("via") in ("sync", "probe")]
    batch = [r for r in raw if r.get("via") not in ("sync", "probe")]
    return (
        sum(row_usd(r, model) for r in batch),
        sum(row_usd(r, model) for r in sync),
        sum(1 for r in batch if (r.get("result") or {}).get("type") == "succeeded"),
    )


def error_gate(name: str, bs: list[dict], cap: int) -> str | None:
    """Batch errors >= cap: cancel the unfinished batches and return the reason (else None)."""
    if sum((b.get("request_counts") or {}).get("errored", 0) for b in bs) < cap:
        return None
    for b in bs:
        if b.get("processing_status") != "ended":
            http("POST", f"{API}/messages/batches/{b['id']}/cancel", {})
    return f"{name} batch errors >= {cap} — remaining batches cancelled"


def folder_usd(d: Path) -> float:
    """Money actually spent by one request folder (model from manifest.json, usage from raw.jsonl)."""
    if not (Path(d) / "manifest.json").is_file():
        return 0.0
    model = json.loads((Path(d) / "manifest.json").read_text())["model"]
    return sum(row_usd(r, model) for r in read_jsonl(Path(d) / "raw.jsonl"))


# ───────────────────────── HTTP ─────────────────────────
_KEY_SRC: list[str] = []  # last printed key source (printed again only on a change; read on every call)


def load_key() -> str:
    """Key source 'file' | 'env' | ''. When KEY_FILE exists it must be a regular file with mode 600 and one
    non-empty line; its value goes into the env (the file wins over the env; the value is never printed). Prints
    key=file(mode 600) | key=env | key=none when the source changes."""
    src = ""
    if KEY_FILE.exists():
        mode = KEY_FILE.stat().st_mode & 0o777
        if not KEY_FILE.is_file() or mode != 0o600:
            die(f"key file {KEY_FILE} mode {oct(mode)}: must be a regular file with mode 600 (chmod 600)")
        v = KEY_FILE.read_text(encoding="utf-8").strip()
        if not v or "\n" in v:
            die(f"key file {KEY_FILE} is empty or has several lines (one line only)")
        os.environ["ANTHROPIC_API_KEY"] = v
        src = "file"
    elif os.environ.get("ANTHROPIC_API_KEY", "").strip():
        src = "env"
    if _KEY_SRC[-1:] != [src]:
        print(f"{MARK} key={ {'file': 'file(mode 600)', 'env': 'env'}.get(src, 'none') }", flush=True)
        _KEY_SRC[:] = [src]
    return src


def api_key() -> str:
    """The API key (dies without one)."""
    load_key()
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not k:
        die("no env ANTHROPIC_API_KEY: provide a key and retry (its value is never printed)")
    return k


def http(method: str, url: str, body: dict | None = None, raw: bool = False, tries: int = 6):
    """REST call: 429, 5xx and connection failures retry with exponential back-off (up to tries); 4xx die."""
    data = None if body is None else json.dumps(body).encode()
    hdr = {"x-api-key": api_key(), "anthropic-version": API_VERSION, "content-type": "application/json"}
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=hdr)
            with urllib.request.urlopen(req, timeout=300) as r:
                b = r.read()
                return b if raw else json.loads(b)
        except urllib.error.HTTPError as e:  # noqa: PERF203 — retry loop around a network call
            msg = e.read()[:400].decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504, 529) and k + 1 < tries:
                time.sleep(min(60, 2**k * 2))
                continue
            if e.code in (400, 401, 402, 403) and KEY_NEEDED_RE.search(msg):
                print(f"{MARK} FAIL key needed HTTP {e.code} — {msg[:200]}", flush=True)
                raise KeyNeeded(f"HTTP {e.code} {KEY_NEEDED_RE.search(msg).group(0)}") from e
            die(f"HTTP {e.code} {method} {url.split('?')[0]} — {msg}")
        except (urllib.error.URLError, TimeoutError) as e:
            if k + 1 < tries:
                time.sleep(min(60, 2**k * 2))
                continue
            die(f"connection failed {url} — {e}")
    die("unreachable")
    return None


# ───────────────────────── commands ─────────────────────────
def cmd_run(a) -> int:
    """Synchronous run (when the batch queue is slow): `workers` threads call the API and append rows to raw.jsonl
    under one lock (one process per folder). At usd_cap no new request starts (requests in flight are still
    recorded; cost = row_usd)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    d = Path(a.dir)
    reqs = read_jsonl(d / "requests.jsonl")
    have = {r["custom_id"] for r in read_jsonl(d / "raw.jsonl")}
    todo = [r for r in reqs if r["custom_id"] not in have][: a.limit]
    workers, cap = max(1, int(getattr(a, "workers", 1) or 1)), getattr(a, "usd_cap", None)
    lock, stop, st = threading.Lock(), threading.Event(), {"n": 0, "usd": 0.0}
    print(f"{MARK} run start {len(todo)} · workers {workers} · cap {cap if cap else 'none'}", flush=True)
    with (d / "raw.jsonl").open("a", encoding="utf-8") as f:

        def one(r: dict) -> None:
            if stop.is_set():
                return
            msg = http("POST", f"{API}/messages", r["params"])
            row = {"custom_id": r["custom_id"], "via": "sync", "result": {"type": "succeeded", "message": msg}}
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                st["n"] += 1
                st["usd"] += row_usd(row, r["params"]["model"])
                if cap and st["usd"] >= cap:
                    stop.set()
                if workers == 1 or st["n"] % 50 == 0 or st["n"] == len(todo):
                    print(f"{MARK} run {st['n']}/{len(todo)} · ${st['usd']:.2f} {r['custom_id']}", flush=True)

        from dexengine.core import capacity  # noqa: PLC0415

        step = "dexengine.record.text.api"  # capacity.NEEDS key: API calls in flight (`workers` = explicit option)
        with capacity.sized(step, len(todo), workers) as pl, ThreadPoolExecutor(pl.workers) as ex:
            futs = [ex.submit(one, r) for r in todo]
            try:
                for fu in futs:
                    fu.result()
            except BaseException:
                stop.set()
                for fu in futs:
                    fu.cancel()
                raise
    tail = f" · stopped at cap ${cap}" if stop.is_set() else ""
    print(
        f"{MARK} run done new {st['n']} · ${st['usd']:.2f} · received {len(have) + st['n']}/{len(reqs)}{tail}",
        flush=True,
    )
    return 0


def chunks(reqs: list[dict]):
    """Split requests into batches within MAX_BATCH_REQ and MAX_BATCH_BYTES."""
    cur, size = [], 0
    for r in reqs:
        b = len(json.dumps(r).encode())
        if cur and (len(cur) >= MAX_BATCH_REQ or size + b > MAX_BATCH_BYTES):
            yield cur
            cur, size = [], 0
        cur.append(r)
        size += b
    if cur:
        yield cur


def probe(d: Path, q: dict) -> None:
    """One synchronous probe request before a batch: does the model accept the request form? A 4xx dies in http; a
    missing output field dies here (no batch is submitted; marker [TEXT_PROBE]). A passing result stays in raw.jsonl
    (that clip leaves the batch)."""
    from dexengine.record.text import check as CK  # noqa: PLC0415 — lazy: check imports this module

    msg = http("POST", f"{API}/messages", q["params"])
    out = CK.output(msg)
    need = q["params"]["output_config"]["format"]["schema"]["required"]  # the output contract of the request
    miss = None if out is None else [k for k in need if k not in out]
    if out is None or miss:  # a failed probe is not written to raw (the next submit probes that clip again)
        die(
            f"[TEXT_PROBE] probe failed {q['custom_id']} — {'no output' if out is None else f'missing fields {miss}'} "
            f"· stop_reason {msg.get('stop_reason')} · no batch submitted"
        )
    with (d / "raw.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"custom_id": q["custom_id"], "via": "probe", "result": {"type": "succeeded", "message": msg}},
                ensure_ascii=False,
            )
            + "\n"
        )
    print(
        f"{MARK} [TEXT_PROBE] probe passed {q['custom_id']} · stop_reason {msg.get('stop_reason')} · "
        f"output tokens {(msg.get('usage') or {}).get('output_tokens')}",
        flush=True,
    )


#: A round smaller than this skips the probe when a sibling round with the same dataset, model and prompt passed one.
SMALL_ROUND = 20


def probe_done(d: Path) -> str | None:
    """Name of a sibling round (same parent, not VOID) with the same model, prompt sha and single dataset as this
    round that holds a passed probe (raw via=probe); None otherwise (also for multi-dataset rounds)."""
    m = json.loads((d / "manifest.json").read_text())
    ds = set(m.get("by_dataset") or {})
    if len(ds) != 1:
        return None
    for s in sorted(p for p in d.parent.iterdir() if p != d and p.is_dir()):
        if (s / "VOID").exists() or not (s / "manifest.json").is_file() or not (s / "raw.jsonl").is_file():
            continue
        sm = json.loads((s / "manifest.json").read_text())
        key = (sm.get("model"), sm.get("prompt_sha"), set(sm.get("by_dataset") or {}))
        if key != (m["model"], m.get("prompt_sha"), ds):
            continue
        if any(r.get("via") == "probe" for r in read_jsonl(s / "raw.jsonl")):
            return s.name
    return None


def cmd_submit(a) -> int:
    """Submit the not-yet-received requests as batches (after one probe)."""
    d = Path(a.dir)
    reqs = read_jsonl(d / "requests.jsonl")
    have = {r["custom_id"] for r in read_jsonl(d / "raw.jsonl")}
    todo = [r for r in reqs if r["custom_id"] not in have]
    bf = d / "batches.json"
    batches = json.loads(bf.read_text()) if bf.exists() else []
    if any(b.get("processing_status") != "ended" for b in batches) and not a.force:
        die("unfinished batches exist: check with status (no overlapping submits; --force to override)")
    prev = probe_done(d) if todo and len(todo) < SMALL_ROUND else None
    if prev:  # small round: the same probe already passed (fixed cost once per dataset)
        print(
            f"{MARK} [TEXT_PROBE] skipped — small round {len(todo)} < {SMALL_ROUND} · same probe passed {prev}",
            flush=True,
        )
    elif todo:  # a batch is submitted only after one probe passes
        probe(d, todo[0])
        todo = todo[1:]
    for part in chunks(todo):
        r = http(
            "POST",
            f"{API}/messages/batches",
            {"requests": [{"custom_id": q["custom_id"], "params": q["params"]} for q in part]},
        )
        batches.append(
            {
                "id": r["id"],
                "n": len(part),
                "processing_status": r.get("processing_status"),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        bf.write_text(json.dumps(batches, indent=1))
        print(f"{MARK} submit {r['id']} n={len(part)}", flush=True)
    print(f"{MARK} submit done requests {len(todo)} · batches {len(batches)}", flush=True)
    return 0


def cmd_status(a) -> int:
    """Refresh the batch states in batches.json."""
    d = Path(a.dir)
    bf = d / "batches.json"
    batches = json.loads(bf.read_text()) if bf.exists() else []
    for b in batches:
        r = http("GET", f"{API}/messages/batches/{b['id']}")
        b.update(
            processing_status=r["processing_status"],
            request_counts=r.get("request_counts"),
            results_url=r.get("results_url"),
        )
        print(f"{MARK} status {b['id']} {r['processing_status']} {r.get('request_counts')}", flush=True)
    bf.write_text(json.dumps(batches, indent=1))
    return 0


def cmd_fetch(a) -> int:
    """Append the results of ended batches to raw.jsonl (rows already received are skipped)."""
    d = Path(a.dir)
    cmd_status(a)
    batches = json.loads((d / "batches.json").read_text())
    have = {r["custom_id"] for r in read_jsonl(d / "raw.jsonl")}
    n_new = 0
    with (d / "raw.jsonl").open("a", encoding="utf-8") as f:
        for b in batches:
            if b.get("processing_status") != "ended" or not b.get("results_url"):
                continue
            for line in http("GET", b["results_url"], raw=True).decode().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["custom_id"] in have:
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                have.add(r["custom_id"])
                n_new += 1
    print(f"{MARK} fetch new {n_new} · received {len(have)}", flush=True)
    return 0
