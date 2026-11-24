#!/bin/bash
# fetch_weights.sh — downloads, with your own Hugging Face token, exactly the weights E5 used into an HF hub cache
# (the HF_HUB_CACHE layout the E5 training and evaluation read offline). The weights are not in this repository: gated
# weights are fetched by each user after accepting their licence.
# What to fetch = weights_manifest.tsv next to this script (repo · full revision · file · bytes · sha256, made from the
# caches E5 trained with). Every file is checked by size and sha256 after the download; any miss → exit 1.
# A gated repo with no token / no accepted licence → prints the model page to accept on and exits 3 (nothing guessed).
#   HF_TOKEN=<your token> bash fetch_weights.sh <HF hub cache dir> [python]
#   then: export HF_HUB_CACHE=<that dir> HF_HUB_OFFLINE=1 · WAN_VAE_PATH=<dir>/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/
#         921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth (the last line prints it)
set -uo pipefail
C=${1:?usage: fetch_weights.sh <HF hub cache dir> [python]}; PY=${2:-python3}
MAN=$(dirname "$(readlink -f "$0")")/weights_manifest.tsv
test -s "$MAN" || { echo "[E5_WEIGHTS] FAIL no manifest $MAN"; exit 1; }
mkdir -p "$C" || exit 1
"$PY" - "$C" "$MAN" <<'PYEOF'
import collections
import hashlib
import os
import sys

cache, man = sys.argv[1], sys.argv[2]
try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
except ImportError as e:
    sys.exit(f"[E5_WEIGHTS] FAIL huggingface_hub missing ({e}) — pip install 'huggingface_hub>=0.30'")
rows = collections.OrderedDict()
for ln in open(man):
    repo, rev, f, size, sha = ln.rstrip("\n").split("\t")
    rows.setdefault((repo, rev), []).append((f, int(size), sha))
bad = 0
for (repo, rev), files in rows.items():
    url = f"https://huggingface.co/{repo}"
    try:
        snap = snapshot_download(repo, revision=rev, cache_dir=cache, allow_patterns=[f for f, _, _ in files])
    except (GatedRepoError, RepositoryNotFoundError) as e:
        print(f"[E5_WEIGHTS] GATED {repo}: accept the licence at {url} with your account, set HF_TOKEN, re-run "
              f"({type(e).__name__})", flush=True)
        sys.exit(3)
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403):
            print(f"[E5_WEIGHTS] GATED {repo}: HTTP {code} — accept at {url}, set HF_TOKEN, re-run", flush=True)
            sys.exit(3)
        raise
    for f, size, sha in files:
        p = os.path.join(snap, f)
        if not os.path.exists(p) or os.path.getsize(p) != size:
            print(f"[E5_WEIGHTS] FAIL {repo}@{rev[:12]} {f}: missing or size ≠ {size}", flush=True)
            bad += 1
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        if h.hexdigest() != sha:
            print(f"[E5_WEIGHTS] FAIL {repo}@{rev[:12]} {f}: sha256 {h.hexdigest()[:12]} ≠ {sha[:12]}", flush=True)
            bad += 1
    print(f"[E5_WEIGHTS] {repo}@{rev[:12]} {len(files)} files checked → {snap}", flush=True)
if bad:
    sys.exit(f"[E5_WEIGHTS] FAIL {bad} file(s)")
print(f"[E5_WEIGHTS] DONE {sum(len(v) for v in rows.values())} files sha256 = E5's · HF_HUB_CACHE={cache}", flush=True)
PYEOF
rc=$?
[ $rc = 0 ] && echo "[E5_WEIGHTS] export HF_HUB_CACHE=$C HF_HUB_OFFLINE=1 WAN_VAE_PATH=$C/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
exit $rc
