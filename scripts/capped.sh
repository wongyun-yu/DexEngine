#!/usr/bin/env bash
# Run a heavy command **in its own cgroup with a memory cap** (systemd-run --user --scope; needs a systemd user session).
#
# Why: under memory pressure systemd-oomd kills a **whole cgroup**. A heavy command started from an interactive session
#     (a terminal pane or a queue runner) makes that shell's scope the first victim; run in its own scope, only
#     the command dies when it goes over the cap.
#
# Usage:  scripts/capped.sh [-m 16G] [-n name] -- <command> [args…]
#   Runs synchronously (foreground) and returns the exit code unchanged. Over the cap **only that command** dies (rc 137).
#   For long background work wrap this in setsid nohup, or use systemd-run --unit when a service is needed.
set -euo pipefail
MEM=16G; NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    -m) MEM=$2; shift 2 ;;
    -n) NAME=$2; shift 2 ;;
    --) shift; break ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "[capped] no command (capped.sh -m 16G -- cmd …)" >&2; exit 2; }
NAME=${NAME:-cap-$(date +%s)-$$}
# Inherit the caller slice: a new scope is by default a **sibling** under app.slice, so when called inside a slice built
# by a low-priority launcher it would escape and lose that priority. So when the caller is under systemd's per-user
# manager, build in its leaf slice (rule: scripts/_caller_slice.sh). Outside systemd's per-user manager (a login scope
# etc.) no --slice is passed. Inheriting a slice other than app.slice leaves a marker.
source "$(dirname "$(readlink -f "$0")")/_caller_slice.sh" || { echo "[capped] ✗ could not read _caller_slice.sh" >&2; exit 2; }
SL=$(caller_slice)
[ -n "$SL" ] && [ "$SL" != app.slice ] && echo "[LOWPRIO] capped scope=$NAME slice=$SL (inherited caller slice · scripts/capped.sh)" >&2
# If the previous scope with the same name died at the memory cap (Result: oom-kill), systemd keeps a failed unit and the next run dies at once with 'already loaded'
# (a fixed-name caller would then fail on every run). Clear failed units only — live units stay, so nothing is launched twice.
if [ "$(systemctl --user show -p ActiveState --value "$NAME.scope" 2>/dev/null)" = failed ]; then
  systemctl --user reset-failed "$NAME.scope" && echo "[capped] cleared previous failed scope $NAME.scope (reset-failed)" >&2
fi
# --scope: run in a new cgroup while attached to the calling shell (synchronous). Swap 0 = do not slow the whole machine by swapping.
exec systemd-run --user --scope --quiet --unit="$NAME" ${SL:+--slice="$SL"} -p MemoryMax="$MEM" -p MemorySwapMax=0 -- "$@"
