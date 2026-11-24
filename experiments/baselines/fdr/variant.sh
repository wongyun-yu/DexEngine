#!/bin/bash
# FlashDexRetarget variant file reader (training overrides and the run-log proofs of a variant).
# A variant file (variants/<name>.txt) holds 'override <hydra override>' and 'proof <min count> <ERE>' lines (# = comment);
# eval_record.py --variant reads the same file for its overrides.
#   variant.sh overrides <file>     → the override lines' values, one per line (none = FAIL: a variant changes something)
#   variant.sh check <file> <log>   → one [VARIANT_PROOF] line per proof (count found / needed); rc 0 = all met
set -uo pipefail
cmd=${1:?overrides|check}; f=${2:?variant file}
[ -s "$f" ] || { echo "[VARIANT] FAIL no variant file $f" >&2; exit 2; }
bad=$(grep -a -v -E '^[[:space:]]*(#|$)|^override [^ ]|^proof [1-9][0-9]* [^ ]' "$f" | head -1)
[ -z "$bad" ] || { echo "[VARIANT] FAIL $f: unknown line '$bad' (override | proof | #)" >&2; exit 2; }
case $cmd in
  overrides) n=0; while IFS= read -r ln; do echo "${ln#override }"; n=$((n + 1)); done < <(grep -a '^override ' "$f")
             [ "$n" -gt 0 ] || { echo "[VARIANT] FAIL $f has no override line" >&2; exit 2; };;
  check) L=${3:?log}; ok=0; n=0
         while IFS= read -r ln; do rest=${ln#proof }; need=${rest%% *}; re=${rest#* }; n=$((n + 1))
           got=$(grep -a -c -E -- "$re" "$L" 2>/dev/null); got=${got:-0}
           [ "$got" -ge "$need" ] && s=met || { s=MISSING; ok=1; }
           echo "[VARIANT_PROOF] $s $got/$need $re"; done < <(grep -a '^proof ' "$f")
         [ "$n" -gt 0 ] || { echo "[VARIANT] FAIL $f has no proof line — an unproven variant is void" >&2; exit 2; }
         exit $ok;;
  *) echo "[VARIANT] FAIL command $cmd (overrides|check)" >&2; exit 2;;
esac
