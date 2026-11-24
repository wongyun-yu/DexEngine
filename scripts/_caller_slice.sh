# The caller's leaf slice under systemd's per-user manager: the one owner of this rule. scripts/capped.sh builds its
# scope in that slice; a low-priority launcher reads the same rule by sourcing this file.
#   caller_cgroup          the cgroup v2 path of this shell (empty if none)
#   caller_slice [cgroup]  the leaf slice name (e.g. app.slice) when the cgroup is under user@<uid>.service; outside
#                          systemd's per-user manager (a login scope, a system service) it prints nothing
caller_cgroup() { sed -n 's/^0:://p' /proc/self/cgroup; }

caller_slice() {  # [cgroup path]
  local cg=${1-$(caller_cgroup)} u rest c s=""
  u=$(id -u)
  case "$cg" in */user@"$u".service/*) rest=${cg#*/user@"$u".service/} ;; *) return 0 ;; esac
  local IFS=/
  for c in $rest; do case "$c" in *.slice) s=$c ;; esac; done
  printf '%s' "$s"
}
