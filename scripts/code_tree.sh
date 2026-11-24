_code_only() { grep -zvP "\tdocs/|\.md\$"; }; _code_tree_ls() { git ls-tree -rz "$1" | _code_only; }; code_tree_of() { echo "$(_code_tree_ls "$1" | sha1sum | cut -c1-40) $(_code_tree_ls "$1" | tr -cd "\000" | wc -c)"; }; code_dirty() { GIT_OPTIONAL_LOCKS=0 git diff -z --name-only HEAD -- | sed -z "s/^/\t/" | _code_only | tr -cd "\000" | wc -c; }
# Code tree fingerprint: the one owner of the rule (line 1 above, kept on one line).
#   code_tree_of <commit>  prints "<sha1> <code files>": sha1 over the `git ls-tree -rz` listing of the commit without
#                          docs/ and *.md (an exclusion rule: a new folder is code by itself), so docs-only commits keep
#                          it; a count of 0 is a failure.
#   code_dirty             prints the number of uncommitted code paths under the same rule.
# Sourcing this file defines the functions. A caller that sends the rule to another host reads line 1 as a string
# (sed -n 1p) and evaluates it there.
