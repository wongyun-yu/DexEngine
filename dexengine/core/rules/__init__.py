"""The rule registry behind `dexengine.core.datasets` (which re-exports every name).

source      read a `path:symbol` reference from its owning file without importing it
profiles    per-dataset pipeline profiles and the scopes derived from them
identity    active arm / hand of the process, the axis hand, combo names
outputs     per-hand output files of the data stages
ledger      final-ledger names and stage-1 exclusion lists
ruleset     rule-set versions of the final ledgers
spec, cards the `Rule` record and the rule list
rule        rule queries (`resolve`, `applies`, `covers`, `rule_datasets`, `rule_scenes`)
reader_calls the reader-call scope: datasets whose reader calls a rule's owner function
thresholds  thresholds bound to hand geometry
cli         the command line
"""
