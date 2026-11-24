#!/bin/bash
# Stage 3 entry for callers that pass the launch as environment tokens (`env SIDE=both GPU=0 … bash scripts/stage3_rl.sh
# RUN`): the same as `python3 scripts/stage3_rl.py RUN` (see its --help).
exec python3 "$(dirname "${BASH_SOURCE[0]}")/stage3_rl.py" "$@"
