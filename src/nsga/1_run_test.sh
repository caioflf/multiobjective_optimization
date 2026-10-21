#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3.11}"
OPTIMIZER="${SCRIPT_DIR}/optimize_nsga.py"
RUN_DIR="${SCRIPT_DIR}/out_data/example_$(date +%Y%m%d_%H%M%S)_$$"
DATABASE="${SCRIPT_DIR}/databases/example.sqlite3"
EVALUATOR="${SCRIPT_DIR}/evaluator_example.py"
SEARCH_SPACE="${EVALUATOR}"

OBJECTIVES="y1:min y2:max y3:min"
CONSTRAINTS="budget"

PARTITIONS=10
POP_SIZE=100
OFFSPRING=${POP_SIZE}
GENERATIONS=10
SEED=2
CROSS_PROB=0.9
CROSS_ETA=10
MUT_PROB=0.25
MUT_ETA=20
PRESSURE=2

WORKERS=8
RETRIES=1
RETRY_DELAY=1
TIMEOUT=30

# Optional arguments are forwarded, for example:
# ./1_run_test.sh --initial-population /path/to/gen_009/generation.json
# ./1_run_test.sh --initial-population /path/to/gen_009/generation.json \
#                 --initial-population /path/to/pareto.json
"${PYTHON}" "${OPTIMIZER}" \
    --run-dir "${RUN_DIR}" \
    --database "${DATABASE}" \
    --search-space "${SEARCH_SPACE}" \
    --evaluator "${EVALUATOR}" \
    --objective "${OBJECTIVES}" \
    --constraint "${CONSTRAINTS}" \
    --partitions "${PARTITIONS}" \
    --pop-size "${POP_SIZE}" \
    --offspring "${OFFSPRING}" \
    --generations "${GENERATIONS}" \
    --seed "${SEED}" \
    --crossover-prob "${CROSS_PROB}" \
    --crossover-eta "${CROSS_ETA}" \
    --mutation-prob "${MUT_PROB}" \
    --mutation-eta "${MUT_ETA}" \
    --selection-pressure "${PRESSURE}" \
    --max-workers "${WORKERS}" \
    --retries "${RETRIES}" \
    --retry-delay "${RETRY_DELAY}" \
    --timeout "${TIMEOUT}" \
    "$@"
