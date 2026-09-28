#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/biazzin/conda-envs/aiedes-env/bin/python}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/results/autoregressive_jobs}"
MAX_JOBS="${MAX_JOBS:-$(( $(nproc) - 1 ))}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable missing: $PYTHON_BIN" >&2
    exit 1
fi
if (( MAX_JOBS < 1 )); then
    echo "MAX_JOBS must be positive" >&2
    exit 2
fi

mkdir -p "$OUTPUT" "$OUTPUT/logs"
running=0
for seed in 1 2 3 4 5; do
    for fold in 1 2 3 4 5; do
        job="$OUTPUT/64_32_seed${seed}_fold${fold}"
        log="$OUTPUT/logs/64_32_seed${seed}_fold${fold}.log"
        PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" \
            -m experiments.counter.run autoregressive \
            --input "$SCRIPT_DIR/data/eggs_y_norm_7days.pkl" \
            --split-manifest "$SCRIPT_DIR/data/balanced_seed42.json" \
            --output "$job" --widths 64,32 --fold "$fold" --seed "$seed" \
            --workers 1 --threads-per-worker 1 --epochs 3000 --report-every 50 \
            >"$log" 2>&1 &
        running=$((running + 1))
        if (( running >= MAX_JOBS )); then
            wait -n
            running=$((running - 1))
        fi
    done
done
wait
echo "Completed 25 folds/seeds for autoregressive architecture [64,32]."
