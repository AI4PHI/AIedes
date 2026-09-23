#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-/home/biazzin/conda-envs/aiedes-env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable missing: $PYTHON_BIN (set PYTHON to your environment's Python)" >&2
    exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
mapfile -t LAYOUT < <("$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
cores = {}
for cpu in sorted(os.sched_getaffinity(0)):
    path = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    key = (path.joinpath('physical_package_id').read_text().strip(),
           path.joinpath('core_id').read_text().strip())
    cores.setdefault(key, cpu)
cpus = list(cores.values())[1:]
print(','.join(map(str, cpus)))
print(len(cpus))
print(len(os.sched_getaffinity(0)))
PY
)
if (( ${#LAYOUT[@]} != 3 )) || (( LAYOUT[1] < 1 )); then
    echo 'At least two available physical cores are required to reserve one.' >&2
    exit 1
fi
OUTPUT="$SCRIPT_DIR/results"
WORKERS="${LAYOUT[1]}"
READ_ONLY=0
SMOKE=0
CUSTOM_OUTPUT=0
CUSTOM_WORKERS=0
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        --output|--workers)
            if (( i+1 >= ${#ARGS[@]} )); then
                echo "Missing value for ${ARGS[i]}" >&2; exit 2
            fi
            if [[ "${ARGS[i]}" == --output ]]; then
                OUTPUT="${ARGS[i+1]}"; CUSTOM_OUTPUT=1
            else
                WORKERS="${ARGS[i+1]}"; CUSTOM_WORKERS=1
            fi
            i=$((i+1))
            ;;
        --output=*) OUTPUT="${ARGS[i]#--output=}"; CUSTOM_OUTPUT=1 ;;
        --workers=*) WORKERS="${ARGS[i]#--workers=}"; CUSTOM_WORKERS=1 ;;
        --dry-run|--status|--help|-h) READ_ONLY=1 ;;
        --smoke) SMOKE=1 ;;
    esac
done
if (( SMOKE )); then
    if (( ! CUSTOM_OUTPUT )); then OUTPUT="$SCRIPT_DIR/smoke_results"; fi
    if (( ! CUSTOM_WORKERS && WORKERS > 4 )); then WORKERS=4; fi
fi
if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]] || (( WORKERS > LAYOUT[1] )); then
    echo "--workers must be an integer from 1 to ${LAYOUT[1]} (one physical core reserved)." >&2
    exit 2
fi
echo "Logical CPUs: ${LAYOUT[2]}; workers: $WORKERS; one physical core reserved; niceness: +19"
COMMAND=(nice -n 19 taskset --cpu-list "${LAYOUT[0]}" "$PYTHON_BIN" "$SCRIPT_DIR/run.py"
         --workers "$WORKERS" --output "$OUTPUT" "$@")
if (( READ_ONLY )); then
    exec "${COMMAND[@]}"
fi
mkdir -p "$OUTPUT"
printf 'CPU affinity: %s; workers: %s; niceness increment: +19; arguments:' "${LAYOUT[0]}" "$WORKERS" >> "$OUTPUT/launcher_config.txt"
printf ' %q' "$@" >> "$OUTPUT/launcher_config.txt"
printf '\n' >> "$OUTPUT/launcher_config.txt"
"${COMMAND[@]}" 2>&1 | tee -a "$OUTPUT/run.log"
