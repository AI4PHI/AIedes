#!/usr/bin/env bash
set -euo pipefail

# Run the complete reproducible Classifier analysis from a clean output
# directory.  Existing results are moved to a timestamped backup instead of
# being deleted, so an interrupted or accidental rerun remains recoverable.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${AIEDES_PYTHON:-python}"
SOURCE_ARCHIVE="$ROOT_DIR/experiments/classifier/data/ecdc_albopictus_cordex_2020.zip"
OUTPUT_DIR="$ROOT_DIR/experiments/classifier/paper_results"
PREPARE_SCRIPT="$ROOT_DIR/experiments/classifier/paper_pipeline/prepare_classifier_data.py"
EXPERIMENT_SCRIPT="$ROOT_DIR/experiments/classifier/run_paper_experiments.py"
MAP_SCRIPT="$ROOT_DIR/experiments/classifier/img_for_paper/make_classifier_comparison_map.py"
SELECTION_FIGURE_SCRIPT="$ROOT_DIR/experiments/classifier/img_for_paper/make_combined_loss_acc.py"

EPOCHS="${AIEDES_EPOCHS:-500}"
BATCH_SIZE="${AIEDES_BATCH_SIZE:-4096}"
JOBS="${AIEDES_JOBS:-32}"

usage() {
    echo "Usage: AIEDES_JOBS=32 AIEDES_EPOCHS=500 $0"
    echo "       $0 --jobs 16 --epochs 500 --batch-size 4096"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$SOURCE_ARCHIVE" ]]; then
    echo "Classifier source archive not found: $SOURCE_ARCHIVE" >&2
    exit 1
fi

cd "$ROOT_DIR"

# Limit each individual neural-network fit to one CPU thread.  Parallelism is
# across independent architectures, folds, seeds, and validation schemes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/aiedes-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ -e "$OUTPUT_DIR" ]]; then
    backup_dir="${OUTPUT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Moving existing results to: $backup_dir"
    mv -- "$OUTPUT_DIR" "$backup_dir"
fi
mkdir -p "$OUTPUT_DIR"

echo "Preparing checked classifier input..."
"$PYTHON_BIN" -u "$PREPARE_SCRIPT" \
    --source "$SOURCE_ARCHIVE" \
    --output-dir "$OUTPUT_DIR/input"

echo "Running selection, held-out evaluation, random OOF map predictions, and NUTS3 sensitivity..."
"$PYTHON_BIN" -u "$EXPERIMENT_SCRIPT" \
    --stage all \
    --data "$OUTPUT_DIR/input/copernicus_classifier_prepared.csv.gz" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --jobs "$JOBS"

echo "Rendering the main classifier map..."
"$PYTHON_BIN" -u "$MAP_SCRIPT" \
    --evaluation random \
    --results-dir "$OUTPUT_DIR" \
    --output "$ROOT_DIR/paper/img/comparison_lr_nn.png"

echo "Rendering the supplementary NUTS3 classifier map..."
"$PYTHON_BIN" -u "$MAP_SCRIPT" \
    --evaluation nuts3 \
    --results-dir "$OUTPUT_DIR" \
    --output "$ROOT_DIR/paper/img/comparison_lr_nn_nuts3.png"

echo "Rendering the classifier model-selection figure..."
"$PYTHON_BIN" -u "$SELECTION_FIGURE_SCRIPT" \
    --results-dir "$OUTPUT_DIR" \
    --output "$ROOT_DIR/paper/img/combined_loss_acc.pdf"

echo
echo "Complete classifier pipeline finished."
echo "Results: $OUTPUT_DIR"
echo "Main map: $ROOT_DIR/paper/img/comparison_lr_nn.png"
echo "NUTS3 map: $ROOT_DIR/paper/img/comparison_lr_nn_nuts3.png"
