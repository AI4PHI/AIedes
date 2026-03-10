#!/usr/bin/env bash
set -euo pipefail

# -------- Configuration --------
MAX_JOBS=18    # set your parallelism here
CLEAN_PREVIOUS=false   # hard default; environment will NOT override

# -------- Avoid hidden oversubscription --------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE

# -------- Stop all children on Ctrl-C / TERM --------
graceful_stop() { 
    echo "[STOP] Killing all child jobs..."; 
    pkill -P "$$" || true; 
    exit 130; 
}
trap graceful_stop INT TERM

# -------- Parameters --------
DROPOUTS=(0.0 0.2 0.4)
LAYERS=(2 3)
HIDDEN_DIMS=(32 64)
SEEDS=({1..20})
VARIABLES=(
  "t2m_mean_znorm d2m_mean_znorm tp_sum_znorm p1zn p2zn"
  "t2m_mean_znorm:mean d2m_mean_znorm:mean tp_sum_znorm:mean p1zn p2zn"
  "t2m_mean_znorm tp_sum_znorm"
  "t2m_mean_znorm:mean tp_sum_znorm:mean"
  "t2m_mean_znorm d2m_mean_znorm tp_sum_znorm"
  "t2m_mean_znorm:mean d2m_mean_znorm:mean tp_sum_znorm:mean"
  "t2m_mean_znorm tp_sum_znorm p1zn p2zn"
  "t2m_mean_znorm:mean tp_sum_znorm:mean p1zn p2zn"
  "p1zn p2zn"
)

declare -A LR=( [ZINBLoss]=0.0005 [FrequencyWeightedMSLE]=0.001 [FrequencyWeightedMSE]=0.0001 )
declare -A DIR=( [ZINBLoss]=results/ZINB [FrequencyWeightedMSLE]=results/WMSLE [FrequencyWeightedMSE]=results/WMSE )
LOSSES=(ZINBLoss FrequencyWeightedMSLE FrequencyWeightedMSE)

clean_name() { echo "$1" | sed 's/[^A-Za-z0-9_]/_/g; s/__*/_/g; s/^_//; s/_$//'; }

# -------- Clean previous results if requested --------
if [[ "$CLEAN_PREVIOUS" == "true" ]]; then
    echo "Cleaning previous results..."
    existing_done=$(find results/ -name ".done" 2>/dev/null | wc -l || echo 0)
    if (( existing_done > 0 )); then
        echo "Found $existing_done existing .done files - removing them"
        find results/ -name ".done" -delete 2>/dev/null || true
        find results/ -name "*.log" -delete 2>/dev/null || true
        echo "Cleanup completed"
    else
        echo "No previous .done files found"
    fi
fi

# -------- Create base directories --------
for loss in "${LOSSES[@]}"; do
    mkdir -p "${DIR[$loss]}"
done

# -------- Pre-scan to count total jobs --------
total=0 done=0
for drop in "${DROPOUTS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    for hidden in "${HIDDEN_DIMS[@]}"; do
      for vars in "${VARIABLES[@]}"; do
        for seed in "${SEEDS[@]}"; do
          for loss in "${LOSSES[@]}"; do
            total=$((total+1))
            # Create unique directory for each job
            vars_clean=$(clean_name "$vars")
            job_dir="${DIR[$loss]}/drop${drop}_layer${layer}_hidden${hidden}_vars${vars_clean}_seed${seed}"
            [[ -f "$job_dir/.done" ]] && done=$((done+1))
          done
        done
      done
    done
  done
done

remaining=$((total - done))
echo "=== Job Summary ==="
echo "Total jobs: $total"
echo "Already completed: $done" 
echo "Remaining: $remaining"
echo "Max parallel jobs: $MAX_JOBS"
echo "=================="

[[ $remaining -eq 0 ]] && { echo "All jobs already completed!"; exit 0; }

# -------- Launch with wait -n throttling --------
start=$(date +%s)
completed=0
running=0
job_counter=0

print_eta() {
  local now elapsed avg left eta
  now=$(date +%s)
  elapsed=$((now - start))
  if (( completed > 0 )); then
    avg=$(( elapsed / completed ))
    left=$(( remaining - completed )); (( left < 0 )) && left=0
    eta=$(( left * avg ))
  else
    eta=0
  fi
  printf "[%s] %d/%d completed | Running: %d | Elapsed %02dh%02dm | ETA %02dh%02dm\n" \
    "$(date '+%F %T')" \
    "$completed" "$remaining" "$running" \
    $((elapsed/3600)) $(((elapsed%3600)/60)) \
    $((eta/3600))     $(((eta%3600)/60))
}

# Create main log file
main_log="job_execution_$(date +%Y%m%d_%H%M%S).log"
echo "Main log file: $main_log"

# Main execution loop
for drop in "${DROPOUTS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    for hidden in "${HIDDEN_DIMS[@]}"; do
      for vars in "${VARIABLES[@]}"; do
        for seed in "${SEEDS[@]}"; do
          for loss in "${LOSSES[@]}"; do
            # Create unique directory for each job
            vars_clean=$(clean_name "$vars")
            job_dir="${DIR[$loss]}/drop${drop}_layer${layer}_hidden${hidden}_vars${vars_clean}_seed${seed}"
            
            # Skip if already done
            [[ -f "$job_dir/.done" ]] && continue
            
            # Create job directory
            mkdir -p "$job_dir"
            
            job_counter=$((job_counter+1))
            
            # Throttle: wait for a slot if needed
            if (( running >= MAX_JOBS )); then
              wait -n || true
              completed=$((completed+1))
              running=$((running-1))
              print_eta
            fi

            # Log job start
            job_id="job_${job_counter}_drop${drop}_layer${layer}_hidden${hidden}_seed${seed}_${loss}"
            echo "[$(date '+%F %T')] Starting $job_id" | tee -a "$main_log"

            # Launch job in background
            (
              set -e
              job_log="$job_dir/job.log"
              
              # Log job parameters
              {
                echo "=== Job Started: $(date) ==="
                echo "Job ID: $job_id"
                echo "Parameters:"
                echo "  Dropout: $drop"
                echo "  Layers: $layer" 
                echo "  Hidden dim: $hidden"
                echo "  Variables: $vars"
                echo "  Seed: $seed"
                echo "  Loss: $loss"
                echo "  Output dir: $job_dir"
                echo "=========================="
              } > "$job_log"
              
              # Run the actual command
              nice -19 python run.py \
                --variables "$vars" \
                --num_epochs 3000 \
                --num_layers "$layer" \
                --hidden_dim "$hidden" \
                --random_state "$seed" \
                --folder_res "$job_dir" \
                --input_file ./data/eggs_y_norm_7days.pkl \
                --model_type positive_ff \
                --loss_type "$loss" \
                --dropout "$drop" \
                --lr "${LR[$loss]}" \
                --num_threads 1 \
                --device cpu \
                --batch_size 64 \
                --bins 1 \
                >> "$job_log" 2>&1
              
              # Mark as completed
              {
                echo ""
                echo "=== Job Completed Successfully: $(date) ==="
              } >> "$job_log"
              
              touch "$job_dir/.done"
              echo "[$(date '+%F %T')] Completed $job_id" >> "$main_log"
              
            ) &

            running=$((running+1))
          done
        done
      done
    done
  done
done

# Wait for all remaining jobs to complete
echo "Waiting for remaining $running jobs to complete..."
while (( running > 0 )); do
  wait -n || true
  completed=$((completed+1))
  running=$((running-1))
  print_eta
done

end=$(date +%s)
total_time=$((end - start))
echo ""
echo "=== EXECUTION SUMMARY ==="
echo "All $remaining jobs completed successfully!"
echo "Total execution time: $((total_time/3600))h $((total_time%3600/60))m $((total_time%60))s"
echo "Average time per job: $((total_time/remaining))s"
echo "Main log file: $main_log"
echo "========================"