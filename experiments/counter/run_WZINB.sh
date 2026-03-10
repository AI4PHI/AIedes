#!/bin/bash

# Wait for 6 hours before starting the execution
echo "Waiting for 1 hours before starting..."
sleep 0  # 1 hours = 3600 seconds

echo "Starting execution after 1 hours delay."

# Set default values for other parameters
LEARNING_RATE=0.0001
TEST_RATIO=0.2
NUM_EPOCHS=3000
FOLDER_RES="./results/WZINB/"
LOSS_TYPE="FrequencyWeightedZeroInflatedNegativeBinomial"
MODEL_TYPE="positive_ff"
DROPOUTS="0.0 0.1 0.2"  # Add the desired values for dropout
NUM_THREADS=1
DEVICE="cpu"
NAME_FILE="./data/eggs_y_norm_7days.pkl"
BATCH_SIZE="64"
BINS="1"


# Arrays of values to iterate over
num_layers_values=(2 3)  # Add the desired values for num_layers
hidden_dim_values=(32 64)  # Add the desired values for hidden_dim
random_seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)  # Add the desired values for RANDOM_SEED
# Array of input variable configurations
input_variable_configs=(
   "t2m_mean_znorm" # temperature
   "t2m_mean_znorm:mean" # temperature weekly mean
   "t2m_mean_znorm tp_sum_znorm" # temperature and precipitation
   "t2m_mean_znorm:mean tp_sum_znorm:mean" # temperature and precipitation weekly mean
   "t2m_mean_znorm tp_sum_znorm d2m_mean_znorm" # temperature, precipitation and dew point
   "t2m_mean_znorm:mean tp_sum_znorm:mean d2m_mean_znorm:mean" # temperature, precipitation and dew point weekly mean
   "t2m_mean_znorm tp_sum_znorm p1zn p2zn" # temperature, precipitation, pressure and previous 2 measurements if present
   "t2m_mean_znorm:mean tp_sum_znorm:mean p1zn p2zn" # temperature, precipitation, pressure and previous 2 measurements if present weekly mean
   "t2m_mean_znorm tp_sum_znorm d2m_mean_znorm p1zn p2zn" # temperature, precipitation, dew point, pressure and previous 2 measurements if present
   "t2m_mean_znorm:mean tp_sum_znorm:mean d2m_mean_znorm:mean p1zn p2zn" # temperature, precipitation, dew point, pressure and previous 2 measurements if present weekly mean
)  # Add different combinations of input variables and stats

# Iterate over different values of RANDOM_SEED
for RANDOM_SEED in "${random_seeds[@]}"; do
  echo "Starting runs with RANDOM_SEED=$RANDOM_SEED"
  for DROPOUT in $DROPOUTS; do
    echo "Using dropout=${DROPOUT}"
    # Iterate over combinations of input variables, num_layers, and hidden_dim
    for num_layers in "${num_layers_values[@]}"; do
      for variables in "${input_variable_configs[@]}"; do
        for hidden_dim in "${hidden_dim_values[@]}"; do
          echo "Running with variables=${variables}, num_layers=${num_layers}, hidden_dim=${hidden_dim}, RANDOM_SEED=${RANDOM_SEED}..."
          
          nice -19 python run.py --variables "$variables" \
                        --num_epochs "$NUM_EPOCHS" \
                        --num_layers "$num_layers" \
                        --hidden_dim "$hidden_dim" \
                        --random_state "$RANDOM_SEED" \
                        --folder_res "$FOLDER_RES" \
                        --input_file "$NAME_FILE" \
                        --model_type "$MODEL_TYPE" \
                        --loss_type "$LOSS_TYPE" \
                        --dropout "$DROPOUT" \
                        --lr "$LEARNING_RATE" \
                        --num_threads $NUM_THREADS \
                        --device "$DEVICE" \
                        --batch_size "$BATCH_SIZE"\
                        --bins $BINS
        done
      done
      echo "Finished all runs for RANDOM_SEED=$RANDOM_SEED and num_layers=$num_layers"
    done
  done
  # Wait for all processes in this iteration to finish before proceeding
  wait
  echo "Finished all runs for RANDOM_SEED=$RANDOM_SEED"
done

echo "All runs completed."
