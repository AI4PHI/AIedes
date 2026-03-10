#!/bin/bash

# Wait for 6 hours before starting the execution
echo "Waiting for 1 hours before starting..."
sleep 0  # 1 hours = 3600 seconds

echo "Starting execution after 1 hours delay."

# Set default values for other parameters
LEARNING_RATE=0.00002
TEST_RATIO=0.2
NUM_EPOCHS=3000
FOLDER_RES="./results/test_znorm/"
LOSS_TYPE="FrequencyWeighted"
MODEL_TYPE="positive_ff"
DROPOUT=0.2
NUM_THREADS=2
DEVICE="cpu"
NAME_FILE="./data/eggs_y_norm_7days.pkl"
BATCH_SIZE=64"
BINS="1"
# Arrays of values to iterate over
num_layers_values=(4)  # Add the desired values for num_layers
hidden_dim_values=(32)  # Add the desired values for hidden_dim
random_seeds=(1)  # Add the desired values for RANDOM_SEED
# Array of input variable configurations
input_variable_configs=(
    "t2m_mean_znorm:30 p1zn p2zn tp_sum_znorm:30"    
)  # Add different combinations of input variables and stats

# Iterate over different values of RANDOM_SEED
for RANDOM_SEED in "${random_seeds[@]}"; do
  echo "Starting runs with RANDOM_SEED=$RANDOM_SEED"

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
                      --num_threads "$NUM_THREADS" \
                      --device "$DEVICE" \
                      --batch_size "$BATCH_SIZE"\
                      --bins $BINS
      done
    done
    wait
    echo "Finished all runs for RANDOM_SEED=$RANDOM_SEED and num_layers=$num_layers"
  done

  # Wait for all processes in this iteration to finish before proceeding
  wait
  echo "Finished all runs for RANDOM_SEED=$RANDOM_SEED"
done

echo "All runs completed."