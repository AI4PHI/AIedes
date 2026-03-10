#!/bin/bash

# Set experiment parameters as variables

INPUT_VARIABLESS=("temp_tp" "4Bio")

LAYERSS=("[]" "[10]" "[30]" "[10,10]"\
         "[30,30]" "[10,10,10]" "[30,30,30]"\
         "[40,40]" "[40,40,40]")

NNUM_EPOCHS=100 

# Run the experiment with the variables
for INPUT_VARIABLES in "${INPUT_VARIABLESS[@]}"; do
    for LAYERS in "${LAYERSS[@]}"; do
        echo "Running experiment with INPUT_VARIABLES=${INPUT_VARIABLES} and LAYERS=${LAYERS}"
        python run_exp_estim.py \
            --input_variables "${INPUT_VARIABLES}" \
            --layers "${LAYERS}" \
            --num_epochs "${NNUM_EPOCHS}"
    done
done
