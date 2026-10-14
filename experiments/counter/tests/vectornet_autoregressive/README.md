# VectorNet external validation with autoregressive training

This test evaluates the completed 25-model `[64, 32]` autoregressive ensemble
(five trap-disjoint folds and five seeds) on the independent VectorNet dataset.
Previous egg-rate values are not supplied to the model: predictions are generated
recursively from a zero initial rate, matching the training and deployment mode.

Run from the repository root:

```bash
PYTHONPATH=src /home/biazzin/conda-envs/aiedes-env/bin/python \
  experiments/counter/tests/vectornet_autoregressive/evaluate_vectornet_autoregressive.py
PYTHONPATH=src /home/biazzin/conda-envs/aiedes-env/bin/python \
  experiments/counter/tests/vectornet_scatter/plot_vectornet_scatter.py \
  --input experiments/counter/tests/vectornet_autoregressive/results/vectornet_autoregressive_predictions.csv \
  --output experiments/counter/tests/vectornet_autoregressive/results/plots
```

The outputs contain the autoregressive predictions, the trap/lag-availability
summary, and the same four period/scale scatter panels as the observed-lag test.
