# Longest VectorNet trap time series

This test uses the observed-lag VectorNet ensemble predictions and selects the
six traps with the most observations. The plotted prediction is the mean of the
25 available fold/seed models (five folds × five seeds), with ±1 standard
deviation error bars showing ensemble fluctuation.

Run from the repository root:

```bash
PYTHONPATH=src /home/biazzin/conda-envs/aiedes-env/bin/python \
  experiments/counter/tests/vectornet_timeseries/plot_longest_traps.py
```
