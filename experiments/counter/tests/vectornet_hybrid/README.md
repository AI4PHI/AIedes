# VectorNet hybrid validation

This test combines the completed predictions without retraining:

- observed-lag ensemble when at least one previous egg-rate value is available;
- climate-only ensemble when both previous values are missing.

Run:

```bash
PYTHONPATH=src /home/biazzin/conda-envs/aiedes-env/bin/python \
  experiments/counter/tests/vectornet_hybrid/evaluate_vectornet_hybrid.py
PYTHONPATH=src /home/biazzin/conda-envs/aiedes-env/bin/python \
  experiments/counter/tests/vectornet_scatter/plot_vectornet_scatter.py \
  --input experiments/counter/tests/vectornet_hybrid/results/vectornet_hybrid_predictions.csv \
  --output experiments/counter/tests/vectornet_hybrid/results/plots \
  --label "Hybrid observed-lag / climate-only"
```
