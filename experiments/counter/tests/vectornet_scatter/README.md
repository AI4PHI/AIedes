# VectorNet external-validation scatter plot

This test experiment plots observed versus predicted weekly egg-laying rates for
the independent VectorNet 2021 subset. It uses the completed predictions written
by the Counter reporting pipeline; it does not retrain or retune the models.

Run from the repository root with:

```bash
/home/biazzin/conda-envs/aiedes-env/bin/python \
  experiments/counter/tests/vectornet_scatter/plot_vectornet_scatter.py
```

Outputs are written to `results/` in this folder:

- `vectornet_scatter.png`
- `vectornet_scatter.pdf`
- `vectornet_scatter_metrics.json`

The x-axis is the observed weekly rate and the y-axis is the observed-lag model
prediction. The four panels compute metrics separately for observations before
1 September 2021 and observations from 1 September onward, using both raw
linear and `log1p` scales. `log1p` is used because the dataset contains zero
egg-laying rates. Each period has one uniform point color; colors do not encode
the individual observations within a panel.
