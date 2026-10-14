# VectorNet trap map

This test creates a single detailed street map of the 40 VectorNet traps
traps using detailed street tiles. Roads, settlements, and local geographic
features are visible beneath the trap overlay. Point size and color indicate
the number of observations per trap, and the red star marks Rome.

Run from the repository root:

```bash
PYTHONPATH=src /home/biazzin/conda-envs/aiedes-env/bin/python \
  experiments/counter/tests/vectornet_map/plot_trap_map.py
```
