# VectorNet trap altitude

This test queries OpenTopoData's SRTM 90 m elevation dataset for the 40
unique VectorNet trap coordinates. The returned values are terrain-elevation
estimates in metres, interpolated bilinearly from the DEM; they are not direct
altimeter measurements.

Run from the repository root:

```bash
PYTHONPATH=src /home/biazzin/conda-envs/aiedes-env/bin/python \
  experiments/counter/tests/vectornet_altitude/query_altitude.py
```
