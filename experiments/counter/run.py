#!/usr/bin/env python
"""Run the Counter revision fixed-budget experiment; see README.md."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import time
import numpy as np

from AIedes.experiment_runner import run_jobs
from AIedes.train.counter_protocol import main

if __name__ == "__main__":
    main()
