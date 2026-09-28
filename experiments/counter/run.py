#!/usr/bin/env python
"""Run the Counter revision fixed-budget experiment; see README.md."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import time
import numpy as np
import sys

from AIedes.experiment_runner import run_jobs
from AIedes.train.counter_protocol import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "autoregressive":
        from AIedes.train.counter_autoregressive_experiment import main as autoregressive_main
        autoregressive_main(sys.argv[2:])
    else:
        main()
