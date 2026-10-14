"""Focused tests for the revised classifier paper workflow."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from AIedes.data_loader.classificator_data_loader import (
    apply_standardizer,
    fit_standardizer,
)
from AIedes.evaluation.classificator_evaluate import classification_metrics


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "classifier_paper_experiments",
    ROOT / "experiments/classifier/run_paper_experiments.py",
)
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)


class ClassifierRevisionTests(unittest.TestCase):
    def test_standardizer_uses_supplied_training_values(self):
        training = np.asarray([[0.0, 10.0], [2.0, 10.0]], dtype=np.float32)
        held_out = np.asarray([[4.0, 12.0]], dtype=np.float32)
        scaler = fit_standardizer(training)
        transformed = apply_standardizer(held_out, scaler)
        np.testing.assert_allclose(scaler["mean"], [1.0, 10.0])
        np.testing.assert_allclose(scaler["std"], [1.0, 1.0])
        np.testing.assert_allclose(transformed, [[3.0, 2.0]])

    def test_binary_metrics(self):
        metrics = classification_metrics(
            labels=[1, 1, 0, 0],
            probabilities=[0.9, 0.2, 0.8, 0.1],
        )
        self.assertEqual(
            {key: metrics[key] for key in ("tp", "tn", "fp", "fn")},
            {"tp": 1, "tn": 1, "fp": 1, "fn": 1},
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["csi"], 1 / 3)

    def test_nuts3_folds_are_disjoint_and_cover_every_row(self):
        groups = np.repeat([f"R{index:02d}" for index in range(20)], 2)
        target = np.tile([0, 1], 20)
        dataframe = pd.DataFrame({"NUTS3_Code": groups})
        X = np.arange(len(target), dtype=np.float32).reshape(-1, 1)
        arguments = SimpleNamespace(map_folds=5, nuts3_folds=5, seed=42)
        folds, manifest = PIPELINE.comparison_folds(
            X, target, dataframe, "nuts3", arguments
        )
        held_out = []
        for train_idx, test_idx in folds:
            self.assertFalse(set(groups[train_idx]).intersection(groups[test_idx]))
            held_out.extend(test_idx.tolist())
        self.assertEqual(sorted(held_out), list(range(len(target))))
        self.assertTrue(manifest)


if __name__ == "__main__":
    unittest.main()
