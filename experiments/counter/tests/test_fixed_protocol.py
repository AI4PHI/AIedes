"""Scientific integrity checks for the fixed-budget, multi-loss revision."""
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'src')]
import numpy as np
import pandas as pd
import torch
from protocol import (grid, load_protocol, metadata, with_id, execute_job, make_job,
                      supplementary_jobs, partitions, summary)
from AIedes.data_loader.counter_prepare import synthetic_counter_data
from AIedes.data_loader.counter_data_loader import (fit_counter_preprocessing, transform_counter_features,
                                                   counter_feature_names)
from AIedes.evaluation.counter_reconstruction import load_model, observed_prediction, trajectory, build_weather
from AIedes.train.counter_train import fit_counter_candidate
from AIedes.utils.counter_experiment import complete, read_json
from AIedes.utils.counter_losses import ZINBLoss


class FixedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_protocol(HERE / 'config.json')
        self.candidates = grid(self.config)
        self.config.update(max_epochs=2, bootstrap_samples=10, patience=1)
        self.frame, self.full = synthetic_counter_data()
        self.frame['row_id'] = np.arange(len(self.frame))

    def candidate(self, loss='WMSLE'):
        return next(c for c in self.candidates if c['loss'] == loss and c['lags']
                    and c['window'] == 90 and c['fields'] == ['t2m_mean', 'd2m_mean', 'tp_sum']
                    and not c['weekly'] and c['widths'] == [32] and c['dropout'] == .2)

    def test_grid_and_exact_supplement_budget(self):
        self.assertEqual(len(self.candidates), 516)
        self.assertEqual(sum(c['window'] in (None, 90) for c in self.candidates), 324)
        self.assertTrue(all(c['loss'] == 'WMSLE' for c in self.candidates if c['window'] in (30, 60)))
        anchor = self.candidate()
        selected = dict(main=anchor, lag_anchor=anchor)
        jobs = supplementary_jobs(Path('/tmp/unused-fixed-tests'), selected, self.config, 'test')
        self.assertEqual(sum(j['kind'] == 'lag_order' for j in jobs), 25)
        self.assertEqual(sum(j['kind'] == 'sensitivity' for j in jobs), 25)
        one = next(j['candidate'] for j in jobs if j['kind'] == 'lag_order')
        self.assertEqual({k:v for k,v in one.items() if k not in ('lag_order','id')},
                         {k:v for k,v in anchor.items() if k != 'id'})
        self.assertEqual(one['lag_order'], 1)
        for fold in range(1, 6):
            job = make_job(Path('/tmp/unused-fixed-tests'), anchor, fold, 1, self.config, 'test')
            train, test, val = partitions(self.frame, job)
            self.assertIsNone(val)
            self.assertEqual(len(train), 96)
            self.assertEqual(len(test), 24)
            self.assertFalse(set(self.frame.iloc[train].id_trap) & set(self.frame.iloc[test].id_trap))

    def test_one_lag_does_not_use_second_lag_or_test_normalization(self):
        one = with_id(dict(self.candidate(), lag_order=1))
        train = np.flatnonzero(self.frame.outer_group.ne(1))
        stats = fit_counter_preprocessing(self.frame, train, one)
        self.assertNotIn('prev2_rates', stats)
        x = transform_counter_features(self.frame, one, stats)
        changed = self.frame.copy()
        changed['prev2_rates'] = 1e10
        np.testing.assert_array_equal(x, transform_counter_features(changed, one, stats))
        self.assertEqual(x.shape[1], 272)
        self.assertEqual(counter_feature_names(one)[-2:], ['p1:value','p1:available'])
        changed.loc[changed.outer_group.eq(1), 'prev1_rates'] = 1e10
        self.assertEqual(stats, fit_counter_preprocessing(changed, train, one))

    def test_all_losses_reload_fixed_budget_and_determinism(self):
        train = np.flatnonzero(self.frame.outer_group.ne(1))
        test = np.flatnonzero(self.frame.outer_group.eq(1))
        for loss in ('WMSE', 'WMSLE', 'ZINB'):
            candidate = self.candidate(loss)
            package, history, _ = fit_counter_candidate(self.frame, train, None, candidate, self.config, 1, epochs=2)
            self.assertEqual(package['epochs_trained'], 2)
            self.assertEqual(package['best_epoch'], 2)
            self.assertEqual(history[0]['learning_rate'], candidate['learning_rate'])
            self.assertEqual(package['optimizer_steps_trained'], 4)
            with tempfile.TemporaryDirectory() as tmp:
                file = Path(tmp) / 'model.pt'; torch.save(package, file)
                model, restored = load_model(file)
                p = observed_prediction(model, restored, self.frame.iloc[test])
                self.assertTrue(np.isfinite(p).all() and (p >= 0).all())
                altered = self.frame.copy(); altered.loc[test, 'weeklyRates'] = 1e12
                other, _, _ = fit_counter_candidate(altered, train, None, candidate, self.config, 1, epochs=2)
                for key, value in package['state_dict'].items():
                    self.assertTrue(torch.equal(value, other['state_dict'][key]), key)
            if loss == 'ZINB':
                self.assertIn('theta_raw', package['criterion_state_dict'])

    def test_zinb_fractional_rates_and_exact_zero(self):
        loss = ZINBLoss(zero_threshold=0.)
        pred = {'mu_raw': torch.tensor([1., 1.]), 'logit_pi': torch.tensor([2., 2.])}
        targets = torch.tensor([0., .25])
        actual = loss(pred, targets)
        old = ZINBLoss(zero_threshold=.5)(pred, targets)
        self.assertTrue(torch.isfinite(actual))
        self.assertNotAlmostEqual(float(actual.detach()), float(old.detach()))

    def test_single_lag_recursive_reload_and_resume(self):
        candidate = with_id(dict(self.candidate(), lag_order=1))
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp), candidate, 1, 1, self.config, 'test', 'lag_order')
            execute_job(job, self.frame)
            self.assertTrue(complete(job['folder'], 'test'))
            model, package = load_model(Path(job['folder']) / 'model.pt')
            weather, _, _ = build_weather(self.frame, self.full)
            obs = self.frame.loc[self.frame.id_trap.eq(0)]
            prediction = trajectory(model, package, weather[0], obs.end_date.min(), obs.end_date.max())
            self.assertEqual(prediction.startup.sum(), 7)
            self.assertTrue(np.isfinite(prediction.prediction).all())
            before = (Path(job['folder']) / 'model.pt').stat().st_mtime_ns
            self.assertEqual(execute_job(job, self.frame)[1], 'cached')
            self.assertEqual(before, (Path(job['folder']) / 'model.pt').stat().st_mtime_ns)
            record = read_json(Path(job['folder']) / 'result.json')
            self.assertIn('test_loss', record)
            self.assertEqual(record['n_train'], 96)

    def test_fold_summary_averages_seeds_first(self):
        records = []
        for fold, seed, metric in [(1,1,0.), (1,2,1.), (2,1,1.), (2,2,1.)]:
            records.append(dict(metadata(self.candidate()), fold=fold, seed=seed, parameters=1, test_r2=metric))
        result = summary(records).iloc[0]
        self.assertAlmostEqual(result.test_r2_mean, .75)
        self.assertAlmostEqual(result.test_r2_se, .25)


if __name__ == '__main__':
    unittest.main()
