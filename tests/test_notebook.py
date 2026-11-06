"""Regression checks execute the notebook itself; synthetic scores are never published."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'spacecraft_anomaly_detection.ipynb'


class NotebookRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = nbformat.read(NOTEBOOK, as_version=4)
        cls.ns = {}
        # Execute imports/config, then load functions directly from notebook AST.
        exec(compile(cls.nb.cells[1].source, str(NOTEBOOK), 'exec'), cls.ns)
        for cell in cls.nb.cells:
            if cell.cell_type == 'code':
                tree = ast.parse(cell.source)
                functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
                exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), 'exec'), cls.ns)

    def metadata(self, intervals='[[0, 0], [3, 4]]', length=5):
        return pd.DataFrame([dict(chan_id='S-1', spacecraft='SMAP', num_values=length,
                                  anomaly_sequences=intervals)])

    def test_valid_notebook_and_no_error_outputs(self):
        nbformat.validate(self.nb)
        for cell in self.nb.cells:
            if cell.cell_type == 'code':
                compile(cell.source, str(NOTEBOOK), 'exec')
                self.assertFalse(any(output.output_type == 'error' for output in cell.outputs))

    def test_labels_include_both_endpoints_and_keep_test_length(self):
        labels, _ = self.ns['build_test_labels'](self.metadata(), 'S-1', 'SMAP', 5)
        np.testing.assert_array_equal(labels, [1, 0, 0, 1, 1])
        real = pd.read_csv(ROOT / 'labeled_anomalies.csv')
        labels, _ = self.ns['build_test_labels'](real, 'S-1', 'SMAP', 7331)
        self.assertEqual(labels.sum(), 448)
        self.assertEqual(labels[5299], 0)
        self.assertEqual(labels[5300], 1)
        self.assertEqual(labels[5747], 1)
        self.assertEqual(labels[5748], 0)

    def test_label_misalignment_fails_instead_of_clipping(self):
        make = self.ns['build_test_labels']
        invalid = [self.metadata(length=6), self.metadata('[[-1, 2]]'),
                   self.metadata('[[3, 5]]'), self.metadata('[[3, 2]]'),
                   self.metadata('[[1.5, 2]]'), self.metadata('[[True, 2]]'),
                   pd.concat([self.metadata(), self.metadata()]),
                   self.metadata().assign(chan_id='T-1')]
        for metadata in invalid:
            with self.subTest(metadata=str(metadata)), self.assertRaises(ValueError):
                make(metadata, 'S-1', 'SMAP', 5)

    def test_features_do_not_look_into_future(self):
        make = self.ns['make_features']
        signal = np.linspace(-1, 1, 60)
        changed = signal.copy()
        changed[30:] = 1000
        original_features = make(signal)
        changed_features = make(changed)
        pd.testing.assert_frame_equal(original_features.iloc[:30], changed_features.iloc[:30])
        self.assertEqual(len(original_features), len(signal))
        self.assertTrue(np.isfinite(original_features).all().all())
        self.assertEqual(original_features.iloc[0]['difference'], 0)
        self.assertEqual(original_features.iloc[0]['rolling_std'], 0)

    def test_metrics_use_predictions_and_handle_absent_class(self):
        metrics, cm = self.ns['evaluate_model']('example', [0, 1, 1, 0], [0, 1, 0, 0])
        self.assertEqual(metrics['Precision'], 1)
        self.assertEqual(metrics['Recall'], 0.5)
        self.assertAlmostEqual(metrics['F1 Score'], 2 / 3)
        np.testing.assert_array_equal(cm, [[2, 0], [1, 1]])
        metrics, cm = self.ns['evaluate_model']('normal', [0, 0], [0, 0])
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(metrics['F1 Score'], 0)
        with self.assertRaises(ValueError):
            self.ns['evaluate_model']('bad alignment', [0], [0, 1])

    def test_all_cells_run_in_order_on_synthetic_splits(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            train = rng.normal(size=(300, 25)).astype(np.float32)
            test = rng.normal(size=(150, 25)).astype(np.float32)
            test[70:81, 0] += 8
            np.save(directory / 'train.npy', train)
            np.save(directory / 'test.npy', test)
            self.metadata('[[70, 80]]', 150).to_csv(directory / 'labels.csv', index=False)
            namespace = {}
            import matplotlib.pyplot as plt
            # Redirect rich display and stream output; synthetic metrics are not final results.
            with contextlib.redirect_stdout(io.StringIO()):
                for index, cell in enumerate(self.nb.cells):
                    if cell.cell_type != 'code':
                        continue
                    exec(compile(cell.source, f'notebook-cell-{index}', 'exec'), namespace)
                    if index == 1:
                        namespace.update(TRAIN_PATH=directory / 'train.npy',
                                         TEST_PATH=directory / 'test.npy',
                                         LABEL_PATH=directory / 'labels.csv',
                                         OUTPUT_DIR=directory / 'results',
                                         display=lambda *args, **kwargs: None)
                    plt.close('all')
            self.assertEqual(namespace['X_train'].shape, (270, 4))
            self.assertEqual(namespace['X_val'].shape, (30, 4))
            self.assertEqual(namespace['X_test'].shape, (150, 4))
            np.testing.assert_allclose(namespace['scaler'].mean_, namespace['fit_features'].mean())
            self.assertEqual(namespace['scaler'].n_samples_seen_, 270)
            self.assertEqual(namespace['model_if'].max_samples_, 256)
            self.assertEqual(namespace['threshold'],
                             float(np.percentile(namespace['validation_errors'], 95)))
            expected = (namespace['errors_test'] > namespace['threshold']).astype(int)
            np.testing.assert_array_equal(namespace['pred_ae'], expected)
            self.assertEqual(namespace['y_test'].sum(), 11)
            self.assertEqual(namespace['autoencoder'].count_params(), 135)
            saved = pd.read_csv(directory / 'results' / 'test_metrics.csv')
            np.testing.assert_allclose(saved['F1 Score'], namespace['results']['F1 Score'])
            self.assertEqual(len(pd.read_csv(directory / 'results' / 'test_predictions.csv')), 150)
            audit = json.loads((directory / 'results' / 'run_metadata.json').read_text())
            self.assertNotEqual(audit['train_sha256'], audit['test_sha256'])
            self.assertEqual(audit['ae_threshold'], namespace['threshold'])


if __name__ == '__main__':
    unittest.main()
