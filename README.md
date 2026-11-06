# Spacecraft Telemetry Anomaly Detection (NASA SMAP & MSL)

Unsupervised anomaly detection on spacecraft telemetry from the **NASA SMAP & MSL anomaly dataset**, comparing two approaches:

1. **Isolation Forest**: a classical tree-based baseline
2. **Dense Autoencoder**: a neural network that flags anomalies by reconstruction error

The project uses telemetry channel **S-1** from SMAP as a worked example, from loading and exploring the data through training, scoring, and comparing both models.

> **Evaluation corrected:** Models fit only the chronological training prefix. Autoencoder calibration uses validation data, and final metrics use the independent official S-1 test split. The notebook contains graphs and outputs from the latest complete S-1 run.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Spacecraft generate large volumes of telemetry (sensor readings and command/status signals). Detecting abnormal behavior early helps operators catch faults before they become failures. Labeled anomalies are rare, so this project uses **unsupervised** methods: the models learn what "normal" looks like and flag deviations from it. Labels are used only to evaluate the results.

**Goals**

- Load and explore a real NASA telemetry channel
- Build an Isolation Forest baseline
- Build a dense autoencoder and detect anomalies via reconstruction error
- Evaluate both models with the same metrics and compare them
- Be explicit about what the results do and do not show

---

## Dataset

**Source:** NASA Soil Moisture Active Passive (SMAP) satellite and Mars Science Laboratory (MSL) rover telemetry, released with the paper *"Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"* (Hundman et al., KDD 2018). The data and labels are distributed through the [`khundman/telemanom`](https://github.com/khundman/telemanom) repository.

| Item | Description |
|---|---|
| Spacecraft | SMAP (satellite) and MSL (Mars rover) |
| Files | `train/<channel>.npy`, `test/<channel>.npy`, `labeled_anomalies.csv` |
| Features | 25 per time step for SMAP (55 for MSL) |
| Feature 0 | The telemetry value for the channel |
| Other features | Encoded command/status information, mostly sparse |
| Labels | Anomalous time-step ranges listed in `labeled_anomalies.csv` |

### Channel used: S-1 (SMAP)

The official dataset has **two files named `S-1.npy`**, in separate `train/` and `test/` directories. `T-1.npy` identifies another SMAP channel; it is not the S-1 test split. This project uses the official S-1 pair.

The supplied S-1 metadata expects **7,331 test time steps**, with the zero-based inclusive anomaly interval **[5300, 5747]** (448 anomalous points). The notebook checks this against the actual test length and rejects invalid intervals rather than clipping them. Array rows are time steps; only column 0 is selected as telemetry, without flattening the other features into time.

---

## Project Structure

```text
.
|-- spacecraft_anomaly_detection.ipynb  # Complete corrected experiment
|-- README.md
|-- labeled_anomalies.csv
|-- requirements.txt
|-- tests/test_notebook.py                 # Synthetic regression checks
|-- data/
|   |-- train/S-1.npy                     # Official training split (supply separately)
|   `-- test/S-1.npy                      # Official independent test split
`-- results/                             # Generated only when the notebook runs
    |-- test_metrics.csv
    |-- test_predictions.csv
    `-- run_metadata.json
```

---


## Setup

### Requirements

Use Python 3.11+ with compatible NumPy, pandas, matplotlib, Seaborn, scikit-learn, TensorFlow, and Jupyter versions:

```bash
python -m pip install -r requirements.txt
```

### Get the data

Obtain the official data through the [Telemanom repository](https://github.com/khundman/telemanom#data). Preserve the `train/` and `test/` directory names and copy the S-1 arrays into the layout above. The local workspace has the S-1 arrays under `data/`; this directory is excluded from Git, so a fresh clone must supply the arrays. Do not copy a single array into both directories, or substitute the T-1 channel.

### Run

In VS Code, select the project `.venv/Scripts/python.exe` as the notebook kernel (the system Python may not have all dependencies). Open `spacecraft_anomaly_detection.ipynb` in Jupyter and use **Restart Kernel and Run All** from this project directory. In Colab, upload the same directory layout and adjust `DATA_DIR` and `LABEL_PATH` in the configuration cell. There is no mandatory Colab dependency.

The notebook raises explicit errors for missing files, duplicate arrays, incompatible shapes, mismatched metadata, and invalid intervals. A matching length is a sanity check, not proof of split provenance.

Run the regression checks with:

```bash
python -m unittest discover -s tests -v
```

These checks use temporary synthetic telemetry and do not create research performance claims.

---

## Methodology

### 1. Training and validation

Reserve the final 10% of the official training sequence for chronological validation. Compute telemetry value, first difference, trailing 20-step mean, and trailing standard deviation without future samples, shuffling, or dropping rows. Validation features retain preceding training history. Test history starts fresh because split adjacency is not guaranteed.

Fit `StandardScaler` on only the first 90% of training features. Transform validation and test features with that scaler. Both models fit this same training prefix; neither is refitted on the validation tail or test data.

### 2. Isolation Forest

Preserve 100 trees, contamination 0.02, and seed 42. Fit on `X_train`, then predict on `X_test`. The decision offset comes from training scores; 0.02 does not guarantee a 2% anomaly rate on independent test data.

### 3. Dense autoencoder

Preserve the 4 → 8 → 3 → 8 → 4 dense architecture (ReLU hidden layers, linear output), Adam/MSE, 15 epochs, batch size 128, and `shuffle=False`. Use explicit chronological validation data. Inputs and reconstruction targets are both `X_train`, which is correct for an autoencoder.

Set the anomaly threshold to the fixed **95th percentile of validation reconstruction errors**. Compute final reconstruction errors and predictions on `X_test`. No test labels, test error percentiles, or test F1 optimization influence the threshold. This is a fixed unsupervised heuristic, not a guarantee of calibrated false positive rates.

### 4. Final evaluation and visualizations

Read S-1/SMAP test labels from the CSV for the original metadata exploration and final evaluation. Labels do not enter feature scaling, model fitting, or threshold calibration. Convert inclusive intervals with `labels[start:end + 1] = 1`; require one matching row, the exact test length, and in-bounds integer endpoints. There is only one test label vector, `y_test`.

Use one shared evaluation function for precision, recall, F1, and confusion matrices, with anomaly as class 1 and `zero_division=0`. These are **point-wise** metrics, without point adjustment or event-based credit. Matplotlib and Seaborn plots include the training distribution and feature correlations, telemetry from both raw splits, actual test intervals, each model's test flags, validation learning curves, test reconstruction errors against the validation threshold, confusion matrices, and dynamically calculated comparison bars.

Model settings are fixed before test prediction. Any later tuning based on these test results would require a new independent evaluation set.

---

## Results

Latest complete local S-1 run (point-wise evaluation; 7,331 test readings):

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Isolation Forest | 0.5825 | 0.4018 | 0.4756 |
| Dense Autoencoder | 0.3631 | 0.6451 | 0.4646 |

Graphs and executed outputs are saved in the notebook. These single-channel results do not establish statistical superiority or generalization.

A successful full run generates the comparison table, charts, and conclusions directly from current predictions. `results/test_metrics.csv` stores actual precision/recall/F1, `test_predictions.csv` stores aligned labels and predictions, and `run_metadata.json` records split hashes, paths, parameters, threshold, sample counts, seed, and package versions. No model ranking is asserted until a real test run completes.

---

## Limitations

- **Single channel:** S-1 results do not establish generalization to other SMAP/MSL channels or statistical superiority.
- **Limited temporal context:** Rolling features add past context, but the dense autoencoder is not a sequence model.
- **Threshold sensitivity:** Fixed contamination and validation percentile affect the precision/recall tradeoff. Validation contamination or distribution shift may undermine calibration.
- **Point-wise metrics:** Long events receive more weight; these metrics do not measure detection delay or event coverage.
- **Reproducibility:** Seeds and deterministic TensorFlow operations improve repeatability. Hardware and dependency versions can still affect numerical results; run metadata records versions rather than claiming cross-platform bitwise equivalence.
- **Provenance:** Hash and shape checks reject obvious overlap and misalignment but cannot prove files were obtained from the official splits.

---

## Future Work

- Evaluate additional channels with a separate development protocol and aggregate results.
- Compare temporal models and range/event metrics, including detection delay.
- Study validation-only threshold robustness without selecting configurations on the final test labels.

---

## References

- Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018). *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest.* IEEE ICDM 2008.
- NASA SMAP & MSL telemetry data: [github.com/khundman/telemanom](https://github.com/khundman/telemanom)

---

## License & Acknowledgements

This project is proudly open-sourced under the **MIT License**. You are free to use, modify, distribute, and commercialize the code, provided you include the original copyright and license notice. See the `LICENSE` file for more details.

> The NASA SMAP & MSL dataset and its labels belong to their original authors and are **not** covered by this repository's license. They are distributed under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0) via the Telemanom project.

### Acknowledgements

A massive thank you to:

* The [Telemanom](https://github.com/khundman/telemanom) team at NASA JPL for releasing the SMAP & MSL anomaly detection dataset and labels:
  * Kyle Hundman
  * [Valentinos Constantinou](https://github.com/vc1492a)
  * Chris Laporte
  * [Ian Colwell](https://github.com/iancolwell)
* The [scikit-learn](https://scikit-learn.org/) maintainers for the Isolation Forest implementation.
* The [TensorFlow / Keras](https://www.tensorflow.org/) teams for the tools used to build the autoencoder.
* [Google Colab](https://colab.research.google.com/) for the free notebook environment.
* The open-source community for the endless inspiration.

If you use this dataset, please cite the original paper listed in [References](#references).

Built with ❤️ by [piyushops01](https://github.com/piyushops01) and contributors.


