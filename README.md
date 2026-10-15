# Spacecraft Telemetry Anomaly Detection (NASA SMAP & MSL)

Unsupervised anomaly detection on spacecraft telemetry from the **NASA SMAP & MSL anomaly dataset**, comparing two approaches:

1. **Isolation Forest**: a classical tree-based baseline
2. **Dense Autoencoder**: a neural network that flags anomalies by reconstruction error

The project uses telemetry channel **S-1** from SMAP as a worked example, from loading and exploring the data through training, scoring, and comparing both models.

> **Important:** Both models were evaluated on the same sequence they were trained on. The reported scores are **not** held-out test performance and should **not** be read as evidence that the models generalize to other telemetry channels. See [Limitations](#limitations).

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

Summary of the `S-1.npy` array used in this project:

| Statistic | Value |
|---|---|
| Signal length | 7,331 time steps |
| Minimum | -1.0 |
| Maximum | ~1.0 |
| Mean | ~ -0.157 |
| Features | 25 |

**Feature structure.** Only feature 0 is nonzero at every time step. Most of the remaining features are sparse, with only a handful of nonzero values across the whole sequence, and several are entirely zero. This is consistent with command or flag features that are mostly inactive. The standard deviation of feature 0 is about 0.46, far larger than that of any other feature.

**Visible anomaly.** The plot of feature 0 shows a clear flat stretch at -1.0 around time steps ~5,360 to ~5,500, followed by a region of rapid swings between the extremes around ~5,500 to ~5,600.

---

## Project Structure

```
.
├── spacecraft-anomaly-detection.ipynb   # Full notebook (data, models, evaluation)
├── data/                                # NASA .npy files and labeled_anomalies.csv (not tracked)
├── images/                              # Saved plots (signal, predictions, comparison chart)
│   └── comparison.png
|   └── isolation-forest-predicted-anomalies.png
|   └── autoencoder-obervation.png
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow / keras (for the autoencoder)

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### Get the data

Download the dataset from the [telemanom repository](https://github.com/khundman/telemanom) (see its README for the data download link), then place the files so the notebook can find them. In Google Colab, upload them to `/content/`:

```python
S1 = np.load("/content/S-1.npy")
```

### Run

Open `spacecraft-anomaly-detection.ipynb` in Jupyter or Google Colab and run the cells in order.

---

## Methodology

### 1. Data exploration

- Loaded `S-1.npy` and inspected its length, range, and mean
- Plotted the telemetry signal over time
- Counted nonzero values and computed the standard deviation per feature to understand which features carry signal

### 2. Isolation Forest (baseline)

Isolation Forest isolates observations by randomly splitting the feature space. Points that are isolated with fewer splits are scored as more anomalous.

- Fit on the telemetry data
- Used a contamination setting so that roughly **2% of readings** are flagged as anomalies
- Plotted predicted anomalies over the telemetry signal

**Observed behavior:** The flagged points concentrate in the high-amplitude region around steps ~5,500-5,600, with one isolated flag near ~5,360. The flat -1.0 stretch is visually the most obvious anomaly but is largely **not** flagged. Isolation Forest scores each point independently of its neighbors, and -1.0 is a value within the observed range, so a flat run is not unusual point by point.

### 3. Dense autoencoder

An autoencoder is trained to reconstruct its input through a narrow bottleneck. It reconstructs normal patterns well and abnormal patterns poorly, so **high reconstruction error indicates a possible anomaly**.

- Built a fully connected (dense) autoencoder
- Trained it on the telemetry data
- Computed the per-sample reconstruction error
- Plotted the errors and flagged samples above a threshold

> This is a **dense** autoencoder, not an LSTM. It does not model temporal dependencies directly.

### 4. Evaluation

Both models were scored with a shared `evaluate_model()` function against the labeled anomaly ranges, reporting **precision**, **recall**, and **F1**. A comparison chart summarizes the results.

---

## Results

Both models were scored on the same S-1 sequence they were trained on.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Isolation Forest | 1.0000	 | 0.1853 | 0.3126 |
| Dense Autoencoder | 0.7493 | 0.6138 | 0.6748 |



![Model comparison](https://github.com/piyushops01/spacecraft-telemetry-anomaly-detection/blob/main/comparision.png)

**Summary of findings**

- The dense autoencoder achieved a higher recall and F1 score than Isolation Forest on this sequence.
- Isolation Forest flagged about 2% of readings, mostly in the high-amplitude anomalous region.
- Both models are sensitive to their thresholding choice (contamination for Isolation Forest, error threshold for the autoencoder), which directly shifts the precision/recall trade-off.

---

## Limitations

Please read these before drawing conclusions from the results.

- **Train and evaluation overlap.** Both models were evaluated on the same sequence used for training. This is not a valid held-out test, and the scores are likely optimistic.
- **Official split not verified.** The official train/test split from the dataset was not verified or used for the reported results.
- **Single channel.** Only channel S-1 was analyzed. The results do not show that either model generalizes to other SMAP or MSL channels.
- **No temporal modeling.** Both models treat samples largely independently, so they can miss anomalies defined by a pattern over time, such as the flat -1.0 segment.
- **Threshold sensitivity.** Reported scores depend on the chosen contamination rate and reconstruction-error threshold.
- **Point-wise metrics.** Anomalies in this dataset are contiguous ranges, and point-wise precision/recall can under- or over-state how well a range was detected.

---

## Future Work

- Train on the official **train** split and evaluate on the official **test** split
- Run across **all SMAP and MSL channels** and report aggregate results
- Add **temporal context** with sliding windows as model input
- Try an **LSTM autoencoder** or LSTM forecaster with dynamic thresholding, as in the original paper
- Use **range-based or event-based evaluation** metrics suited to contiguous anomalies
- Tune thresholds on a validation split rather than on the evaluation data

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


