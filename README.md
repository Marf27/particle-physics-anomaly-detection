# ⚛️ Particle Physics Anomaly Detection

Unsupervised anomaly detection in particle physics using **Autoencoders** and **Isolation Forest** on the **LHC Olympics 2020** dataset.

## Overview

This project investigates unsupervised machine learning techniques for identifying anomalous particle-collision events that differ from the Standard Model background.

Two complementary anomaly detection methods are implemented and compared:

* **Fully connected Autoencoder** — learns to reconstruct Standard Model background events and uses the reconstruction error as an anomaly score.
* **Isolation Forest** — identifies anomalous events through recursive random partitioning of the feature space.

The project follows a complete analysis pipeline, from exploratory data analysis and preprocessing to model training, performance evaluation, threshold studies, robustness analysis, and latent-space visualization.

## Dataset

The project uses the **LHC Olympics 2020 high-level feature dataset**, containing reconstructed kinematic variables from particle-collision events.

The dataset is stored locally as:

```text
data/events_anomalydetection_v2.features.h5
```

The dataset contains:

* reconstructed physics observables as input features;
* a `label` column identifying background (`0`) and signal (`1`) events.

The official LHC Olympics 2020 website provides the datasets and additional information about the challenge:

[LHC Olympics 2020](https://lhco2020.github.io/homepage/?utm_source=chatgpt.com)

## Methodology

### 1. Exploratory Data Analysis

The dataset is first investigated through:

* class-distribution analysis;
* missing-value checks;
* statistical summaries;
* feature-distribution comparisons;
* correlation analysis;
* Principal Component Analysis (PCA).

For computational efficiency, dimensionality-reduction visualizations use a random subset of events.

### 2. Data preprocessing

The anomaly detection setup follows an unsupervised approach:

* **Background events** are used for training.
* A fraction of the background is reserved for validation.
* The complete dataset is used as the test set.
* Features are standardized using statistics computed **only from the training background**, preventing data leakage.

The default validation fraction is 15%.

### 3. Autoencoder

A fully connected neural network is trained to reconstruct Standard Model background events.

Architecture:

```text
Input
  ↓
Linear
  ↓
BatchNorm + ReLU
  ↓
Linear
  ↓
ReLU
  ↓
Latent Space
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
BatchNorm + ReLU
  ↓
Output
```

The default latent-space dimensionality is **8**.

The model is trained using:

* Adam optimizer
* Mean Squared Error (MSE) loss
* learning-rate reduction
* early stopping
* best-model checkpointing

The reconstruction error of each event is used as the Autoencoder anomaly score.

### 4. Isolation Forest

An **Isolation Forest** provides a classical machine-learning baseline.

The model is trained exclusively on background events and subsequently evaluated on the complete test dataset.

Default configuration:

```text
n_estimators = 200
contamination = 0.01
```

### 5. Performance evaluation

The two anomaly detectors are compared using:

* ROC curves;
* Area Under the ROC Curve (AUC);
* signal efficiency / True Positive Rate;
* False Positive Rate;
* background rejection.

Several background-percentile thresholds are investigated:

```text
95%
97%
98%
99%
99.5%
```

### 6. Robustness study

The robustness of both methods is investigated by adding progressively stronger Gaussian perturbations to the background test events while keeping the trained models unchanged.

The noise levels studied are:

```text
0.00
0.02
0.05
0.10
0.20
```

The resulting degradation in ROC AUC is used to compare the robustness of the two approaches.

### 7. Latent-space visualization

The Autoencoder latent representation is extracted for the test events.

PCA is then applied to the latent vectors to obtain a two-dimensional representation, allowing the structure of background and signal events to be visually investigated.

## Configuration

The main hyperparameters are defined in a single configuration dictionary:

```python
PARAMS = {
    "val_size": 0.15,

    "latent_dim": 8,
    "batch_size": 128,
    "epochs": 50,
    "lr": 1e-3,
    "patience": 10,

    "n_estimators": 200,
    "contamination": 0.01,

    "percentile": 99,
}
```

This makes it possible to experiment with different model configurations without modifying the rest of the pipeline.

## Running the project

After downloading the dataset and placing it in the expected location:

```text
data/events_anomalydetection_v2.features.h5
```

run the notebook:

```text
Particles.ipynb
```

The notebook contains the complete analysis pipeline, including:

1. Dataset loading
2. Exploratory data analysis
3. Preprocessing
4. Autoencoder construction
5. Autoencoder training
6. Learning-curve analysis
7. Reconstruction-error analysis
8. Isolation Forest
9. ROC/AUC comparison
10. Threshold study
11. Robustness study
12. Latent-space visualization
13. Conclusions

## Hardware Acceleration

PyTorch automatically selects the available device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
```

Therefore, the project can run on:

* NVIDIA GPUs through CUDA;
* Apple Silicon GPUs through MPS;
* CPU-only systems.

## Results

The project compares the anomaly-detection performance of the Autoencoder and Isolation Forest using ROC AUC and threshold-dependent metrics.

The analysis also investigates:

* reconstruction-error distributions;
* anomaly-score thresholds;
* signal efficiency;
* background rejection;
* robustness to perturbations;
* latent-space structure.

The exact numerical results are generated when the notebook is executed with the specified dataset and configuration.

## Reproducibility

Random seeds are fixed where applicable, including:

```text
random_state = 42
```

The dataset is standardized using training-background statistics only, and the same preprocessing transformation is subsequently applied to validation and test events.

## Technologies

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **SciPy / scikit-learn**
* **PyTorch**
* **PCA**
* **HDF5**

## References

The dataset and challenge are provided by the **LHC Olympics 2020** collaboration.

* LHC Olympics 2020: https://lhco2020.github.io/homepage/
* LHC Olympics 2020 R&D dataset: https://lhco2020.github.io/homepage/RnD.html

## Author

**Mathias Rendón Fernández**

Physics student — Universidad Autónoma de Madrid
