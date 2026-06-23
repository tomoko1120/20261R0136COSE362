# 20261R0136COSE362 — AI + Climate Resilience

## Heavy Rainfall Prediction Using VAE-Based Data Augmentation

This repository contains the final project code for **COSE362 Machine Learning, Spring 2026** at Korea University.

Our project focuses on **AI + Climate Resilience**, specifically improving heavy rainfall prediction under rare-event class imbalance. Heavy rainfall events occur much less frequently than normal weather days, so standard machine learning classifiers may achieve high overall accuracy while still missing dangerous heavy-rain cases.

To address this problem, we use a **Variational Autoencoder (VAE)** as a generative data augmentation framework. The VAE is trained on heavy-rain samples and generates synthetic minority-class samples. These generated samples are added to the training data, and we compare baseline classifiers with VAE-augmented classifiers.

## Team Members

| Name                 | Student ID |
| -------------------- | ---------- |
| Narantuya Batsaikhan | 2024320224 |
| Jeonhyun Cha         | 2026951260 |
| Prasai Sashwat       | 2025952744 |
| Alirshad             | 2024320013 |
| Ishraq Ahmed         | 2022320175 |

## Main Submission Notebook

The main notebook for final evaluation is:

```text
implementation.ipynb
```

This notebook contains the complete project pipeline:

1. Dataset loading
2. Feature selection and preprocessing
3. HeavyRainTomorrow target label creation
4. Train/test split and standardization
5. VAE training on heavy-rain samples
6. Synthetic heavy-rain sample generation
7. Logistic Regression and Random Forest classification
8. Evaluation using recall, precision, and F1-score

## Repository Structure

```text
20261R0136COSE362/
│
├── README.md
├── dataset.csv
├── implementation.ipynb
├── Data_collection_and_preprocessing.ipynb
│
└── experiments/
    ├── README.md
    ├── experiment_(with_GMM).ipynb
    └── vae_fc3_with_PReLU.ipynb
```

## Dataset

The project uses `dataset.csv`, based on the weatherAUS dataset.
The selected features include temperature, rainfall, humidity, pressure, wind speed, and cloud cover.

The target variable is:

```text
HeavyRainTomorrow
```

Heavy rainfall is defined using a top-percentile rainfall threshold. If tomorrow’s rainfall exceeds the threshold, the sample is labeled as a heavy-rain event.

## Method Summary

The project follows this pipeline:

```text
Weather Data
→ Preprocessing
→ HeavyRainTomorrow Label Creation
→ VAE Training on Heavy-Rain Samples
→ Synthetic Heavy-Rain Data Generation
→ Augmented Training Dataset
→ Classifier Training
→ Evaluation
```

Two classifiers are used for comparison:

* Logistic Regression
* Random Forest

Each classifier is tested in two settings:

1. Baseline: trained only on the original data
2. VAE-Augmented: trained on original data plus VAE-generated heavy-rain samples

## Evaluation Metrics

Because heavy rainfall is a rare-event prediction problem, standard accuracy is not the main focus. We evaluate the models using:

* Recall
* Precision
* F1-score

Recall is especially important because missing an actual heavy-rainfall event can be dangerous in real disaster-warning scenarios.

## How to Run

1. Open `implementation.ipynb` in Google Colab or Jupyter Notebook.
2. Make sure `dataset.csv` is in the same directory as the notebook.
3. Run all cells from top to bottom.
4. The notebook will train the baseline and VAE-augmented models and display the evaluation results.

## Additional Experiments

The `experiments/` folder contains supporting experiment notebooks, including activation-function testing, PReLU-based VAE experiments, and GMM-related experiments. These files are included for reference and additional analysis, but the official final pipeline is provided in `implementation.ipynb`.

## Conclusion

This project investigates whether generative data augmentation can improve rare heavy-rainfall prediction. The results show that VAE-generated samples can provide useful insight into minority-class learning, especially when combined with nonlinear classifiers such as Random Forest. However, the improvement is limited, suggesting that future work should explore stronger generative models, better feature engineering, and more robust evaluation methods such as PR-AUC.
