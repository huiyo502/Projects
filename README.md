# 🔬 Research Projects Overview

This document provides a overview of my research projects focused on machine learning

---

## 1. Pharmacokinetics (PK) Prediction (Regression)

* **Aim:** To build Machine Learning (ML) models capable for predicting drug concentration over time for COVID-19 disease with small and high uncertaintiesclinical data.
* **Methodology:** We explored the predictive capacity of various ML algorithms for drug pharmacokinetics. Based on simulated data, we trained models including **Linear Regression**, **Gradient Boosting Machines (GBM)**, **XGBoost**, and **Random Forest** to predict the plasma concentration-time series and the rifampicin **Area Under the Concentration-versus-Time Curve from 0–24 h ($\text{AUC}_{0–24h}$)** after repeated dosing.

---

## 2. Time-to-Event Prediction (Classification)

* **Aim:** To develop Machine Learning models for predicting whether a patient will recover, effectively handling imbalanced categorical data in clinical outcomes.
* **Methodology:** We integrated traditional **Statistical Pharmacodynamic (PD) models** with ML classifiers, including **Extreme Gradient Boosting (XGBoost)**, **Random Forest (RF)**, **C-Support Vector Classification (SVC)**, and the **K-Nearest Neighbours (KNN)** algorithm. Different techniques were explored to enhance model performance, notably resampling strategies, fine-tuned parameter tuning, and adjusting classifier decision thresholds.

---

## 3. Time Series Forecasting

* **Aim:** To explore the use of machine learning and statistical models for forecasting individual-level treatment outcomes in clinical Phase 2b trials from Phase 2a.
* **Methodology:** We tested a range of machine learning and statistical approaches to forecast biomarker responses across different treatment arms. These included classical time-series models such as **ARIMA** and **ARIMAX**, alongside advanced ML algorithms like **XGBoost**, **Gaussian Process**, and **Random Forest**.

---

## 4. Molecular optimization
* **Aim:**  To explore the ability of machine learning model for generating molecules satisfying the desirable properties.
* **Methodology:** Two main models are employed: the classic Sequence-to-Sequence model with Attention Mechanism and the state-of-the-art Transformer model. To allow project-specific optimization, user-specified desirable property changes (e.g., changes in $\text{logD}$, solubility, or clearance) are incorporated into the input sequence as an additional conditional input, guiding the model to generate molecules satisfying these criteria.

---
## 4. Main challenge in my projects:

1. Modify the model structure
2. Data preparation:
   1. small data
   2. imbalance data
   3. missing information
   4. outlier
