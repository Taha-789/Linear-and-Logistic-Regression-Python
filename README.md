# Regression Models from Scratch using NumPy

## Overview
This project implements Linear Regression and Logistic Regression from scratch using NumPy. The objective is to understand the internal workings of machine learning algorithms without relying on high-level libraries.

The project includes gradient descent optimization, evaluation metrics, cross-validation, and comparison with scikit-learn implementations.

---

## Objectives
- Implement Linear Regression using gradient descent
- Implement Logistic Regression for binary classification
- Understand loss functions and optimization
- Evaluate model performance using standard metrics
- Compare results with scikit-learn models

---

## Datasets

### 1. Boston Housing Dataset (Linear Regression)
- 506 samples
- 13 features
- Target: house prices

Loaded using:
```
from Fetch_openMl
```

---

### 2. Breast Cancer Wisconsin Dataset (Logistic Regression)
Binary classification dataset.

Download from:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

---

## Installation

Install required dependencies:

```
pip install numpy matplotlib scikit-learn
```

---

## How to Run

### Linear Regression
```
python linear_regression.py
```

### Logistic Regression
```
python logistic_regression.py
```

---

## Implementation Details

### Linear Regression
- Model: Y = wx + b
- Loss Function: Mean Squared Error
- Optimization: Gradient Descent
- Features:
  - Dataset splitting (train/test)
  - Learning rate experiments
  - Convergence based on parameter updates
  - Training loss visualization

---

### Logistic Regression
- Model: Sigmoid(wx + b)
- Loss Function: Binary Cross-Entropy
- Optimization: Gradient Descent
- Features:
  - Sigmoid function implementation
  - Binary classification
  - Threshold tuning
  - Evaluation metrics (Accuracy, Precision, Recall, F1-score)
  - k-fold cross-validation

---

## Evaluation Metrics

### Linear Regression
- Mean Squared Error (MSE)

### Logistic Regression
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Experiments

### Learning Rate Analysis
Tested multiple learning rates to observe:
- Convergence speed
- Stability of training
- Risk of divergence

---

### Threshold Tuning
Tested thresholds:
- 0.3, 0.4, 0.5, 0.6, 0.7

Observed trade-offs between:
- Sensitivity (Recall)
- Specificity

---

### Cross-Validation
- k-fold cross-validation (k = 5)
- Average performance metrics reported

---

## Results
- Linear Regression:
  - Training loss convergence observed
  - Performance compared with scikit-learn

- Logistic Regression:
  - Competitive performance with scikit-learn model
  - Metrics vary based on threshold selection

---
## Notes
- All models are implemented from scratch using NumPy
- scikit-learn is only used for dataset loading and comparison
- Plots are generated using matplotlib

---

## Conclusion
This project demonstrates a complete implementation of regression models from scratch, providing a deeper understanding of optimization, loss functions, and model evaluation.

The comparison with scikit-learn validates the correctness and efficiency of the implementations.
