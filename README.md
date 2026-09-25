# Breast Cancer Classification: Custom PCA & Comparative Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9899-brightgreen?style=for-the-badge)](#)
[![Accuracy](https://img.shields.io/badge/Accuracy-95.61%25-success?style=for-the-badge)](#)

This project performs dimensionality reduction using **Principal Component Analysis (PCA)** on the **Breast Cancer Wisconsin** dataset and compares two modeling approaches: a Manual Linear Probability Model + Sigmoid vs. Scikit-Learn Logistic Regression.

---

## 📌 Methodology

1. **Standardization:** Features were standardized using `StandardScaler`.
2. **Covariance & Eigen Analysis:** Eigenvalues and eigenvectors were extracted from the $30 \times 30$ covariance matrix using `np.linalg.eig`.
3. **PCA Component Selection:** The top 2 principal components ($PC_0, PC_1$) with the highest correlation to the target feature were selected.
4. **Manual Approach:** Coefficients were calculated via the Normal Equation $\hat{\beta} = (A^T A)^{-1} A^T b$ and passed through the Sigmoid activation function.
5. **Scikit-Learn Approach:** An $L_1$-regularized `LogisticRegression` model was trained.
6. **Threshold Optimization:** The optimal decision threshold was identified using Youden's J statistic.

---

## 📊 PCA & Model Comparison

### Selected Principal Components

| Component | Absolute Correlation (`corr`) | Status |
| :---: | :---: | :--- |
| **`PC_0`** | **0.7855**| Selected (Highest eigenvalue & correlation) |
| **`PC_1`** | **0.1884**| Selected (Second highest eigenvalue & correlation) |
| **total** | **0.9739** | |

### Approach Comparison

| Feature | Manual Approach (OLS + Sigmoid) | Scikit-Learn (`LogisticRegression`) |
| :--- | :--- | :--- |
| **Logic** | Linear Probability Model + Sigmoid | Maximum Likelihood Estimation (MLE) |
| **Solution** | Closed-Form Analytical Solution | `liblinear` ($L_1$ Regularization) |
| **Coefficients** | $\beta_0 = 0.6274$, $\beta_1 = -0.1042$, $\beta_2 = 0.0382$ | $L_1$ Optimized Coefficients |

---
## Roc-Curve from Logistic Regression

<img width="711" height="572" alt="roc_auc" src="https://github.com/user-attachments/assets/ba943cc2-f033-4ac3-b668-a4a473e1b549" />

---

## 📈 Performance & Evaluation

### Model Metrics (Threshold = 0.5587)

| Metric | Value | Description |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.9899** | Excellent classification discriminative power. |
| **Accuracy** | **0.9561 (95.61%)** | Overall correct prediction rate. |
| **TPR (Sensitivity)** | **0.9608 (96.08%)** | True cancer detection rate. |
| **FPR (1 - Specificity)** | **0.0472 (4.72%)** | False alarm rate. |

### Confusion Matrix included
- From business logic perspective, the mest measures are:
    - TPR is 0.96, high enough!
    - FNR is 0.04. low enough!
