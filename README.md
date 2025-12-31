# 📊 PCA-using-Co-Median
## Use Median instead of Mean in Principal-Component-Analysis

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

---

## 📝 Overview

In traditional **Principal Component Analysis (PCA)**, the **Mean** is used as the statistical measure to calculate covariance, decompose Eigenvalues, and Eigenvectors. However, the mean is highly sensitive to outliers.

**This project rewrites the entire PCA process using the Median.** By employing the **Co-Median**, we aim to achieve a robust dimensionality reduction that maintains data integrity even in the presence of significant noise or outliers.

### 🎯 Objectives

1.  **Skewness Analysis**: Determine how much traditional PCA skews the dataset due to outliers compared to the Median-based approach.
2.  **Distance Comparison**: Compare the projected points as a squareform **Pairwise Distance Matrix**.
3.  **Error Quantification**: Use the **Frobenius Norm** to calculate the error between projected points obtained from Mean-based PCA and Median-based PCA.

---

## 🧠 What is COMAD PCA?

**COMAD (Co-Median Absolute Deviation)** PCA is a robust alternative to standard PCA.

*   **Standard PCA**: Relies on the Covariance Matrix, which minimizes the $L_2$ norm (Least Squares). This squares the error, causing outliers to disproportionately influence the principal components.
*   **Co-Median PCA**: Replaces the covariance calculation with the **Co-Median**. For two vectors $X$ and $Y$, the co-median is often defined as the median of the products of their deviations from their respective medians:
    $$ \text{CoMed}(X, Y) = \text{median}((X - \text{median}(X))(Y - \text{median}(Y))) $$
    This approach (or variations using Median Absolute Deviation) ensures that extreme values do not pull the principal components away from the main data cluster.

**Inspired by:**
1.  [On the estimation of the copula function](https://www.ism.ac.jp/editsec/aism/pdf/049_4_0615.pdf)
2.  [Robust Principal Component Analysis based on Co-Median](https://link.springer.com/chapter/10.1007/978-3-030-32047-8_24)

---

## 📂 Data Analysis

### 🔢 Data Types Compared
*   **Synthetic Data**: Controlled datasets generated with specific distributions (e.g., Multivariate Gaussian) to test theoretical limits.
*   **Outlier Contamination**: Datasets injected with "salt and pepper" noise or extreme values to simulate real-world sensor errors or anomalies.

### 📐 2D vs 3D Visualization
*   **2D Projections**: Scatter plots of the first two Principal Components (PC1 vs PC2) to visualize cluster separation and the impact of outliers on the primary axes.
*   **3D Projections**: 3D scatter plots to understand the manifold structure and the rotation of the basis vectors in higher-dimensional space.

### 💾 Datasets
*   **Synthetic Gaussian**: A generated dataset representing a standard normal distribution, used as a control group.
*   **Contaminated Gaussian**: The same synthetic dataset with a percentage of points moved to extreme values to test robustness.
*   **Benchmark Datasets**: (e.g., Iris, Wine) Standard machine learning datasets used to verify that the Co-Median approach preserves known class structures.

---

## 🛠️ Tech Stack

*   **Python**: The core programming language.
*   **NumPy**: For high-performance vector and matrix operations.
*   **Pandas**: For data frame manipulation and analysis.
*   **Scikit-Learn**: For standard PCA implementation and dataset utilities.
*   **Matplotlib**: For generating 2D and 3D visualizations.
*   **Jupyter Notebooks**: For interactive coding and result presentation.
