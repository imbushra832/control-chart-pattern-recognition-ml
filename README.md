
# 🧠 Pattern Recognition in Process Control Charts

### *Performance Evaluation of MLP, RBF, and SVM Networks*

---

## 📄 Executive Summary

This project evaluates and compares the performance of three neural network models — **Multilayer Perceptron (MLP)**, **Radial Basis Function (RBF)**, and **Support Vector Machine (SVM)** — for automating **pattern recognition in process control charts**.

The objective is to automate the **runs rules** used in **Statistical Process Control (SPC)** charts based on the X-Chart methodology, classifying whether a process is in statistical control or violates one of six rules.

* **SVM achieved the highest accuracy (89.2%)**, followed by **RBF (88.4%)**, while **MLP performed poorly (14.4%)**.
* Results show that **radial basis function models** (SVM, RBF) are inherently well-suited for recognizing localized patterns in process data.

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Problem Definition](#problem-definition)
3. [Dataset and Preprocessing](#dataset-and-preprocessing)
4. [Model Architectures](#model-architectures)
5. [Training Procedure](#training-procedure)
6. [Results and Analysis](#results-and-analysis)
7. [Discussion](#discussion)
8. [Conclusions](#conclusions)
9. [Recommendations](#recommendations)
10. [Future Work](#future-work)
11. [Project Structure](#project-structure)
12. [Usage Instructions](#usage-instructions)
13. [Dependencies](#dependencies)
14. [References](#references)

---

## 🔬 Introduction

**Statistical Process Control (SPC)** charts are foundational tools in quality assurance, enabling process engineers to detect when a process deviates from stability. Traditionally, SPC relies on manual interpretation by experts.

This project explores the use of **machine learning for automated pattern recognition** in control charts, comparing MLP, RBF, and SVM networks for identifying in-control and out-of-control conditions.

---

## 🧩 Problem Definition

Each model is trained to classify patterns in **X-Charts** into **seven distinct categories**:

| Class | Description                                              |
| ----- | -------------------------------------------------------- |
| 0     | In control (stable process)                              |
| 1     | Rule 1: A point beyond ±3σ                               |
| 2     | Rule 2: Two of three consecutive points beyond ±2σ       |
| 3     | Rule 3: Four of five consecutive points beyond ±1σ       |
| 4     | Rule 4: Seven consecutive points above/below center line |
| 5     | Rule 5: Seven points in continuous upward/downward trend |
| 6     | Rule 6: Nine of ten points within ±1σ band               |

Each pattern consists of **10 consecutive data points (features)**, representing a short sequence of process observations.

---

## 📊 Dataset and Preprocessing

### Data Overview

| Dataset  | Samples | Description                               |
| -------- | ------- | ----------------------------------------- |
| Training | 2,500   | 1,000 in-control + 250 per rule violation |
| Testing  | 250     | 100 in-control + 25 per rule violation    |

### Data Characteristics

* Each input = 10 sequential observations
* Each output = One of 7 categorical classes
* Class imbalance intentionally reflects real-world manufacturing data (in-control states more common)

### Preprocessing Steps

1. **Standardization** using `StandardScaler`
2. **Label encoding** and one-hot transformation as required
3. **Class weighting** in MLP to counter imbalance

---

## ⚙️ Model Architectures

### 1. 🧠 Multilayer Perceptron (MLP)

| Component      | Description                         |
| -------------- | ----------------------------------- |
| Input layer    | 10 neurons                          |
| Hidden layers  | 3 (128, 64, 32 neurons)             |
| Activation     | LeakyReLU                           |
| Regularization | Dropout (0.3, 0.3, 0.2) + L2 (0.01) |
| Optimizer      | Adam (LR=0.0001)                    |
| Loss Function  | Categorical Cross-Entropy           |
| Callbacks      | Early Stopping, LR Scheduler        |

**Observation:** Despite tuning, MLP failed to converge effectively and showed severe overfitting.

---

### 2. 🌐 Radial Basis Function (RBF)

Implemented using **`sklearn.neural_network.MLPClassifier`** to approximate RBF behavior.

| Parameter          | Value       |
| ------------------ | ----------- |
| Hidden layer size  | 150 neurons |
| Activation         | tanh        |
| Regularization (α) | 0.01        |
| Learning rate      | 0.01        |
| Early stopping     | Enabled     |

**Validation Accuracy:** 88.4%

---

### 3. 📈 Support Vector Machine (SVM)

| Parameter          | Value       |
| ------------------ | ----------- |
| Kernel             | RBF         |
| Regularization (C) | 10          |
| Gamma              | 'scale'     |
| Strategy           | One-vs-Rest |

**Test Accuracy:** **89.2%** (best performer)

---

## 🧮 Training Procedure

1. **Grid search** for RBF and SVM hyperparameter optimization
2. **Manual tuning** with callbacks for MLP
3. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
4. **Visualization:** Learning curves, decision boundaries, confusion matrices

---

## 📈 Results and Analysis

### Summary of Model Performance

| Model | Accuracy  | Macro F1 | Best Rule Detection          | Weakest Rule           |
| ----- | --------- | -------- | ---------------------------- | ---------------------- |
| MLP   | 14.4%     | 0.12     | Rule 3 (52% recall)          | In-control (0% recall) |
| RBF   | 88.4%     | 0.87     | Rule 5 (100% recall)         | Rule 3 (64% recall)    |
| SVM   | **89.2%** | **0.88** | Rules 1 & 5 (100% precision) | Rule 3 (64% recall)    |

### Confusion Matrix Highlights

* **SVM:** Minor confusion between Rule 2 and Rule 3; some overlap between in-control and Rule 6.
* **RBF:** Similar confusion but more in-control misclassifications.
* **MLP:** Extensive confusion across all classes, indicating failure to generalize.

---

## 💬 Discussion

### Key Insights

1. **Model Suitability:**
   SVM (RBF kernel) and RBF networks effectively captured localized, nonlinear patterns characteristic of SPC rule violations.
2. **Theoretical Alignment:**
   Radial basis functions excel at recognizing localized structures — ideal for control chart abnormalities.
3. **MLP Limitations:**
   MLP’s architecture was not well-suited for short temporal pattern recognition, even with heavy regularization.
4. **Rule Difficulty:**
   Rule 3 proved consistently difficult, possibly due to its subtle nature or pattern overlap.

### Limitations

* Limited feature representation (raw data only).
* Temporal dependencies not explicitly modeled.
* Class imbalance could be further mitigated through data augmentation.

---

## 🧠 Conclusions

1. **SVM (RBF kernel)** achieved the highest accuracy (89.2%) and overall balanced performance.
2. **RBF networks** performed nearly as well, confirming that radial basis functions are well-suited for this domain.
3. **MLP models** underperformed significantly, suggesting feedforward architectures may be unsuitable for temporal SPC pattern recognition.
4. **Rule 3 violations** remain the hardest to classify, highlighting a potential area for focused model refinement.

---

## 🚀 Recommendations

1. **Model Selection:**
   Use **SVM with RBF kernel** for practical deployment.
2. **Ensemble Strategy:**
   Combine RBF and SVM models to enhance robustness.
3. **Feature Engineering:**
   Incorporate derived statistical features (e.g., variance, skewness, run length, slope).
4. **Advanced Architectures:**
   Explore **LSTM** or **GRU** networks for capturing temporal dependencies.
5. **Rule-Specific Modeling:**
   Develop separate models for challenging violations (e.g., Rule 3).
6. **Data Augmentation:**
   Generate synthetic examples for underrepresented patterns to improve discrimination.

---

## 🔭 Future Work

* Integration of **ensemble learning** to merge SVM and RBF predictions.
* Development of **interpretable ML** models for operator insights.
* Extension to **multivariate control charts** (multiple process variables).
* Exploration of **recurrent and attention-based architectures**.
* Design of **interactive systems** for visual anomaly explanation.

---

## 🗂️ Project Structure

```
├── data/
│   └── control_chart_patterns.csv
├── models/
│   ├── mlp_model.pkl
│   ├── rbf_model.pkl
│   └── svm_model.pkl
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_analysis_visualization.ipynb
├── results/
│   ├── performance_metrics.csv
│   ├── confusion_matrices/
│   └── model_comparison.png
├── src/
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── visualize_results.py
├── README.md
└── requirements.txt
```

---

## ⚡ Usage Instructions

### 4. Evaluate Models


python src/evaluate_models.py


### 5. Visualize Results


python src/visualize_results.py


## 🧩 Dependencies

* Python ≥ 3.8
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* TensorFlow / PyTorch (for MLP)
* Seaborn

Install all with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow


## 🧾 References

1. Montgomery, D. C. (2019). *Introduction to Statistical Quality Control.*
2. Himmel, C. D., & May, G. S. (1993). *Application of Neural Networks for Process Monitoring in Semiconductor Manufacturing.*
3. Chinnam, R. B., Ding, J., & May, G. S. (2000). *Pattern Recognition in Process Control Charts using Neural Networks.*
4. Box, G. E. P., Hunter, W. G., & Hunter, J. S. (1978). *Statistics for Experimenters.*

