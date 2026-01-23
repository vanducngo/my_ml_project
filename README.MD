# A Dynamic Integrated Framework for Highly Accurate Real-Time Customer Classification

[![Springer Link](https://img.shields.io/badge/Paper-Springer-orange)](https://link.springer.com/chapter/10.1007/978-981-95-4724-1_6)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📖 Introduction

This repository contains the implementation/demo code for the research paper: **"A Dynamic Integrated Framework for Highly Accurate Real-Time Customer Classification"**.

**Authors:** My-Hai Nguyen, Son-Ha Van, Duc-Ngo Van, Duc-Huy Huynh, and Tuan-Anh Nguyen (University of Information Technology, VNU-HCM, Vietnam).

### The Problem
Traditional customer segmentation methods (like RFM combined with K-Means) provide only a **static snapshot** of the customer base. They fail to capture the temporal dynamics of customer behavior in fast-paced e-commerce environments.

### Our Solution
We propose an end-to-end framework that bridges the gap between static analysis and dynamic action. The system transforms unsupervised clustering results into a high-performance supervised classification system (XGBoost), enabling **instant classification** for every new transaction and handling **concept drift** through periodic retraining.

---

## 🚀 Key Features

*   **Hybrid K-Means Segmentation:** Combines data-driven analysis (Elbow method) with business heuristics to define actionable segments.
*   **Robust Preprocessing:** Handles outliers (IQR), reduces skewness (Box-Cox transformation), and standardizes data (StandardScaler).
*   **High-Accuracy Classifier:** Utilizes **XGBoost** to learn segment definitions, achieving **99.0% accuracy**, outperforming Random Forest and MLP.
*   **Real-Time Operation:** Automatically updates RFM profiles and re-classifies customers upon new transactions.
*   **Lifecycle Management:** Includes strategies for handling new customers, existing customers, and "At-Risk" retention.

---

## 🛠️ Methodology Pipeline

The framework operates in three main phases:

1.  **Phase 1: Foundational Segmentation (Offline)**
    *   Data Cleaning & RFM Calculation.
    *   Outlier Removal & Data Transformation.
    *   Clustering (K-Means) & Labeling (VIP, Loyal, Potential, etc.).
2.  **Phase 2: Classification Model Training**
    *   Training XGBoost on labeled RFM data.
    *   Model Evaluation (F1-Score, Precision, Recall).
3.  **Phase 3: Real-Time Operation**
    *   Live inference pipeline for incoming transactions.
    *   Dynamic segment updating.

---

## 📊 Dataset

The project uses the **Online Retail II** dataset from the UCI Machine Learning Repository.
*   **Transactions:** ~779,425 records.
*   **Customers:** ~5,878 unique profiles.

---

## 📈 Results

### Customer Segments (K=5)
| Cluster | Label | Characteristics |
| :--- | :--- | :--- |
| **0** | **VIP** | High Spending, High Frequency, Recent purchase. |
| **2** | **Loyal** | Moderate Spending, Recent purchase. |
| **3** | **Potential** | High historical spending, but inactive recently. |
| **4** | **New** | Low Frequency, Low Spending, Recent purchase. |
| **1** | **At-Risk** | Long time since last purchase (High Recency). |

### Model Performance
| Model | Accuracy | F1-Score (Macro) |
| :--- | :---: | :---: |
| **XGBoost** | **99.0%** | **0.989** |
| Random Forest | 98.4% | 0.982 |
| MLP (Optimized)| 87.6% | 0.862 |

---