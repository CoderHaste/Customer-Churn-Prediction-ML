# 📉 Customer Churn Prediction using Machine Learning

## 📌 Overview

This project predicts customer churn using machine learning models and compares their performance.
Customer churn is a critical problem in telecom and subscription-based businesses, where retaining customers is more cost-effective than acquiring new ones.

This project applies SMOTE to handle class imbalance and evaluates multiple models to identify the best-performing algorithm.

---

## 🚀 Features

* Data preprocessing and cleaning
* Handling missing values
* Label encoding for categorical features
* Feature scaling using StandardScaler
* Handling imbalanced data using SMOTE
* Model training and comparison

Models used:

* Random Forest
* XGBoost
* LightGBM

---

## 📊 Model Performance

Evaluation metrics used:

* Precision
* Recall
* F1-score
* ROC-AUC score

LightGBM achieved the highest ROC-AUC score, while Random Forest provided strong recall for churn prediction.

---

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn
* XGBoost
* LightGBM
* Imbalanced-learn (SMOTE)

---

## 📂 Project Structure

```
Customer-Churn-Prediction-ML
│── Telco-Customer-Churn.csv
│── churn_prediction.py
│── requirements.txt
│── README.md
```

---

## ▶️ How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run:

```
python churn_prediction.py
```

---

## 🎯 Learning Outcomes

* Handling imbalanced datasets
* Comparing ensemble & boosting models
* Evaluating ML models using ROC-AUC
* End-to-end ML workflow

---

## 👨‍💻 Author

Prateek Manjunath</br>
Machine Learning & Data Science Enthusiast
