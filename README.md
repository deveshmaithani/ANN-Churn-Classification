# 🧠 Customer Churn Prediction using ANN

An end-to-end **Deep Learning project** that predicts whether a bank customer is likely to churn using an **Artificial Neural Network (ANN)**.  
The project includes **model training, evaluation, and a Streamlit web application** for real-time predictions.

---

## 📌 Project Overview

Customer churn is a major challenge in the banking sector. Retaining existing customers is often more cost-effective than acquiring new ones.  
This project aims to **identify customers at high risk of churn**, enabling proactive retention strategies.

---

## 🧠 Model Architecture

The ANN model is designed to capture non-linear relationships in customer behavior.



- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Regularization:** Early Stopping  
- **Output:** Churn Probability  

---

## 📊 Model Performance

| Metric | Value |
|------|------|
| Accuracy | ~87% |
| ROC-AUC | **0.8568** |
| Precision (Churn) | 0.77 |
| Recall (Churn) | 0.45 |

📌 ROC-AUC was prioritized due to **class imbalance**, making the model more reliable than accuracy alone.

---

## 📈 Evaluation Techniques

- Confusion Matrix
- Classification Report
- ROC Curve
- Precision–Recall Curve

These metrics help balance business trade-offs between **false positives and false negatives**.

---

## 🖥️ Streamlit Web Application

An interactive Streamlit app allows users to:
- Enter customer details
- Predict churn probability
- Get business-friendly output:
  - **“The customer is likely to churn.”**
  - **“The customer is not likely to churn.”**

The app uses the **same scaler and ANN model** used during training to ensure consistency.

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Data Processing:** NumPy, Pandas  
- **Machine Learning:** Scikit-learn  
- **Deep Learning:** TensorFlow / Keras  
- **Visualization:** Matplotlib, Seaborn  
- **Web App:** Streamlit  



