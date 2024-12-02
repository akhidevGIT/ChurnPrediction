# Bank Customer Churn Prediction

## 📜 Project Overview
This project builds an end-to-end machine learning pipeline to predict customer churn (Exited = 1). The goal is to identify patterns in customer data to help businesses improve retention strategies. A balanced evaluation using various algorithms ensures actionable insights into churn prediction.

---

## 📊 Steps in the Workflow
1. **EDA (Exploratory Data Analysis)**:
   - Analyzed distributions and relationships among features.
   - Identified imbalanced target classes (Exited = 1 significantly underrepresented).

2. **Data Preprocessing**:
   - **Feature Engineering**: Handled categorical features using encoding techniques.
   - **Scaling**: Standardized numerical features for uniform input.
   - **SMOTE (Synthetic Minority Oversampling Technique)**: Applied to address target class imbalance and improve model learning for the minority class (Exited = 1).

3. **Model Training**:
   - Models used: **Logistic Regression**, **Support Vector Machines (SVM)**, **Random Forest**, and **XGBoost**.
   - Fine-tuned hyperparameters for optimal performance using grid search.

4. **Performance Evaluation**:
   - Metrics used: **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
   - Focused on **Recall** for churn prediction (Exited = 1) to minimize false negatives.

---

## 🔍 Results
| Model                 | Accuracy | Precision (Exited=1) | Recall (Exited=1) | F1-Score (Exited=1) |
|------------------------|----------|-----------------------|--------------------|----------------------|
| Logistic Regression    | 0.7145   | 0.39                  | 0.72               | 0.51                 |
| Support Vector Machine | 0.8035   | 0.52                  | 0.57               | 0.54                 |
| **Random Forest**      | 0.8405   | 0.61                  | 0.60               | 0.60                 |
| **XGBoost**            | 0.8535   | 0.67                  | 0.55               | 0.61                 |

### Key Insights:
- **Random Forest** emerged as the best model based on overall metrics, balancing performance for both classes.
- **XGBoost** slightly outperformed Random Forest in **accuracy** and **F1-Score** for the churn class (Exited = 1).

---

## 🛠️ Tools and Technologies
- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, SMOTE, XGBoost
- **Modeling Frameworks**: Logistic Regression, SVM, Random Forest, XGBoost

---

## 📌 Key Learnings
- The importance of addressing data imbalance (via SMOTE) in churn prediction to improve minority class recall.
- Random Forest and XGBoost deliver strong performance for tabular data with categorical and numerical features.
- Class imbalance challenges persist even after SMOTE, highlighting the need for tailored strategies or domain-specific features.

---

## 🚧 Future Enhancements
1. Deploy the model as a web application for real-time churn prediction.
2. Experiment with advanced balancing techniques like **SMOTE-ENN** or **ADASYN**.
3. Perform feature importance analysis to identify key drivers of churn.
4. Compare other ensemble algorithms like **CatBoost** and **LightGBM** for additional insights.
