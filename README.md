# 🏦 Bank Customer Churn Classification

## 📌 Project Overview

This project builds a **binary classification model** to predict whether a bank customer is likely to leave the bank (customer churn).

The target variable is:

- `Exited = 0` → Customer stays with the bank
- `Exited = 1` → Customer leaves the bank

The primary business objective is to identify customers at risk of churn early so that the bank can implement retention strategies and reduce customer loss.

The notebook covers:

- Exploratory Data Analysis (EDA)
- Data preprocessing with Scikit-Learn Pipelines
- Class imbalance handling using RandomOverSampler
- Hyperparameter tuning with GridSearchCV
- Model evaluation and comparison
- Selection of the best model based primarily on Recall

---

# 📂 Dataset

Dataset file:

```text
Churn_Modelling.csv
```

Dataset source:

Kaggle: <https://www.kaggle.com/datasets/aakash50897/churn-modellingcsv>

## Dataset Size

| Attribute | Value |
|------------|---------|
| Rows | 10,000 |
| Columns | 14 |
| Target Variable | Exited |

## Original Features

- RowNumber
- CustomerId
- Surname
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited

## Data Types

- Object columns: Surname, Geography, Gender
- Integer columns: 9
- Float columns: 2

## Data Quality

The dataset contains:

- No missing values
- No duplicate rows

---

# 🎯 Project Objectives

The notebook follows this workflow:

1. Understand the data through EDA.
2. Build preprocessing pipelines.
3. Handle class imbalance.
4. Optimize hyperparameters using GridSearchCV.
5. Compare optimized models.
6. Select the best model based on Recall.

---

# ⚙️ Machine Learning Pipeline

```text
Dataset
   ↓
EDA
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
ColumnTransformer
   ↓
RandomOverSampler
   ↓
GridSearchCV
   ↓
Model Evaluation
   ↓
Best Model Selection
```

---

# 📊 Exploratory Data Analysis (EDA)

## Missing Values

```python
df.isnull().sum()
```

Result:

- No missing values found.

## Duplicate Rows

```python
df.duplicated().sum()
```

Result:

- No duplicate records.

## Target Variable Analysis

Class distribution:

| Class | Samples |
|---------|---------:|
| Exited = 0 | 7963 |
| Exited = 1 | 2037 |

Class percentages:

- Non-churn: 79.63%
- Churn: 20.37%

This indicates a significant class imbalance.

## Gender Analysis

Churn rate:

- Female: 25.07%
- Male: 16.46%

## Geography Analysis

Churn rate:

- Germany: 32.44%
- Spain: 16.67%
- France: 16.15%

## Age Analysis

Age distributions are visualized to compare churned and non-churned customers.

## Credit Score Analysis

Credit score distributions are examined to identify potential relationships with churn.

## Balance Analysis

Customer balances are analyzed to determine whether account balance affects churn probability.

## Number of Products

Churn rate by product count:

| Products | Churn Rate |
|----------|------------|
| 4 | 100.00% |
| 3 | 82.71% |
| 1 | 27.71% |
| 2 | 7.58% |

## Activity Status

Churn rate:

- Inactive customers: 26.85%
- Active customers: 14.27%

## Credit Card Ownership

Churn rate:

- Without credit card: 20.81%
- With credit card: 20.18%

## Correlation Analysis

A correlation heatmap is generated to inspect relationships among numerical variables and the target.

---

# 🧹 Data Preprocessing

## Removing Unnecessary Features

The following columns are removed:

- RowNumber
- CustomerId
- Surname

Reason:

- Identifier columns provide little predictive value.
- They may introduce noise.

## Feature and Target Split

```python
X = df_model.drop(columns=["Exited"])
y = df_model["Exited"]
```

## Feature Types

### Categorical Features

- Geography
- Gender

### Numerical Features

- CreditScore
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

---

# 🔧 Feature Transformation

## Numerical Features

```python
StandardScaler()
```

Used to standardize numerical variables.

## Categorical Features

```python
OneHotEncoder(handle_unknown="ignore")
```

Used to encode categorical variables safely.

## ColumnTransformer

```python
ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
```

---

# ✂️ Train/Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Benefits:

- 80% training data
- 20% testing data
- Preserves class distribution

---

# ⚖️ Handling Class Imbalance

## Why Oversampling?

The churn class represents only about 20% of the dataset.

Without balancing, models may favor the majority class.

## RandomOverSampler

Pipeline structure:

```text
Preprocessor
      ↓
RandomOverSampler
      ↓
Classifier
```

Class distribution before oversampling:

- Class 0: 6370
- Class 1: 1630

After oversampling:

- Class 0: 6370
- Class 1: 6370

---

# 🤖 Models Evaluated

The notebook evaluates four classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier

---

# 🔍 Hyperparameter Tuning

## Cross Validation

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

## Optimization Metric

```python
scoring="recall"
```

Recall is prioritized because missing churn customers is more costly than generating some false positives.

---

# 📈 Best Hyperparameters

## Logistic Regression

Best Recall (CV):

```text
0.688957
```

## Decision Tree

Best Recall (CV):

```text
0.753374
```

## Random Forest

Best Recall (CV):

```text
0.728834
```

## Gradient Boosting

Best Recall (CV):

```text
0.739877
```

---

# 🏆 Final Test Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8030 | 0.510638 | 0.766585 | 0.612967 | 0.865834 |
| Decision Tree | 0.7740 | 0.466165 | 0.761671 | 0.578358 | 0.847372 |
| Random Forest | 0.7875 | 0.485669 | 0.749386 | 0.589372 | 0.854839 |
| Logistic Regression | 0.7160 | 0.391069 | 0.710074 | 0.504363 | 0.778103 |

---

# 🥇 Best Model

## Gradient Boosting Classifier

Reasons:

- Highest Recall: 0.766585
- Highest F1-score: 0.612967
- Highest ROC-AUC: 0.865834

Classification Report:

### Class 0

- Precision: 0.93
- Recall: 0.81
- F1-score: 0.87

### Class 1

- Precision: 0.51
- Recall: 0.77
- F1-score: 0.61

Overall Accuracy:

```text
80.3%
```

---

# 📁 Output Files

The notebook exports:

```python
final_results_df.to_csv(
    "gridsearch_model_results.csv",
    index=False
)
```

---

# 💡 Advantages

- Comprehensive EDA
- Proper preprocessing pipeline
- Handles class imbalance correctly
- Prevents data leakage
- Uses Stratified Cross Validation
- Hyperparameter optimization
- Multiple model comparison
- Exportable results

---

# ⚠️ Limitations

## Google Colab Dependency

Some code is specific to Google Colab:

```python
drive.mount()
files.download()
```

## Default Classification Threshold

The model uses the default threshold of 0.5.

Threshold tuning may further improve business performance.

## RandomOverSampler Risk

RandomOverSampler duplicates minority samples and may increase overfitting risk.

Advanced techniques such as SMOTE could be explored.

---

# 🚀 How to Run

## Google Colab

1. Upload the notebook.
2. Upload or mount `Churn_Modelling.csv`.
3. Verify dataset path.
4. Run all cells.

## Local Environment

Replace the loading code with:

```python
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")
```

Remove:

```python
drive.mount()
files.download()
```

---

# 📦 Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost kaggle ydata-profiling
```

---

# 📋 Project Workflow Summary

1. Load dataset
2. Check data quality
3. Perform EDA
4. Remove unnecessary columns
5. Split X and y
6. Apply preprocessing
7. Train/test split
8. Apply RandomOverSampler
9. Tune models using GridSearchCV
10. Evaluate models
11. Select Gradient Boosting
12. Export results

---

# 🔮 Future Improvements

Potential improvements:

- Threshold optimization
- SHAP explainability
- SMOTE and SMOTENC
- XGBoost optimization
- Ensemble methods
- Business cost-based evaluation
- Deployment using Streamlit or Flask

---

# 👨‍💻 Author

```text
Author : Nguyen Thi Thao My
Project: Bank Customer Churn Classification
```

---

# 📜 License

```text
MIT License
```
