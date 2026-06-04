# Bank Customer Churn Classification

## Project Overview

This project builds a binary classification system to predict whether a bank customer will leave the bank.

Target variable:

- `Exited = 0`: customer stays
- `Exited = 1`: customer leaves

The current notebook focuses on:

- exploratory data analysis (EDA);
- automated EDA reporting with `ydata_profiling`;
- preprocessing with `ColumnTransformer` and `Pipeline`;
- class imbalance handling with `RandomOverSampler` inside `ImbPipeline`;
- hyperparameter tuning with `GridSearchCV`;
- model comparison using test-set metrics;
- best-model selection with priority on `Recall`, then `F1-score`;
- feature-importance analysis for the selected model.

## Dataset

Dataset file:

```text
Churn_Modelling.csv
```

Source:

Kaggle: <https://www.kaggle.com/datasets/aakash50897/churn-modellingcsv>

### Dataset size

- Rows: `10,000`
- Columns: `14`
- Target: `Exited`

### Original columns

- `RowNumber`
- `CustomerId`
- `Surname`
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited`

### Data types

- `object`: `Surname`, `Geography`, `Gender`
- `int64`: 9 columns
- `float64`: 2 columns

### Data quality

Notebook output shows:

- no missing values;
- no duplicate rows.

## Current Notebook Workflow

1. Install required libraries in Google Colab.
2. Mount Google Drive and load `Churn_Modelling.csv`.
3. Review schema, descriptive statistics, and sample records.
4. Perform manual EDA with plots.
5. Generate an automated HTML EDA report with `ydata_profiling`.
6. Remove identifier columns.
7. Split features and target.
8. Build preprocessing pipelines.
9. Split train/test with stratification.
10. Use `RandomOverSampler` inside `ImbPipeline`.
11. Run `GridSearchCV` for 4 models.
12. Evaluate tuned models on the test set.
13. Select the best model by `Recall`, then `F1-score`.
14. Plot confusion matrix and ROC curve.
15. Analyze feature importance of the best model.
16. Export final comparison results to CSV.

## Data Loading

The notebook is currently written for Google Colab:

```python
from google.colab import drive, files
drive.mount('/content/drive')
dataset_path = '/content/drive/MyDrive/ML/BTL/Datasets/Churn_Modelling.csv'
df = pd.read_csv(dataset_path)
```

It also uses `files.download(...)` to download generated outputs.

## Exploratory Data Analysis

### Class distribution

Notebook output:

- `Exited = 0`: `7963`
- `Exited = 1`: `2037`

Class percentages:

- non-churn: `79.63%`
- churn: `20.37%`

This is why the notebook prioritizes recall-oriented evaluation.

### EDA visuals currently included

The notebook plots churn relationships for:

- `Exited`
- `Gender`
- `Geography`
- `Age`
- `CreditScore`
- `Balance`
- `NumOfProducts`
- `IsActiveMember`
- `HasCrCard`
- numeric correlation heatmap

### Automated EDA report

The notebook also generates:

```text
bank_customer_churn_eda_report.html
```

using:

```python
profile = ProfileReport(
    df,
    title="Bank Customer Churn - EDA Report",
    explorative=True
)
profile.to_file("bank_customer_churn_eda_report.html")
```

## Preprocessing

### Removed columns

The notebook drops:

- `RowNumber`
- `CustomerId`
- `Surname`

### Features used for modeling

After dropping identifiers:

- `X` shape: `(10000, 10)`
- `y` shape: `(10000,)`

### Feature groups used in the current code

Categorical features:

- `Geography`
- `Gender`
- `HasCrCard`
- `IsActiveMember`

Numeric features:

- `CreditScore`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `EstimatedSalary`

This is an important change from earlier versions: `HasCrCard` and `IsActiveMember` are now treated as categorical features and one-hot encoded.

### Transformer setup

```python
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
```

## Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
```

Notebook output:

- train size: `(8000, 10)`
- test size: `(2000, 10)`
- train class ratio: class 0 = `0.79625`, class 1 = `0.20375`
- test class ratio: class 0 = `0.7965`, class 1 = `0.2035`

## Class Imbalance Handling

The notebook uses this structure for every tuned model:

```text
preprocessor -> RandomOverSampler -> model
```

This keeps oversampling inside the training pipeline and avoids leakage into validation and test data.

## Models Tuned With GridSearchCV

The notebook tunes 4 models:

1. `LogisticRegression`
2. `DecisionTreeClassifier`
3. `RandomForestClassifier`
4. `GradientBoostingClassifier`

Cross-validation:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```

Optimization metric:

```python
scoring="recall"
```

### Best hyperparameters from the notebook output

#### Logistic Regression

- `model__C = 1`
- `model__solver = 'lbfgs'`
- `model__class_weight = None`
- Best CV Recall: `0.6901840490797546`

#### Decision Tree

- `model__max_depth = 5`
- `model__min_samples_split = 2`
- `model__min_samples_leaf = 4`
- `model__class_weight = None`
- Best CV Recall: `0.7539877300613497`

#### Random Forest

- `model__n_estimators = 200`
- `model__max_depth = 5`
- `model__min_samples_split = 2`
- `model__class_weight = None`
- Best CV Recall: `0.7049079754601226`

#### Gradient Boosting

- `model__n_estimators = 100`
- `model__learning_rate = 0.1`
- `model__max_depth = 2`
- Best CV Recall: `0.7398773006134969`

## Final Test Results

The notebook sorts final results by `Recall` descending, then `F1-score` descending.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.803000 | 0.510638 | 0.766585 | 0.612967 | 0.865834 |
| Decision Tree | 0.774000 | 0.466165 | 0.761671 | 0.578358 | 0.847119 |
| Logistic Regression | 0.716000 | 0.391069 | 0.710074 | 0.504363 | 0.778086 |
| Random Forest | 0.793000 | 0.493892 | 0.695332 | 0.577551 | 0.852518 |

## Best Model

Selected model:

- `Gradient Boosting`

Reason:

- highest `Recall`: `0.766585`
- highest `F1-score`: `0.612967`
- highest `ROC-AUC`: `0.865834`

Classification report stored in the notebook:

- class `0`: precision `0.93`, recall `0.81`, f1-score `0.87`
- class `1`: precision `0.51`, recall `0.77`, f1-score `0.61`
- overall accuracy: `0.80`

## Feature Importance

The notebook includes a feature-importance section for the selected best model.

Top features shown in the stored output:

1. `Age` - `0.432889`
2. `NumOfProducts` - `0.316866`
3. `Balance` - `0.067476`
4. `Geography_Germany` - `0.061781`
5. `IsActiveMember_0` - `0.051158`

This is another change from the older README: feature-importance analysis is now part of the notebook workflow.

## Output Files

The current notebook explicitly exports:

```python
final_results_df.to_csv("gridsearch_model_results.csv", index=False)
```

It also downloads:

```python
files.download("gridsearch_model_results.csv")
```

Additional generated file:

- `bank_customer_churn_eda_report.html`

## Libraries Used

Installed in the notebook:

```bash
pip install kaggle ydata-profiling xgboost imbalanced-learn -q
```

Main imported libraries include:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`
- `ydata-profiling`
- `xgboost`

Note: `xgboost` is installed/imported, but the current notebook version does not train an XGBoost model.

## Limitations

- The notebook depends on Google Colab utilities such as `drive.mount()` and `files.download()`.
- It does not include a baseline-training section anymore; it goes directly to `GridSearchCV`.
- `RandomOverSampler` may increase overfitting risk because it duplicates minority samples.
- The classification threshold is still the default `0.5`.

## How to Run Locally

To run outside Colab, replace the loading step with:

```python
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")
```

You should also remove or replace:

```python
from google.colab import drive, files
drive.mount('/content/drive')
files.download(...)
```

## Author

```text
Author : Nguyen Thi Thao My
Project: Bank Customer Churn Classification
```

## License

```text
MIT License
```
