# README - Bank Customer Churn Classification

## 1. Giới thiệu

Notebook [Bank_Customer_Churn_Classification.ipynb](/D:/University/Machine%20learning/BTL/Bank_Customer_Churn_Classification/Bank_Customer_Churn_Classification.ipynb) xây dựng bài toán phân loại nhị phân để dự đoán khách hàng có rời bỏ ngân hàng hay không.

Biến mục tiêu:

- `Exited = 0`: khách hàng ở lại
- `Exited = 1`: khách hàng rời bỏ

Phiên bản notebook hiện tại tập trung vào:

- EDA thủ công bằng biểu đồ;
- tạo báo cáo EDA tự động bằng `ydata_profiling`;
- tiền xử lý bằng `ColumnTransformer` và `Pipeline`;
- xử lý mất cân bằng bằng `RandomOverSampler` trong `ImbPipeline`;
- tối ưu siêu tham số bằng `GridSearchCV`;
- so sánh mô hình theo các chỉ số trên tập test;
- chọn mô hình tốt nhất theo ưu tiên `Recall`, sau đó `F1-score`;
- phân tích `Feature Importance` của mô hình tốt nhất.

## 2. Dữ liệu sử dụng

Notebook dùng file `Churn_Modelling.csv`.

Nguồn dataset:

Kaggle: <https://www.kaggle.com/datasets/aakash50897/churn-modellingcsv>

### 2.1. Quy mô dữ liệu

Theo output đang lưu trong notebook:

- số dòng: `10,000`
- số cột: `14`
- biến mục tiêu: `Exited`

### 2.2. Danh sách cột gốc

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

### 2.3. Kiểu dữ liệu

Notebook xác nhận:

- `object`: `Surname`, `Geography`, `Gender`
- `int64`: 9 cột
- `float64`: 2 cột

### 2.4. Chất lượng dữ liệu

Kết quả trong notebook:

- không có giá trị thiếu;
- không có dòng trùng lặp.

## 3. Quy trình notebook hiện tại

1. Cài thư viện trong Google Colab.
2. Mount Google Drive và đọc file CSV.
3. Xem cấu trúc dữ liệu, thống kê mô tả và một số dòng đầu.
4. Thực hiện EDA bằng biểu đồ.
5. Sinh báo cáo HTML tự động bằng `ydata_profiling`.
6. Loại bỏ các cột định danh.
7. Tách `X` và `y`.
8. Xây pipeline tiền xử lý.
9. Chia train/test với `stratify`.
10. Đưa `RandomOverSampler` vào `ImbPipeline`.
11. Chạy `GridSearchCV` cho 4 mô hình.
12. Đánh giá các mô hình tuned trên tập test.
13. Chọn mô hình tốt nhất theo `Recall`, rồi `F1-score`.
14. Vẽ `Confusion Matrix` và `ROC Curve`.
15. Phân tích `Feature Importance` của mô hình tốt nhất.
16. Xuất bảng kết quả cuối ra CSV.

## 4. Cách tải dữ liệu trong notebook

Notebook hiện được viết theo ngữ cảnh Google Colab:

```python
from google.colab import drive, files
drive.mount('/content/drive')
dataset_path = '/content/drive/MyDrive/ML/BTL/Datasets/Churn_Modelling.csv'
df = pd.read_csv(dataset_path)
```

Ngoài ra notebook còn dùng `files.download(...)` để tải các file kết quả xuống máy.

## 5. Phân tích khám phá dữ liệu (EDA)

### 5.1. Phân bố biến mục tiêu

Kết quả đếm nhãn trong notebook:

- `Exited = 0`: `7963`
- `Exited = 1`: `2037`

Tỷ lệ phần trăm:

- không churn: `79.63%`
- churn: `20.37%`

Điều này giải thích vì sao notebook ưu tiên `Recall` khi tối ưu mô hình.

### 5.2. Các biểu đồ EDA hiện có

Notebook hiện trực quan hóa mối quan hệ churn với các biến:

- `Exited`
- `Gender`
- `Geography`
- `Age`
- `CreditScore`
- `Balance`
- `NumOfProducts`
- `IsActiveMember`
- `HasCrCard`
- ma trận tương quan giữa các biến số

### 5.3. Báo cáo EDA tự động

Notebook có thêm phần tạo báo cáo HTML:

```text
bank_customer_churn_eda_report.html
```

bằng đoạn code:

```python
profile = ProfileReport(
    df,
    title="Bank Customer Churn - EDA Report",
    explorative=True
)
profile.to_file("bank_customer_churn_eda_report.html")
```

## 6. Tiền xử lý dữ liệu

### 6.1. Loại bỏ cột không cần thiết

Notebook xóa 3 cột:

- `RowNumber`
- `CustomerId`
- `Surname`

### 6.2. Tách biến đầu vào và biến mục tiêu

Sau khi bỏ cột định danh:

- `X` có kích thước `(10000, 10)`
- `y` có kích thước `(10000,)`

### 6.3. Nhóm biến theo code hiện tại

Biến phân loại:

- `Geography`
- `Gender`
- `HasCrCard`
- `IsActiveMember`

Biến số:

- `CreditScore`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `EstimatedSalary`

Đây là thay đổi quan trọng cần phản ánh đúng: `HasCrCard` và `IsActiveMember` hiện đã được đưa sang nhóm biến phân loại và sẽ được `OneHotEncoder` xử lý.

### 6.4. `ColumnTransformer`

Notebook xây tiền xử lý như sau:

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

## 7. Chia tập train/test

Notebook dùng:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
```

Output đang lưu:

- train size: `(8000, 10)`
- test size: `(2000, 10)`
- tỷ lệ lớp trong train: class 0 = `0.79625`, class 1 = `0.20375`
- tỷ lệ lớp trong test: class 0 = `0.7965`, class 1 = `0.2035`

## 8. Xử lý mất cân bằng dữ liệu

Tất cả mô hình tuned đều dùng cấu trúc:

```text
preprocessor -> RandomOverSampler -> model
```

Cách này giúp oversampling chỉ xảy ra trong quy trình train/CV, giảm nguy cơ leakage sang validation hoặc test.

## 9. Tối ưu siêu tham số với GridSearchCV

Notebook hiện tối ưu 4 mô hình:

1. `LogisticRegression`
2. `DecisionTreeClassifier`
3. `RandomForestClassifier`
4. `GradientBoostingClassifier`

Cross-validation:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```

Thước đo tối ưu:

```python
scoring="recall"
```

### 9.1. Kết quả tốt nhất theo output của notebook

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

## 10. Kết quả đánh giá trên tập test

Notebook sắp xếp bảng kết quả theo `Recall` giảm dần, sau đó `F1-score` giảm dần.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.803000 | 0.510638 | 0.766585 | 0.612967 | 0.865834 |
| Decision Tree | 0.774000 | 0.466165 | 0.761671 | 0.578358 | 0.847119 |
| Logistic Regression | 0.716000 | 0.391069 | 0.710074 | 0.504363 | 0.778086 |
| Random Forest | 0.793000 | 0.493892 | 0.695332 | 0.577551 | 0.852518 |

## 11. Mô hình tốt nhất

Notebook chọn:

- `Gradient Boosting`

Lý do:

- `Recall` cao nhất: `0.766585`
- `F1-score` cao nhất: `0.612967`
- `ROC-AUC` cao nhất: `0.865834`

### 11.1. Classification report

Kết quả đang lưu trong notebook:

- lớp `0`: precision `0.93`, recall `0.81`, f1-score `0.87`
- lớp `1`: precision `0.51`, recall `0.77`, f1-score `0.61`
- accuracy tổng thể: `0.80`

## 12. Feature Importance

Notebook hiện có thêm phần phân tích `Feature Importance` của mô hình tốt nhất.

Top feature theo output hiện tại:

1. `Age` - `0.432889`
2. `NumOfProducts` - `0.316866`
3. `Balance` - `0.067476`
4. `Geography_Germany` - `0.061781`
5. `IsActiveMember_0` - `0.051158`

Đây là thay đổi lớn so với README cũ: notebook hiện có bước giải thích mô hình ở mức cơ bản.

## 13. File đầu ra hiện tại

Notebook hiện xuất trực tiếp file:

```python
final_results_df.to_csv("gridsearch_model_results.csv", index=False)
```

và tải file đó xuống bằng:

```python
files.download("gridsearch_model_results.csv")
```

Ngoài ra còn sinh thêm:

- `bank_customer_churn_eda_report.html`

Lưu ý: tên file output trong notebook hiện là `gridsearch_model_results.csv`, không phải tên cũ khác.

## 14. Thư viện sử dụng

Notebook cài:

```bash
pip install kaggle ydata-profiling xgboost imbalanced-learn -q
```

Các thư viện chính được import gồm:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`
- `ydata-profiling`
- `xgboost`

Lưu ý: `xgboost` đang được cài/import nhưng phiên bản notebook hiện tại không huấn luyện mô hình XGBoost.

## 15. Hạn chế hiện tại

- Notebook đang phụ thuộc vào Google Colab qua `drive.mount()` và `files.download()`.
- Phiên bản hiện tại bỏ phần baseline, đi thẳng vào `GridSearchCV`.
- `RandomOverSampler` có thể làm tăng overfitting do nhân bản mẫu lớp thiểu số.
- Mô hình vẫn dùng ngưỡng phân loại mặc định `0.5`.

## 16. Cách chạy local

Nếu chạy ngoài Colab, nên đổi phần đọc dữ liệu thành:

```python
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")
```

và bỏ hoặc thay thế các dòng:

```python
from google.colab import drive, files
drive.mount('/content/drive')
files.download(...)
```

## 17. Tác giả

```text
Author : Nguyen Thi Thao My
Project: Bank Customer Churn Classification
```

## 18. License

```text
MIT License
```
