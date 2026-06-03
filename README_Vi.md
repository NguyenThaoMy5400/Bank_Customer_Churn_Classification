# README - Bank Customer Churn Classification

## 1. Giới thiệu

File [Bank_Customer_Churn_Classification.ipynb](D:/University/Machine learning/BTL/Bank_Customer_Churn_Classification/Bank_Customer_Churn_Classification.ipynb) xây dựng một bài toán **phân loại nhị phân** để dự đoán khách hàng có rời bỏ ngân hàng hay không.

Biến mục tiêu là `Exited`:

- `Exited = 0`: Khách hàng không rời bỏ ngân hàng
- `Exited = 1`: Khách hàng rời bỏ ngân hàng

Ý nghĩa thực tế của bài toán là hỗ trợ ngân hàng nhận diện sớm nhóm khách hàng có nguy cơ rời bỏ để đưa ra chiến lược giữ chân phù hợp.

Notebook này tập trung vào hướng làm việc sau:

- khám phá và phân tích dữ liệu;
- tiền xử lý dữ liệu đúng pipeline;
- xử lý mất cân bằng lớp bằng `RandomOverSampler`;
- tối ưu siêu tham số bằng `GridSearchCV`;
- đánh giá và chọn mô hình tốt nhất theo tiêu chí ưu tiên `Recall`.

## 2. Dữ liệu sử dụng

Notebook dùng file `Churn_Modelling.csv` trong cùng thư mục.

Nguồn dataset:

Kaggle: <https://www.kaggle.com/datasets/aakash50897/churn-modellingcsv>

### 2.1. Quy mô dữ liệu

Theo phần kiểm tra trong notebook:

- số dòng: `10,000`
- số cột: `14`

### 2.2. Danh sách cột gốc

Dataset gồm các cột:

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

- `3` cột kiểu `object`: `Surname`, `Geography`, `Gender`
- `9` cột kiểu `int64`
- `2` cột kiểu `float64`

### 2.4. Chất lượng dữ liệu

Kết quả kiểm tra trong notebook:

- không có giá trị thiếu ở tất cả các cột;
- không có dòng bị trùng lặp.

Điều này giúp quy trình tiền xử lý đơn giản hơn vì không cần bước imputing hay xóa duplicate.

## 3. Mục tiêu và cách tiếp cận chính

Notebook không đi theo kiểu làm baseline trước rồi mới tối ưu. Thay vào đó, tác giả chọn cách:

1. hiểu dữ liệu qua EDA;
2. xây dựng pipeline tiền xử lý;
3. xử lý mất cân bằng ngay trong pipeline;
4. dùng `GridSearchCV` để tìm cấu hình tốt nhất cho từng mô hình;
5. so sánh các mô hình sau tối ưu.

Đây là điểm quan trọng của notebook vì nó quyết định cấu trúc toàn bộ phần huấn luyện.

## 4. Cấu trúc notebook theo từng phần

### 4.1. Cài đặt thư viện

Notebook cài các thư viện bằng lệnh:

```python
!pip install kaggle ydata-profiling xgboost imbalanced-learn -q
```

Sau đó import các nhóm thư viện chính:

- xử lý dữ liệu: `numpy`, `pandas`
- trực quan hóa: `matplotlib`, `seaborn`
- tiền xử lý và mô hình hóa: `scikit-learn`
- xử lý mất cân bằng: `imblearn`
- mô hình nâng cao: `xgboost`

Lưu ý: trong notebook có import thêm `KNeighborsClassifier`, `SVC`, `XGBClassifier` nhưng ở phiên bản hiện tại các mô hình này **không được đưa vào huấn luyện thực tế**.

### 4.2. Tải dữ liệu

Phần tải dữ liệu được viết theo ngữ cảnh **Google Colab**:

```python
from google.colab import drive, files
drive.mount('/content/drive')
dataset_path = '/content/drive/MyDrive/ML/BTL/Datasets/Churn_Modelling.csv'
```

Notebook nêu 2 cách lấy dữ liệu:

- tải từ Kaggle;
- upload file CSV thủ công lên Colab.

Trong mã hiện tại, dữ liệu được đọc từ Google Drive. Nếu chạy local bằng Jupyter trên máy, phần này cần đổi lại đường dẫn file.

### 4.3. Tổng quan dữ liệu

Các cell đầu của phần này dùng để:

- xem kích thước dữ liệu với `df.shape`;
- xem cấu trúc cột với `df.info()`;
- xem thống kê mô tả với `df.describe()`;
- xem một vài dòng đầu với `df.head()`.

Mục tiêu của phần này là nắm nhanh:

- số lượng bản ghi;
- loại dữ liệu của từng cột;
- khoảng giá trị của các biến số;
- dữ liệu có đọc đúng hay không.

### 4.4. Mô tả ý nghĩa các thuộc tính

Notebook có một bảng markdown giải thích ngắn ý nghĩa của các cột thường gặp trong bài toán churn. Đây là phần giúp người đọc không chỉ nhìn cột dữ liệu mà còn hiểu bối cảnh nghiệp vụ.

Ví dụ:

- `CreditScore`: điểm tín dụng;
- `Geography`: khu vực khách hàng;
- `Balance`: số dư tài khoản;
- `IsActiveMember`: khách hàng có đang hoạt động hay không;
- `Exited`: khách hàng có rời bỏ hay không.

## 5. Phân tích khám phá dữ liệu (EDA)

Đây là phần quan trọng nhất trước khi xây mô hình.

### 5.1. Kiểm tra missing values và duplicate

Notebook dùng:

```python
df.isnull().sum().sort_values(ascending=False)
df.duplicated().sum()
```

Kết quả:

- missing values: `0` cho toàn bộ cột;
- duplicate rows: `0`.

### 5.2. Phân tích biến mục tiêu `Exited`

Kết quả đếm nhãn:

- `Exited = 0`: `7963` mẫu
- `Exited = 1`: `2037` mẫu

Tỷ lệ phần trăm:

- lớp 0: `79.63%`
- lớp 1: `20.37%`

Nhận xét:

- dữ liệu bị **mất cân bằng lớp**;
- nếu chỉ nhìn `Accuracy` thì mô hình có thể trông tốt nhưng lại bỏ sót nhiều khách hàng churn;
- vì vậy notebook ưu tiên các chỉ số `Recall`, `F1-score`, `ROC-AUC`.

### 5.3. Phân tích theo giới tính

Notebook dùng `countplot` và tính tỷ lệ churn theo `Gender`.

Kết quả:

- `Female`: khoảng `25.07%`
- `Male`: khoảng `16.46%`

Gợi ý từ dữ liệu: nhóm nữ trong dataset này có tỷ lệ churn cao hơn nhóm nam.

### 5.4. Phân tích theo khu vực địa lý

Notebook phân tích `Geography` bằng biểu đồ cột và trung bình của `Exited` theo nhóm.

Kết quả:

- `Germany`: khoảng `32.44%`
- `Spain`: khoảng `16.67%`
- `France`: khoảng `16.15%`

Diễn giải: khách hàng ở Germany có xu hướng churn cao hơn rõ rệt trong dataset này.

### 5.5. Phân tích theo tuổi

Notebook dùng `histplot` với `Age` và tô màu theo `Exited`.

Mục đích:

- xem phân bố tuổi của khách hàng churn và không churn;
- kiểm tra liệu tuổi có phải biến ảnh hưởng mạnh đến khả năng rời bỏ hay không.

### 5.6. Phân tích theo điểm tín dụng

Notebook trực quan hóa `CreditScore` theo trạng thái churn.

Mục tiêu:

- kiểm tra xem điểm tín dụng thấp hay cao có liên quan đến hành vi rời bỏ không;
- quan sát mức phân tách giữa hai lớp.

### 5.7. Phân tích theo số dư tài khoản

Biến `Balance` được kiểm tra bằng histogram theo hai lớp mục tiêu.

Ý nghĩa:

- xem khách hàng có số dư cao hoặc thấp có xu hướng churn khác nhau không.

### 5.8. Phân tích theo số lượng sản phẩm

Kết quả churn theo `NumOfProducts`:

- `4`: `100.00%`
- `3`: `82.71%`
- `1`: `27.71%`
- `2`: `7.58%`

Nhận xét:

- nhóm có `3` hoặc `4` sản phẩm có tỷ lệ churn rất cao;
- nhóm có `2` sản phẩm lại có tỷ lệ churn thấp nhất.

Đây là tín hiệu mạnh cho mô hình, dù cần cẩn trọng vì một số nhóm có thể có ít mẫu.

### 5.9. Phân tích theo trạng thái hoạt động

Tỷ lệ churn theo `IsActiveMember`:

- không hoạt động (`0`): `26.85%`
- đang hoạt động (`1`): `14.27%`

Diễn giải: khách hàng không active có xu hướng rời bỏ cao hơn rõ rệt.

### 5.10. Phân tích theo thẻ tín dụng

Tỷ lệ churn theo `HasCrCard`:

- không có thẻ: `20.81%`
- có thẻ: `20.18%`

Nhận xét: biến này có vẻ không tạo khác biệt lớn như `Age`, `Geography`, `NumOfProducts` hoặc `IsActiveMember`.

### 5.11. Ma trận tương quan

Notebook chọn các cột số và vẽ heatmap tương quan.

Mục tiêu của bước này:

- xem mối liên hệ tuyến tính giữa các biến số;
- phát hiện biến có khả năng liên quan đến `Exited`;
- hỗ trợ trực giác trước khi mô hình hóa.

### 5.12. Kết luận sau EDA

Notebook tổng hợp 4 ý chính:

- dữ liệu có cả biến số và biến phân loại;
- biến mục tiêu bị mất cân bằng;
- các biến như `Age`, `Balance`, `NumOfProducts`, `IsActiveMember`, `Geography` có thể quan trọng;
- các cột định danh như `RowNumber`, `CustomerId`, `Surname` không có nhiều giá trị dự đoán.

## 6. Tiền xử lý dữ liệu

### 6.1. Loại bỏ cột không cần thiết

Notebook tạo `df_model = df.copy()` rồi xóa 3 cột:

- `RowNumber`
- `CustomerId`
- `Surname`

Lý do:

- đây là các cột nhận diện hoặc thứ tự;
- ít giá trị dự đoán trực tiếp cho churn;
- giữ lại có thể gây nhiễu cho mô hình.

### 6.2. Tách biến đầu vào và biến mục tiêu

```python
X = df_model.drop(columns=["Exited"])
y = df_model["Exited"]
```

Kích thước sau khi tách:

- `X`: `(10000, 10)`
- `y`: `(10000,)`

### 6.3. Xác định biến phân loại và biến số

Notebook tự động tách:

- biến phân loại: `Geography`, `Gender`
- biến số: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`

### 6.4. Xây dựng `ColumnTransformer`

Pipeline tiền xử lý được xây như sau:

- nhánh số: `StandardScaler()`
- nhánh phân loại: `OneHotEncoder(handle_unknown="ignore")`

Mục đích:

- chuẩn hóa các biến số để mô hình học ổn định hơn;
- one-hot encode biến phân loại để mô hình có thể xử lý dữ liệu dạng text;
- tránh lỗi khi tập test có category mới nhờ `handle_unknown="ignore"`.

## 7. Chia tập train/test

Notebook dùng:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Ý nghĩa của các tham số:

- `test_size=0.2`: 80% train, 20% test;
- `random_state=42`: đảm bảo tái lập kết quả;
- `stratify=y`: giữ tỷ lệ lớp giống nhau giữa train và test.

Phân bố lớp ở tập train và test được in ra để xác nhận việc chia dữ liệu là hợp lý.

## 8. Xử lý mất cân bằng bằng `RandomOverSampler`

### 8.1. Vì sao cần oversampling

Vì lớp churn chỉ chiếm khoảng `20.37%`, nếu huấn luyện trực tiếp thì mô hình có thể thiên về lớp đa số.

### 8.2. Cách notebook xử lý

Notebook đặt oversampling bên trong `ImbPipeline`:

```text
preprocessor -> RandomOverSampler -> model
```

Điều này rất quan trọng vì:

- chỉ oversample trên dữ liệu train;
- tránh rò rỉ dữ liệu sang validation/test;
- đảm bảo mỗi fold trong `GridSearchCV` xử lý đúng quy trình.

### 8.3. Minh họa trước và sau oversampling

Phân bố lớp trong tập train trước oversampling:

- lớp 0: `6370`
- lớp 1: `1630`

Sau `RandomOverSampler` demo:

- lớp 0: `6370`
- lớp 1: `6370`

Kết luận: lớp thiểu số đã được nhân bản để cân bằng với lớp đa số.

## 9. Hàm đánh giá mô hình

Notebook định nghĩa hàm `evaluate_model()` để tính các chỉ số sau trên tập test:

- `Accuracy`
- `Precision`
- `Recall`
- `F1-score`
- `ROC-AUC`

Thiết kế của hàm:

- dùng `predict()` để lấy nhãn dự đoán;
- nếu mô hình có `predict_proba()`, notebook lấy xác suất lớp 1 để tính `ROC-AUC`;
- trả về dictionary kết quả để dễ tổng hợp thành DataFrame.

Đây là cách tổ chức tốt vì tách riêng phần đánh giá khỏi phần huấn luyện.

## 10. Tối ưu siêu tham số với `GridSearchCV`

### 10.1. Cross-validation

Notebook dùng:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Ý nghĩa:

- chia 5 fold;
- giữ tỷ lệ lớp tương đối ổn định ở mỗi fold;
- trộn dữ liệu trước khi chia.

### 10.2. Thước đo tối ưu

Tất cả các `GridSearchCV` đều dùng:

```python
scoring="recall"
```

Đây là lựa chọn có chủ đích. Trong bài toán churn, bỏ sót khách hàng sắp rời bỏ thường tệ hơn việc cảnh báo nhầm một số khách hàng không rời bỏ.

### 10.3. Các mô hình được tối ưu

Notebook huấn luyện và tìm tham số tốt nhất cho 4 mô hình:

1. `LogisticRegression`
2. `DecisionTreeClassifier`
3. `RandomForestClassifier`
4. `GradientBoostingClassifier`

Mỗi mô hình đều được đặt trong cùng một khung pipeline:

```text
preprocessor -> RandomOverSampler -> model
```

### 10.4. Không gian tìm kiếm của từng mô hình

#### Logistic Regression

Grid tham số:

- `C`: `[0.01, 0.1, 1, 10]`
- `solver`: `['liblinear', 'lbfgs']`
- `class_weight`: `[None, 'balanced']`

Kết quả tốt nhất:

- `model__C = 1`
- `model__solver = 'liblinear'`
- `model__class_weight = None`
- Best CV Recall: `0.688957055214724`

#### Decision Tree

Grid tham số:

- `max_depth`: `[3, 5, 7, 10, None]`
- `min_samples_split`: `[2, 5, 10]`
- `min_samples_leaf`: `[1, 2, 4]`
- `class_weight`: `[None, 'balanced']`

Kết quả tốt nhất:

- `model__max_depth = 5`
- `model__min_samples_split = 2`
- `model__min_samples_leaf = 4`
- `model__class_weight = None`
- Best CV Recall: `0.7533742331288343`

#### Random Forest

Grid tham số:

- `n_estimators`: `[100, 200]`
- `max_depth`: `[5, 10, None]`
- `min_samples_split`: `[2, 5]`
- `class_weight`: `[None, 'balanced']`

Kết quả tốt nhất:

- `model__n_estimators = 100`
- `model__max_depth = 5`
- `model__min_samples_split = 5`
- `model__class_weight = None`
- Best CV Recall: `0.7288343558282209`

#### Gradient Boosting

Grid tham số:

- `n_estimators`: `[100, 200]`
- `learning_rate`: `[0.01, 0.05, 0.1]`
- `max_depth`: `[2, 3, 5]`

Kết quả tốt nhất:

- `model__n_estimators = 100`
- `model__learning_rate = 0.1`
- `model__max_depth = 2`
- Best CV Recall: `0.7398773006134969`

## 11. Kết quả đánh giá trên tập test

Sau khi lấy `best_estimator_` từ từng `GridSearchCV`, notebook đánh giá lại trên tập test và sắp xếp theo:

1. `Recall` giảm dần
2. `F1-score` giảm dần

### 11.1. Bảng kết quả cuối

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8030 | 0.510638 | 0.766585 | 0.612967 | 0.865834 |
| Decision Tree | 0.7740 | 0.466165 | 0.761671 | 0.578358 | 0.847372 |
| Random Forest | 0.7875 | 0.485669 | 0.749386 | 0.589372 | 0.854839 |
| Logistic Regression | 0.7160 | 0.391069 | 0.710074 | 0.504363 | 0.778103 |

### 11.2. Mô hình tốt nhất

Theo tiêu chí của notebook, mô hình tốt nhất là:

- `Gradient Boosting`

Lý do:

- có `Recall` cao nhất: `0.766585`
- đồng thời có `F1-score` cao nhất trong số các mô hình so sánh
- `ROC-AUC` cũng cao nhất: `0.865834`

### 11.3. Classification Report của mô hình tốt nhất

Kết quả được notebook in ra:

- lớp `0`: precision `0.93`, recall `0.81`, f1-score `0.87`
- lớp `1`: precision `0.51`, recall `0.77`, f1-score `0.61`
- accuracy toàn bộ: `0.80`

Diễn giải:

- mô hình phát hiện tương đối tốt nhóm khách hàng churn với recall khoảng `77%`;
- precision của lớp churn còn ở mức trung bình, nghĩa là vẫn có số lượng dự đoán dương tính giả đáng kể;
- đây là đánh đổi thường gặp khi tối ưu theo `Recall`.

### 11.4. Confusion Matrix và ROC Curve

Notebook trực quan hóa thêm:

- `Confusion Matrix` để quan sát số dự đoán đúng và sai theo từng lớp;
- `ROC Curve` để nhìn khả năng phân tách hai lớp của mô hình.

Hai biểu đồ này giúp kiểm chứng kết quả ngoài các chỉ số dạng số.

## 12. File kết quả được xuất ra

Notebook lưu bảng kết quả cuối thành file:

```python
final_results_df.to_csv("gridsearch_model_results.csv", index=False)
```

Trong Colab, file này còn được tải xuống bằng:

```python
files.download("gridsearch_model_results.csv")
```

## 13. Ý nghĩa kỹ thuật của pipeline

Một điểm tốt trong notebook là cách ghép các bước theo pipeline chuẩn:

```text
ColumnTransformer -> RandomOverSampler -> Model
```

Lợi ích:

- tránh rò rỉ dữ liệu;
- đảm bảo preprocessing và oversampling được áp dụng nhất quán;
- dễ mở rộng sang mô hình khác;
- dễ đưa vào `GridSearchCV`.

Đây là cách làm đúng hơn nhiều so với việc oversample toàn bộ dữ liệu trước khi chia train/test.

## 14. Ưu điểm

- Có EDA tương đối đầy đủ trước khi huấn luyện.
- Có kiểm tra mất cân bằng lớp và giải thích vì sao ưu tiên `Recall`.
- Sử dụng `ColumnTransformer` và `Pipeline` hợp lý.
- Đặt `RandomOverSampler` trong `ImbPipeline`, tránh leakage.
- Dùng `StratifiedKFold` để giữ phân phối lớp trong cross-validation.
- So sánh nhiều mô hình sau khi tối ưu siêu tham số.
- Có lưu kết quả cuối ra file CSV.

## 15. Những điểm cần lưu ý hoặc hạn chế

### 15.1. Notebook phụ thuộc Google Colab

Phần đọc dữ liệu và tải file dùng `google.colab`. Nếu chạy local, cần sửa:

- bỏ `drive.mount()`;
- đổi đường dẫn `dataset_path` về file CSV trong máy;
- bỏ `files.download()` nếu không cần.


### 15.2. Chưa tối ưu ngưỡng phân loại

Mô hình vẫn dùng ngưỡng mặc định `0.5`. Với bài toán churn, tối ưu threshold theo business goal có thể cải thiện hiệu quả thực tế.

### 15.3. RandomOverSampler có nguy cơ overfitting

Do kỹ thuật này chỉ nhân bản lại mẫu lớp thiểu số, mô hình có thể học quá sát dữ liệu train hơn so với một số phương pháp khác như `SMOTE`.

## 16. Cách chạy notebook

### 16.1. Chạy trên Google Colab

1. Upload notebook lên Colab.
2. Upload hoặc mount file `Churn_Modelling.csv`.
3. Kiểm tra lại `dataset_path` cho đúng.
4. Chạy lần lượt từ trên xuống dưới.

### 16.2. Chạy trên máy local

Nếu dùng Jupyter Notebook local, nên sửa phần đọc dữ liệu thành dạng đơn giản như sau:

```python
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv")
```

Và bỏ các dòng:

```python
from google.colab import drive, files
drive.mount('/content/drive')
files.download(...)
```

### 16.3. Thư viện cần có

Các thư viện chính notebook đang dùng:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost kaggle ydata-profiling
```

Trong thực tế, `kaggle` và `ydata-profiling` không phải là bắt buộc để chạy toàn bộ phần mô hình hiện tại nếu bạn chỉ dùng file CSV sẵn có.

## 17. Tóm tắt ngắn gọn logic toàn bài

Toàn bộ notebook có thể tóm tắt thành chuỗi xử lý sau:

1. đọc dữ liệu churn;
2. kiểm tra cấu trúc và chất lượng dữ liệu;
3. phân tích các đặc trưng liên quan đến churn;
4. loại bỏ cột không cần thiết;
5. tách `X`, `y`;
6. chuẩn hóa biến số và one-hot encode biến phân loại;
7. chia train/test có `stratify`;
8. oversample lớp thiểu số bằng `RandomOverSampler` trong pipeline;
9. tối ưu 4 mô hình bằng `GridSearchCV` với `scoring='recall'`;
10. đánh giá trên test set;
11. chọn `Gradient Boosting` là mô hình tốt nhất;
12. lưu kết quả ra CSV.

## 18. Kết luận

Về mặt kết quả, `Gradient Boosting` là mô hình tốt nhất trong 4 mô hình đã thử, với khả năng phát hiện khách hàng churn tương đối tốt nhờ `Recall` cao nhất. Tuy nhiên, notebook vẫn còn không gian cải thiện ở các phần như tối ưu threshold, giải thích mô hình và thử thêm kỹ thuật resampling nâng cao.

# 👨‍💻 Tác giả

```text
Author : <Nguyen Thi Thao My>
Project: House Rent Prediction
```

---

# 📜 License

Dự án có thể sử dụng giấy phép:

```text
MIT License
```
