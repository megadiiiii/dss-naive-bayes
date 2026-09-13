# DSS - Hệ Thống Hỗ Trợ Ra Quyết Định Phân Loại Email Spam (Naive Bayes)

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NLP Library](https://img.shields.io/badge/NLP-Underthesea-brightgreen.svg)](https://github.com/undertheseanlp/underthesea)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Environment-Jupyter%20Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Đồ án môn học:** Hệ Thống Hỗ Trợ Ra Quyết Định (Decision Support System - DSS)  
> **Đề tài:** Ứng dụng mô hình học máy **Naive Bayes** và kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) tiếng Việt để phát hiện và lọc email rác (Spam).

---

## Mục Lục
- [Giới Thiệu Đề Tài](#giới-thiệu-đề-tài)
- [Tính Năng Nổi Bật](#tính-năng-nổi-bật)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Quy Trình Xử Lý & Kiến Trúc](#quy-trình-xử-lý--kiến-trúc)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Hướng Dẫn Cài Đặt & Chạy](#hướng-dẫn-cài-đặt--chạy)
  - [Cách 1: Chạy trên máy cục bộ (PyCharm / VS Code / Jupyter)](#cách-1-chạy-trên-máy-cục-bộ-pycharm--vs-code--jupyter)
  - [Cách 2: Chạy trên Google Colab](#cách-2-chạy-trên-google-colab)
- [Kết Quả & Đánh Giá](#kết-quả--đánh-giá)
- [Lưu Ý Quan Trọng](#lưu-ý-quan-trọng)
- [Tài Liệu Báo Cáo](#tài-liệu-báo-cáo)

---

## Giới Thiệu Đề Tài

Email là kênh truyền thông phổ biến nhưng cũng là đích nhắm hàng đầu của thư rác (Spam), email quảng cáo không mong muốn và các chiến dịch lừa đảo mạo danh (Phishing). 

Dự án này xây dựng một **Hệ thống hỗ trợ ra quyết định (DSS)** nhằm:
1. **Phân tích ngữ nghĩa văn bản tiếng Việt** từ tiêu đề (`Subject`) và nội dung (`Body`) của email.
2. Ứng dụng định lý xác suất **Bayes** kết hợp giả định độc lập có điều kiện (**Naive Bayes**) để tính toán xác suất một email là **Spam** hay **Ham** (thư thường).
3. Cung cấp báo cáo xác suất định lượng giúp người dùng đưa ra quyết định cảnh giác trước các rủi ro bảo mật thông tin.

---

## Tính Năng Nổi Bật

- **Xử lý ngôn ngữ tự nhiên tiếng Việt chuyên sâu:** Tích hợp bộ tách từ `underthesea` chuẩn hóa từ ghép tiếng Việt (ví dụ: *miễn_phí*, *trúng_thưởng*, *hôm_nay*).
- **Mô phỏng toán học chi tiết (`bayes.ipynb`):** Cài đặt thuật toán Naive Bayes thủ công từ gốc, minh họa trực quan:
  - Tần suất xuất hiện từ vựng.
  - Xác suất tiên nghiệm $P(c)$ (Prior Probability).
  - Xác suất có điều kiện $P(w|c)$ có làm mịn **Laplace Smoothing** ($+1$).
  - Tính tổng Log-likelihood và chuẩn hóa xác suất đầu ra (0% - 100%).
- **Pipeline Machine Learning hoàn chỉnh (`input.ipynb`):**
  - Đọc và xử lý tập dữ liệu thực tế hơn 2.300 email tiếng Việt (`dataset.csv`).
  - Loại bỏ từ dừng với từ điển chuyên dụng `vietnamese-stopwords-dash.txt`.
  - Trích xuất đặc trưng với **TF-IDF Vectorizer**.
  - Huấn luyện mô hình **Multinomial Naive Bayes** (`MultinomialNB`) đạt độ chính xác cao.
- **Trực quan hóa dữ liệu (Visualization):** Vẽ biểu đồ phân bố Top 20 từ khóa đặc trưng nhất cho từng nhãn Spam và Ham bằng `matplotlib`.
- **Giao diện dòng lệnh tương tác trực tiếp:** Người dùng có thể dán tiêu đề và nội dung email bất kỳ để kiểm tra tức thì.
- **Lưu vết lịch sử (Audit Log):** Mọi lượt kiểm tra đều được tự động lưu vào `filter_history.csv` kèm thời gian thực để theo dõi và đối soát.

---

## Cấu Trúc Thư Mục

```text
dss-naive-bayes/
│
├── dataset.csv                     # Bộ dữ liệu mẫu email tiếng Việt (Spam/Ham)
├── vietnamese-stopwords-dash.txt    # Danh sách từ dừng (Stopwords) tiếng Việt
├── filter_history.csv              # Lịch sử các lần nhập và phân loại email
│
├── bayes.ipynb                     # Notebook cốt lõi: Minh họa chi tiết thuật toán toán học
├── input.ipynb                     # Notebook chính: Pipeline Train/Test + Giao diện nhập email
├── lib_install.ipynb               # Notebook phụ trợ: Cài đặt các thư viện phụ thuộc
│
├── requirements.txt                # Danh sách thư viện Python cần cài đặt
├── README.md                       # Tài liệu hướng dẫn sử dụng dự án
└── DSS - Final Project Report.pdf  # Báo cáo tổng kết đồ án DSS
```

---

## Quy Trình Xử Lý & Kiến Trúc

```text
============================== 1. QUY TRÌNH HUẤN LUYỆN ==============================

  [ Dữ liệu Email: dataset.csv ]
                 │
                 ▼
  [ Tiền xử lý văn bản: Làm sạch, ghép dòng Subject + Body ]
                 │
                 ▼
  [ Tách từ ghép tiếng Việt: underthesea (word_tokenize) ]
                 │
                 ▼
  [ Loại bỏ từ dừng: vietnamese-stopwords-dash.txt ]
                 │
                 ▼
  [ Trích xuất đặc trưng: TF-IDF Vectorizer ]
                 │
                 ▼
  [ Phân chia tập dữ liệu: Train 80% / Test 20% ]
                 │
                 ▼
  [ Huấn luyện mô hình: Multinomial Naive Bayes (alpha = 0.5) ]
                 │
                 ▼
  [ Đánh giá mô hình: Classification Report (Accuracy 92%) ]


============================= 2. QUY TRÌNH DỰ ĐOÁN THỰC TẾ =============================

  [ Người dùng nhập Email: Tiêu đề (Subject) + Nội dung (Body) ]
                                 │
                                 ▼
              [ Tiền xử lý văn bản & Tách từ tiếng Việt ]
                                 │
                                 ▼
                  [ Chuyển đổi qua TF-IDF Vectorizer ]
                                 │
                                 ▼
             [ Dự đoán bằng mô hình Multinomial Naive Bayes ]
                                 │
                                 ▼
                 [ Kết quả: Nhãn (Spam / Ham) + Tỉ lệ % ]
                                 │
                                 ▼
                 [ Tự động lưu vết vào: filter_history.csv ]
```

---

## Yêu Cầu Hệ Thống

- **Hệ điều hành:** Windows 10/11, macOS, hoặc Linux.
- **Phiên bản Python:** Khuyến nghị **Python 3.10.x** (tương thích hoàn hảo với `underthesea` và các thư viện khoa học dữ liệu).
- **Trình soạn thảo / Môi trường:** PyCharm, VS Code, Jupyter Notebook, JupyterLab hoặc Google Colab.

---

## Hướng Dẫn Cài Đặt & Chạy

### Cách 1: Chạy trên máy cục bộ (PyCharm / VS Code / Jupyter)

#### 1. Clone repository
```bash
git clone https://github.com/megadiiiii/dss-naive-bayes.git
cd dss-naive-bayes
```

#### 2. Mở thư mục bằng IDE
- Mở **PyCharm** hoặc **VS Code**, chọn **Open Folder** trỏ đến thư mục vừa clone.

#### 3. Thiết lập môi trường ảo (Virtual Environment)
Khuyến nghị tạo môi trường ảo để không xung đột thư viện:

- **Trên Windows (PowerShell / Command Prompt):**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```

- **Trên macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### 4. Cài đặt các thư viện cần thiết
Cài đặt nhanh qua file `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Hoặc mở file `lib_install.ipynb` trong IDE và bấm **Run All**).*

#### 5. Thực thi các Notebook
- **Bước 1 - Xem bản chất thuật toán:** Mở `bayes.ipynb` và chạy từng cell để xem cách tính xác suất chi tiết theo từng bước công thức toán học.
- **Bước 2 - Huấn luyện mô hình & Thử nghiệm thực tế:** 
  - Mở `input.ipynb`.
  - Chọn Kernel Python (`.venv`).
  - Bấm **Run All** để nạp dữ liệu, vector hóa TF-IDF, huấn luyện mô hình và xem biểu đồ Top từ khóa.
  - Tại Cell cuối cùng, nhập **Subject** và **Body** của email bạn muốn kiểm tra:
    ```text
    Nhập Subject (tiêu đề): Nhận quà tặng miễn phí ngay hôm nay
    Nhập Body (nội dung): Bấm vào liên kết sau để nhận 500k quà tặng khuyến mãi
    ```
  - Hệ thống sẽ trả về dự đoán nhãn (`spam`/`ham`), xác suất tin cậy (`%`) và lưu lại vào `filter_history.csv`.

---

### Cách 2: Chạy trên Google Colab

1. Tải toàn bộ repository về máy dưới dạng `.zip` hoặc clone vào Google Drive.
2. Tải các file `.ipynb`, `dataset.csv`, `vietnamese-stopwords-dash.txt` lên Google Colab.
3. Trong notebook, cài đặt thư viện bằng lệnh:
   ```python
   !pip install underthesea scikit-learn pandas numpy matplotlib
   ```
4. Chạy tuần tự các cell trong `bayes.ipynb` hoặc `input.ipynb`.

---

## Kết Quả & Đánh Giá

Mô hình **Multinomial Naive Bayes** ($\alpha = 0.5$) kết hợp bộ lọc TF-IDF và từ dừng tiếng Việt đạt được kết quả ấn tượng trên tập kiểm thử (Test set):

| Nhãn (Class) | Độ chính xác (Precision) | Khả năng bao quát (Recall) | Điểm F1-Score | Số lượng mẫu (Support) |
|:------------:|:------------------------:|:--------------------------:|:-------------:|:----------------------:|
| **HAM**      | **0.94**                 | 0.85                       | **0.89**      | 20                     |
| **SPAM**     | 0.91                     | **0.97**                   | **0.94**      | 30                     |
| **Accuracy** |                          |                            | **92%**       | **50**                 |

### Phân tích đặc trưng từ vựng:
- **Lớp SPAM:** Xuất hiện với tần suất cao các từ khóa mang tính hối thúc, tài chính hoặc quà tặng như: *nhận*, *quà*, *miễn_phí*, *khuyến_mãi*, *vi_phạm*, *thanh_toán*, *tiền*, *tài_khoản*, v.v.
- **Lớp HAM:** Tập trung vào các từ ngữ công việc, hội thoại thường ngày như: *họp*, *báo_cáo*, *hội_thảo*, *công_việc*, *dự_án*, *thông_tin*, v.v.

---

## Lưu Ý Quan Trọng

1. **Phiên bản Python:** Đảm bảo sử dụng **Python 3.10** để tránh lỗi khi cài đặt hoặc phân tích từ vựng với `underthesea`.
2. **Cảnh báo an toàn thông tin:**
   > **Lưu ý an toàn:** Kết quả phân loại mang tính chất khuyến nghị và tham khảo. Tuyệt đối **không click vào các đường link lạ**, **không mở tệp đính kèm đáng ngờ** và **không cung cấp thông tin nhạy cảm** (mật khẩu, mã OTP, số thẻ tín dụng) khi nhận được các email bị cảnh báo Spam/Phishing.

---

## Tài Liệu Báo Cáo

Chi tiết về cơ sở lý thuyết, mô hình ra quyết định, phân tích ma trận nhầm lẫn (Confusion Matrix) và giải trình thuật toán được trình bày đầy đủ trong file:
- [DSS - Final Project Report.pdf](DSS%20-%20Final%20Project%20Report.pdf)

---

<div align="center">
  <sub>Đồ án môn Hệ thống hỗ trợ ra quyết định (Decision Support System)</sub>
</div>