# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Bùi Đức Hiếu |
| MSSV | 2A202601820 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/buiduchieu24/K4-Track2-Day21-2A202601820-BuiDucHieu |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 200 | 0.2 | 5 | 0.7032 | 0.8700 |
| 3 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`.

**Lý do:** Bộ siêu tham số ở lần chạy 1 đạt điểm `f1_score` cao nhất là 0.7109 và vượt qua ngưỡng chất lượng tối thiểu (0.65). Lần chạy có accuracy cao nhất (0.8780) cũng đồng thời trùng với lần có f1_score cao nhất. Tuy nhiên, khi so sánh giữa lần 2 và lần 3, ta thấy ở lần 3 dù accuracy vẫn đạt 0.8460 nhưng f1_score chỉ đạt 0.6051 do mô hình quá đơn giản (underfitting). Ở lần 2, việc tăng đồng thời cả `n_estimators=200`, `learning_rate=0.2` và `max_depth=5` khiến mô hình có xu hướng học quá mức trên tập huấn luyện, làm giảm nhẹ hiệu quả tổng quát hóa trên tập holdout (f1 giảm xuống 0.7032). Do đó, sự cân bằng giữa số lượng cây vừa phải và tốc độ học 0.1 ở cấu hình 1 mang lại hiệu năng tối ưu nhất.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có sự mất cân bằng lớp rõ rệt khi lớp thiểu số (thu nhập > 50K) chỉ chiếm khoảng 24.8% tổng số mẫu. Nếu một mô hình đơn giản chỉ luôn dự đoán nhãn "thu nhập thấp" cho tất cả các mẫu thì độ chính xác (Accuracy) vẫn đạt tới 75.2%, tạo ra cảm giác sai lệch rằng mô hình hoạt động tốt trong khi thực tế nó hoàn toàn vô dụng vì không phát hiện được bất kỳ trường hợp thu nhập cao nào (F1 = 0.0). Chỉ số F1 của lớp dương đo lường giá trị trung bình điều hòa giữa Precision và Recall, phản ánh chính xác năng lực phát hiện và độ tin cậy đối với nhóm khách hàng thu nhập cao. Lab không sử dụng tham số `average="macro"` hay `average="weighted"` nhằm tránh việc lớp đa số chiếm ưu thế làm sai lệch và nâng cao điểm số một cách giả tạo, đảm bảo cổng kiểm soát chất lượng (Quality Gate) hoạt động thực chất.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow UI không hiển thị các lần chạy huấn luyện | Lệnh export biến môi trường không hoạt động trên PowerShell Windows khiến MLflow ghi vào ./mlruns thay vì SQLite | Bổ sung hàm thiết lập Tracking URI trực tiếp trong file mã nguồn `src/train.py` trỏ vào SQLite database |
| Lỗi thiếu pkg_resources khi chạy pytest và import mlflow | Phiên bản setuptools mới tự động loại bỏ module pkg_resources | Cài đặt cố định phiên bản setuptools tương thích (`setuptools<70`) trong môi trường ảo |
| Khác biệt phiên bản Python giữa môi trường gốc và các gói ML | Python 3.13 chưa có sẵn binary wheel cho một số thư viện scikit-learn cũ | Khởi tạo môi trường ảo chuyên biệt với Python 3.12 để đảm bảo tính tương thích |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7109 | 0.8780 |
| Bước 3 (thêm `train_batch2`) | 0.7014 | 0.8740 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu dữ liệu mới ở Bước 3 (tổng cộng 44.722 mẫu), điểm F1 giảm nhẹ từ 0.7109 xuống 0.7014 và Accuracy dao động nhẹ từ 0.8780 xuống 0.8740. Điều này hoàn toàn phù hợp với thực tế vì tập dữ liệu mới được phân chia ngẫu nhiên từ cùng một phân phối dữ liệu ban đầu, không mang thêm thông tin đột phá mới mà mô hình chưa học được từ 22.361 mẫu đầu. Điểm cốt lõi quan trọng nhất của Bước 3 là đã chứng minh được tính liên tục và hoàn toàn tự động của pipeline CI/CD: chỉ cần cập nhật dữ liệu và push commit DVC lên GitHub, toàn bộ chu trình kiểm thử, huấn luyện lại, kiểm tra chất lượng và tái triển khai lên máy chủ EC2 đều diễn ra trơn tru mà không cần can thiệp thủ công.
