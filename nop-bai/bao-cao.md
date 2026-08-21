# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

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

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | ___ | ___ |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___
