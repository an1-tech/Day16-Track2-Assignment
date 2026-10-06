Triển khai Terraform tại AWS us-east-1, chạy LightGBM trên EC2 t3.medium.
Dataset gồm 284.807 giao dịch, chia train/validation/test theo tỷ lệ 60/20/20.
Thời gian load dữ liệu: 2,529 giây; training: 2,303 giây.
AUC-ROC: 0,9381; accuracy: 99,907%; F1: 0,7644.
Precision: 67,72%; recall: 87,76%; cần đánh giá cùng nhau vì dữ liệu mất cân bằng.
Early stopping chọn best_iteration=1; tốc độ đo được thuộc mô hình này.
Latency một dòng: 1,181 ms; throughput batch 1.000 dòng: khoảng 698.614 dòng/giây.
Ảnh tài nguyên sau benchmark cho thấy RAM dùng khoảng 244 MiB, CPU chủ yếu idle.
Billing ghi nhận tại us-east-1: NAT Gateway 0,10 USD, EC2 0,04 USD và ALB 0,02 USD; tổng hiển thị sau các khoản bù trừ là 0 USD tại thời điểm chụp.
Đã hoàn tất terraform destroy, xóa thành công 27 tài nguyên.