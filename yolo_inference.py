from ultralytics import YOLO  # Nhập thư viện YOLO từ UltraLytics để sử dụng mô hình YOLO.

# Tải mô hình đã được huấn luyện từ file 'best.pt' (chứa mô hình đã được huấn luyện với dữ liệu cụ thể).
model = YOLO('models/best.pt')

# Dự đoán các đối tượng trong video '08fd33_4.mp4'. Lưu kết quả vào thư mục mặc định sau khi dự đoán.
results = model.predict('input_videos/08fd33_4.mp4', save=True)

# In kết quả dự đoán của video đầu tiên.
print(results[0])

print('=====================================')

# Duyệt qua tất cả các bounding boxes (hộp giới hạn) trong kết quả dự đoán và in chi tiết của từng box.
for box in results[0].boxes:
    print(box)
