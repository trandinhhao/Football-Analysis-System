import pickle  # Thư viện pickle để lưu và đọc dữ liệu từ file
import cv2  # Thư viện OpenCV để xử lý video và hình ảnh
import numpy as np  # Thư viện NumPy để làm việc với mảng số học
import os  # Thư viện OS để thao tác với hệ thống file
import sys  # Thư viện sys để thay đổi đường dẫn hệ thống
sys.path.append('../')  # Thêm thư mục cha vào đường dẫn hệ thống để import module từ thư mục khác
from utils import measure_distance, measure_xy_distance  # Import hàm đo khoảng cách từ file utils

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5  # Ngưỡng tối thiểu của khoảng cách để xác định chuyển động camera

        # Tham số cho thuật toán Lucas-Kanade Optical Flow
        self.lk_params = dict(
            winSize = (15,15),  # Kích thước của cửa sổ tìm kiếm
            maxLevel = 2,  # Số mức trong tháp ảnh (image pyramid)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)  # Điều kiện dừng của thuật toán
        )

        # Chuyển frame đầu tiên sang ảnh xám
        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        # Tạo mask để xác định khu vực chứa các đặc trưng có thể theo dõi
        mask_features = np.zeros_like(first_frame_grayscale)  # Khởi tạo mask trống
        mask_features[:,0:20] = 1  # Chọn vùng bên trái khung hình
        mask_features[:,900:1050] = 1  # Chọn vùng bên phải khung hình

        # Tham số để tìm kiếm các đặc trưng quan trọng trong ảnh
        self.features = dict(
            maxCorners = 100,  # Số lượng điểm đặc trưng tối đa
            qualityLevel = 0.3,  # Ngưỡng chất lượng của điểm đặc trưng
            minDistance = 3,  # Khoảng cách tối thiểu giữa các điểm đặc trưng
            blockSize = 7,  # Kích thước vùng lân cận để tìm điểm đặc trưng
            mask = mask_features  # Mask xác định khu vực để tìm điểm
        )

    # Thêm vị trí đã điều chỉnh của các đối tượng theo chuyển động của camera
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']  # Lấy vị trí ban đầu
                    camera_movement = camera_movement_per_frame[frame_num]  # Lấy chuyển động của camera tại frame đó
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])  # Điều chỉnh vị trí
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted  # Lưu vị trí đã điều chỉnh

    # Ước lượng chuyển động của camera trong từng frame của video
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Nếu tùy chọn đọc từ file stub được bật và file tồn tại, đọc dữ liệu từ file
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        # Khởi tạo danh sách chứa chuyển động của camera, ban đầu là [0,0] cho mỗi frame
        camera_movement = [[0,0]]*len(frames)

        # Chuyển frame đầu tiên sang ảnh xám
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # Tìm các điểm đặc trưng trên frame đầu tiên
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Duyệt qua từng frame của video
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)  # Chuyển frame hiện tại sang ảnh xám
            # Tính toán Optical Flow giữa frame trước và frame hiện tại
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0  # Khởi tạo khoảng cách lớn nhất giữa các điểm đặc trưng
            camera_movement_x, camera_movement_y = 0, 0  # Khởi tạo chuyển động camera

            # Duyệt qua các cặp điểm đặc trưng cũ và mới
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()  # Điểm đặc trưng mới
                old_features_point = old.ravel()  # Điểm đặc trưng cũ

                # Đo khoảng cách giữa hai điểm đặc trưng
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:  # Nếu khoảng cách lớn hơn khoảng cách lớn nhất đã biết
                    max_distance = distance
                    # Tính toán chuyển động theo từng trục X và Y
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            # Nếu khoảng cách lớn nhất vượt quá ngưỡng, lưu chuyển động của camera
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)  # Cập nhật các điểm đặc trưng mới

            old_gray = frame_gray.copy()  # Cập nhật frame cũ

        # Nếu có đường dẫn stub, lưu chuyển động của camera vào file stub
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement  # Trả về danh sách chuyển động của camera

    # Vẽ chuyển động của camera lên video
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []  # Danh sách chứa các frame sau khi đã vẽ thông tin

        # Duyệt qua từng frame và vẽ chuyển động của camera
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()  # Tạo bản sao của frame

            # Tạo lớp overlay và vẽ hình chữ nhật chứa thông tin
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)  # Tạo hiệu ứng trong suốt cho hình chữ nhật

            # Lấy thông tin chuyển động theo trục X và Y
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            # Vẽ thông tin chuyển động lên frame
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)  # Thêm frame vào danh sách output

        return output_frames  # Trả về danh sách frame đã vẽ
