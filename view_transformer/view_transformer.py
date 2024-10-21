import numpy as np  # Nhập thư viện NumPy để xử lý mảng và toán học.
import cv2  # Nhập thư viện OpenCV để xử lý hình ảnh và video.

class ViewTransformer():  # Định nghĩa lớp `ViewTransformer` để thực hiện phép biến đổi phối cảnh.
    def __init__(self):  # Hàm khởi tạo lớp.
        court_width = 68  # Độ rộng sân (bằng đơn vị gì đó, có thể là mét).
        court_length = 23.32  # Độ dài sân.

        # Các đỉnh của sân bóng trong không gian pixel (tạo ma trận các tọa độ của bốn góc).
        self.pixel_vertices = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])

        # Các đỉnh tương ứng của sân bóng trong không gian thực tế (tạo ma trận các tọa độ của bốn góc).
        self.target_vertices = np.array([
            [0, court_width],  # Tọa độ góc trên bên trái.
            [0, 0],  # Tọa độ góc dưới bên trái.
            [court_length, 0],  # Tọa độ góc dưới bên phải.
            [court_length, court_width]  # Tọa độ góc trên bên phải.
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)  # Chuyển đổi kiểu dữ liệu các điểm pixel thành float32.
        self.target_vertices = self.target_vertices.astype(np.float32)  # Chuyển đổi kiểu dữ liệu các điểm mục tiêu thành float32.

        # Tạo ma trận biến đổi phối cảnh từ các đỉnh pixel đến các đỉnh mục tiêu.
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):  # Hàm này nhận vào một điểm và thực hiện phép biến đổi phối cảnh của điểm đó.
        p = (int(point[0]), int(point[1]))  # Chuyển điểm về dạng tuple (x, y) với giá trị int.
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0  # Kiểm tra điểm có nằm trong tứ giác không gian pixel hay không.
        if not is_inside:  # Nếu điểm không nằm trong tứ giác, trả về None.
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)  # Biến đổi điểm thành dạng mảng phù hợp với hàm OpenCV.
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)  # Áp dụng phép biến đổi phối cảnh.
        return transform_point.reshape(-1, 2)  # Trả về điểm đã được biến đổi về dạng mảng 2D.

    def add_transformed_position_to_tracks(self, tracks):  # Hàm này áp dụng phép biến đổi cho tất cả các vị trí trong tracks.
        for object, object_tracks in tracks.items():  # Duyệt qua từng đối tượng trong tracks.
            for frame_num, track in enumerate(object_tracks):  # Duyệt qua từng frame của track.
                for track_id, track_info in track.items():  # Duyệt qua từng track_id trong track.
                    position = track_info['position_adjusted']  # Lấy vị trí đã được điều chỉnh.
                    position = np.array(position)  # Chuyển vị trí thành mảng NumPy.
                    position_transformed = self.transform_point(position)  # Áp dụng phép biến đổi phối cảnh lên vị trí.
                    if position_transformed is not None:  # Nếu vị trí được biến đổi thành công.
                        position_transformed = position_transformed.squeeze().tolist()  # Biến đổi kết quả về dạng list.
                    # Cập nhật vị trí đã biến đổi vào track dưới khóa 'position_transformed'.
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
