import cv2
import sys 
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5  # Số lượng khung hình để lấy mẫu khi tính toán tốc độ
        self.frame_rate = 24    # Tốc độ khung hình của video (fps)

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}  # Dictionary lưu tổng quãng đường của các đối tượng

        # Duyệt qua từng đối tượng và track của nó
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue  # Bỏ qua bóng và trọng tài trong quá trình tính toán

            number_of_frames = len(object_tracks)  # Tổng số khung hình của đối tượng

            # Tính toán tốc độ và quãng đường cho từng track qua các khung hình
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)  # Xác định khung hình cuối trong cửa sổ khung hình

                # Duyệt qua từng đối tượng trong khung hình
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue  # Bỏ qua nếu track không tồn tại trong khung hình cuối

                    # Lấy vị trí ban đầu và vị trí kết thúc đã được biến đổi
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue  # Bỏ qua nếu không có vị trí

                    # Tính khoảng cách và tốc độ
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate  # Tính thời gian đã trôi qua giữa các khung hình
                    speed_meteres_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meteres_per_second * 3.6  # Đổi tốc độ từ m/s sang km/h

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered  # Cập nhật tổng quãng đường của đối tượng

                    # Gán tốc độ và quãng đường cho từng khung hình trong cửa sổ khung hình hiện tại
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour  # Gán tốc độ
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]  # Gán quãng đường

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []

        # Duyệt qua từng khung hình của video
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue  # Bỏ qua bóng và trọng tài khi vẽ tốc độ và quãng đường

                # Duyệt qua từng track của đối tượng trong khung hình
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)  # Lấy tốc độ từ track
                        distance = track_info.get('distance', None)  # Lấy quãng đường từ track

                        if speed is None or distance is None:
                            continue  # Bỏ qua nếu không có dữ liệu về tốc độ hoặc quãng đường

                        bbox = track_info['bbox']  # Lấy hộp giới hạn của đối tượng
                        position = get_foot_position(bbox)  # Lấy vị trí của chân đối tượng từ hộp giới hạn
                        position = list(position)
                        position[1] += 40  # Điều chỉnh vị trí để in thông tin phía dưới đối tượng

                        position = tuple(map(int, position))
                        # Vẽ thông tin tốc độ lên khung hình
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        # Vẽ thông tin quãng đường lên khung hình
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            output_frames.append(frame)  # Thêm khung hình đã vẽ vào danh sách output

        return output_frames  # Trả về danh sách các khung hình đã được vẽ thông tin
