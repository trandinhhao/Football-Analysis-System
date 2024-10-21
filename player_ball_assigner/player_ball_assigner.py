import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70  # Ngưỡng tối đa để gán bóng cho người chơi, nếu khoảng cách lớn hơn 70 thì không gán bóng

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)  # Lấy vị trí tâm của quả bóng từ hộp giới hạn

        miniumum_distance = 99999  # Giá trị khởi tạo của khoảng cách nhỏ nhất, ban đầu rất lớn để dễ so sánh
        assigned_player = -1  # Ban đầu không có người chơi nào được gán bóng

        # Duyệt qua tất cả người chơi trong khung hình
        for player_id, player in players.items():
            player_bbox = player['bbox']  # Lấy hộp giới hạn của từng người chơi

            # Tính khoảng cách từ bóng đến các cạnh trái và phải của hộp giới hạn người chơi
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)  # Lấy khoảng cách nhỏ nhất từ hai cạnh trái/phải

            # Kiểm tra nếu khoảng cách nhỏ hơn ngưỡng tối đa và là khoảng cách ngắn nhất đã tìm thấy
            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance  # Cập nhật khoảng cách nhỏ nhất
                    assigned_player = player_id  # Gán bóng cho người chơi có khoảng cách ngắn nhất

        return assigned_player  # Trả về ID của người chơi được gán bóng, hoặc -1 nếu không ai gần bóng
