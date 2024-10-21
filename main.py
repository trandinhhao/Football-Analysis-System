from utils import read_video, save_video  # Import hàm đọc và lưu video từ file utils
from trackers import Tracker  # Import class Tracker để theo dõi các đối tượng trong video
import cv2  # Thư viện OpenCV để xử lý video và hình ảnh
import numpy as np  # Thư viện NumPy để xử lý mảng số học
from team_assigner import TeamAssigner  # Import class để gán đội cho người chơi
from player_ball_assigner import PlayerBallAssigner  # Import class gán bóng cho người chơi
from camera_movement_estimator import CameraMovementEstimator  # Import class để ước lượng chuyển động của camera
from view_transformer import ViewTransformer  # Import class để biến đổi góc nhìn
from speed_and_distance_estimator import SpeedAndDistance_Estimator  # Import class ước lượng tốc độ và khoảng cách

def main():
    # Đọc video từ file
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Khởi tạo Tracker để theo dõi đối tượng
    tracker = Tracker('models/best.pt')

    # Lấy thông tin các đối tượng được theo dõi, đọc từ file stub để tiết kiệm thời gian xử lý
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    # Lấy vị trí của các đối tượng từ thông tin theo dõi
    tracker.add_position_to_tracks(tracks)

    # Khởi tạo ước lượng chuyển động của camera dựa trên frame đầu tiên
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    
    # Lấy chuyển động của camera cho mỗi frame, đọc từ file stub để tiết kiệm thời gian
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
    
    # Thêm thông tin điều chỉnh vị trí của các đối tượng dựa trên chuyển động của camera
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Khởi tạo ViewTransformer để biến đổi vị trí các đối tượng sang hệ tọa độ khác
    view_transformer = ViewTransformer()
    
    # Thêm vị trí đã biến đổi cho các đối tượng vào danh sách tracks
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Nội suy vị trí bóng dựa trên thông tin đã theo dõi
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Khởi tạo ước lượng tốc độ và khoảng cách
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    
    # Thêm thông tin về tốc độ và khoảng cách của các đối tượng vào danh sách tracks
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Khởi tạo TeamAssigner để gán màu đội cho người chơi
    team_assigner = TeamAssigner()
    
    # Gán màu đội cho người chơi đầu tiên dựa trên frame đầu tiên
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # Duyệt qua từng frame và gán đội cho từng người chơi
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Xác định đội của người chơi dựa trên bounding box của họ
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            # Gán thông tin đội và màu đội cho người chơi
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Khởi tạo PlayerBallAssigner để gán bóng cho người chơi
    player_assigner = PlayerBallAssigner()
    
    # Danh sách kiểm soát bóng của đội
    team_ball_control= []
    
    # Duyệt qua từng frame và gán bóng cho người chơi gần nhất
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']  # Lấy bounding box của bóng
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)  # Gán bóng cho người chơi

        if assigned_player != -1:  # Nếu có người chơi được gán bóng
            tracks['players'][frame_num][assigned_player]['has_ball'] = True  # Đánh dấu người chơi có bóng
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])  # Ghi nhận đội có bóng
        else:
            team_ball_control.append(team_ball_control[-1])  # Nếu không có người chơi mới, giữ nguyên đội có bóng từ frame trước
    
    # Chuyển team_ball_control thành mảng NumPy
    team_ball_control= np.array(team_ball_control)
    
    ########################################################################################### NEW
    
    unique_teams, team_counts = np.unique(team_ball_control, return_counts=True)
    team_control_summary = dict(zip(unique_teams, team_counts))
    
    # Tính tổng số lần kiểm soát bóng
    total_controls = sum(team_counts)

    # Ghi kết quả vào file thong_ke_so_lieu_sau_tran_dau.txt
    with open('output_videos/thong_ke_so_lieu_sau_tran_dau.txt', 'w', encoding='utf-8') as f:
        f.write("Bảng thống kê kiểm soát bóng của các đội:\n")
        for team, count in team_control_summary.items():
            percentage = (count / total_controls) * 100  # Tính tỉ lệ phần trăm
            f.write(f"Đội {team}: {count} lần kiểm soát bóng (Chiếm tỉ lệ {percentage:.2f}%)\n")
        
        # Ghi kết luận vào file
        if len(unique_teams) > 1:
            conclusion = f"=> Qua đó, ta thấy đội {unique_teams[0]} đang thể hiện tốt hơn đội {unique_teams[1]}."
            f.write(conclusion + "\n")

    
    ###############################################################################################

    # Vẽ thông tin theo dõi đối tượng lên video
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Vẽ chuyển động của camera lên video
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Vẽ thông tin về tốc độ và khoảng cách lên video
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Lưu video đã xử lý ra file
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()  # Chạy chương trình chính
