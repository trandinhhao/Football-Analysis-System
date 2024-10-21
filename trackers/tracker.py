from ultralytics import YOLO  # Import mô hình YOLO từ thư viện ultralytics
import supervision as sv  # Import thư viện supervision với tên viết tắt là sv
import pickle  # Import module pickle để xử lý dữ liệu dạng byte
import os  # Import module os để thao tác với hệ thống tệp
import numpy as np  # Import thư viện numpy với tên viết tắt là np
import pandas as pd  # Import thư viện pandas với tên viết tắt là pd
import cv2  # Import thư viện OpenCV
import sys  # Import module sys
sys.path.append('../')  # Thêm thư mục cha vào đường dẫn tìm kiếm module
from utils import get_center_of_bbox, get_bbox_width, get_foot_position  # Import các hàm tiện ích

class Tracker:  # Định nghĩa lớp Tracker
    def __init__(self, model_path):  # Hàm khởi tạo với đường dẫn mô hình
        self.model = YOLO(model_path)  # Tạo đối tượng YOLO với mô hình được chỉ định
        self.tracker = sv.ByteTrack()  # Tạo đối tượng ByteTrack từ thư viện supervision

    def add_position_to_tracks(sekf,tracks):  # Phương thức thêm vị trí vào các track
        for object, object_tracks in tracks.items():  # Lặp qua từng đối tượng và track của nó
            for frame_num, track in enumerate(object_tracks):  # Lặp qua từng frame và track
                for track_id, track_info in track.items():  # Lặp qua từng track_id và thông tin track
                    bbox = track_info['bbox']  # Lấy bounding box của track
                    if object == 'ball':  # Nếu đối tượng là quả bóng
                        position= get_center_of_bbox(bbox)  # Lấy vị trí trung tâm của bounding box
                    else:  # Nếu không phải quả bóng
                        position = get_foot_position(bbox)  # Lấy vị trí chân của đối tượng
                    tracks[object][frame_num][track_id]['position'] = position  # Thêm vị trí vào thông tin track

    def interpolate_ball_positions(self,ball_positions):  # Phương thức nội suy vị trí bóng
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]  # Lấy danh sách các bounding box của bóng
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])  # Tạo DataFrame từ danh sách vị trí bóng

        # Nội suy các giá trị bị thiếu
        df_ball_positions = df_ball_positions.interpolate()  # Nội suy các giá trị bị thiếu
        df_ball_positions = df_ball_positions.bfill()  # Điền các giá trị còn thiếu bằng phương pháp backward fill

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]  # Chuyển đổi DataFrame thành danh sách các dictionary

        return ball_positions  # Trả về danh sách vị trí bóng đã được nội suy

    def detect_frames(self, frames):  # Phương thức phát hiện đối tượng trong các frame
        batch_size=20  # Kích thước batch
        detections = []  # Danh sách lưu kết quả phát hiện
        for i in range(0,len(frames),batch_size):  # Lặp qua các frame theo batch
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)  # Dự đoán đối tượng trong batch frame
            detections += detections_batch  # Thêm kết quả dự đoán vào danh sách detections
        return detections  # Trả về danh sách các kết quả phát hiện

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):  # Phương thức lấy tracks của các đối tượng
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):  # Nếu đọc từ stub và đường dẫn stub tồn tại
            with open(stub_path,'rb') as f:  # Mở file stub
                tracks = pickle.load(f)  # Đọc dữ liệu tracks từ file
            return tracks  # Trả về tracks đã đọc

        detections = self.detect_frames(frames)  # Phát hiện đối tượng trong các frame

        tracks={  # Khởi tạo dictionary lưu tracks cho các loại đối tượng
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):  # Lặp qua các kết quả phát hiện
            cls_names = detection.names  # Lấy tên các lớp
            cls_names_inv = {v:k for k,v in cls_names.items()}  # Tạo dictionary ngược của tên lớp

            # Chuyển đổi sang định dạng Detection của supervision
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Chuyển đổi GoalKeeper thành đối tượng player
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Theo dõi các đối tượng
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})  # Thêm dictionary rỗng cho players trong frame hiện tại
            tracks["referees"].append({})  # Thêm dictionary rỗng cho referees trong frame hiện tại
            tracks["ball"].append({})  # Thêm dictionary rỗng cho ball trong frame hiện tại

            for frame_detection in detection_with_tracks:  # Lặp qua các phát hiện có track
                bbox = frame_detection[0].tolist()  # Lấy bounding box
                cls_id = frame_detection[3]  # Lấy ID lớp
                track_id = frame_detection[4]  # Lấy ID track

                if cls_id == cls_names_inv['player']:  # Nếu là player
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}  # Thêm thông tin bbox vào tracks của player
                
                if cls_id == cls_names_inv['referee']:  # Nếu là referee
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}  # Thêm thông tin bbox vào tracks của referee
            
            for frame_detection in detection_supervision:  # Lặp qua các phát hiện
                bbox = frame_detection[0].tolist()  # Lấy bounding box
                cls_id = frame_detection[3]  # Lấy ID lớp

                if cls_id == cls_names_inv['ball']:  # Nếu là ball
                    tracks["ball"][frame_num][1] = {"bbox":bbox}  # Thêm thông tin bbox vào tracks của ball

        if stub_path is not None:  # Nếu có đường dẫn stub
            with open(stub_path,'wb') as f:  # Mở file stub để ghi
                pickle.dump(tracks,f)  # Lưu tracks vào file stub

        return tracks  # Trả về tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):  # Phương thức vẽ ellipse
        y2 = int(bbox[3])  # Lấy tọa độ y dưới cùng của bbox
        x_center, _ = get_center_of_bbox(bbox)  # Lấy tọa độ x trung tâm của bbox
        width = get_bbox_width(bbox)  # Lấy chiều rộng của bbox

        cv2.ellipse(  # Vẽ ellipse
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40  # Chiều rộng hình chữ nhật
        rectangle_height=20  # Chiều cao hình chữ nhật
        x1_rect = x_center - rectangle_width//2  # Tọa độ x trái của hình chữ nhật
        x2_rect = x_center + rectangle_width//2  # Tọa độ x phải của hình chữ nhật
        y1_rect = (y2- rectangle_height//2) +15  # Tọa độ y trên của hình chữ nhật
        y2_rect = (y2+ rectangle_height//2) +15  # Tọa độ y dưới của hình chữ nhật

        if track_id is not None:  # Nếu có track_id
            cv2.rectangle(frame,  # Vẽ hình chữ nhật
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12  # Tọa độ x của văn bản
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(  # Vẽ văn bản (track_id)
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame  # Trả về frame đã vẽ

    def draw_traingle(self,frame,bbox,color):  # Phương thức vẽ tam giác
        y= int(bbox[1])  # Lấy tọa độ y trên cùng của bbox
        x,_ = get_center_of_bbox(bbox)  # Lấy tọa độ x trung tâm của bbox

        triangle_points = np.array([  # Tạo mảng các điểm của tam giác
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)  # Vẽ tam giác đặc
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)  # Vẽ viền tam giác

        return frame  # Trả về frame đã vẽ

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):  # Phương thức vẽ thông tin kiểm soát bóng của đội
        # Vẽ một hình chữ nhật bán trong suốt
        overlay = frame.copy()  # Tạo bản sao của frame
        cv2.rectangle(overlay, (1250, 850), (1900,970), (255,255,255), -1 )  # Vẽ hình chữ nhật trắng
        alpha = 0.4  # Độ trong suốt
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Kết hợp overlay với frame gốc

        team_ball_control_till_frame = team_ball_control[:frame_num+1]  # Lấy thông tin kiểm soát bóng đến frame hiện tại
        # Đếm số frame mỗi đội kiểm soát bóng
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)  # Tính tỷ lệ kiểm soát bóng của đội 1
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)  # Tính tỷ lệ kiểm soát bóng của đội 2

        cv2.putText(frame, f"Ti le kiem soat bong doi 1: {team_1*100:.2f}%",(1300,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)  # Vẽ text thông tin kiểm soát bóng đội 1
        cv2.putText(frame, f"Ti le kiem soat bong doi 2: {team_2*100:.2f}%",(1300,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)  # Vẽ text thông tin kiểm soát bóng đội 2

        return frame  # Trả về frame đã vẽ

    def draw_annotations(self,video_frames, tracks,team_ball_control):  # Phương thức vẽ chú thích lên video
        output_video_frames= []  # Danh sách lưu các frame đã vẽ chú thích
        for frame_num, frame in enumerate(video_frames):  # Lặp qua từng frame trong video
            frame = frame.copy()  # Tạo bản sao của frame

            player_dict = tracks["players"][frame_num]  # Lấy thông tin tracks của players trong frame
            ball_dict = tracks["ball"][frame_num]  # Lấy thông tin tracks của ball trong frame
            referee_dict = tracks["referees"][frame_num]  # Lấy thông tin tracks của referees trong frame

            # Vẽ Players
            for track_id, player in player_dict.items():  # Lặp qua từng player
                color = player.get("team_color",(0,0,255))  # Lấy màu của đội
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)  # Vẽ ellipse cho player

                if player.get('has_ball',False):  # Nếu player đang có bóng
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))  # Vẽ tam giác trên đầu player

            # Vẽ Referee
            for _, referee in referee_dict.items():  # Lặp qua từng referee
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))  # Vẽ ellipse cho referee
            
            # Vẽ ball 
            for track_id, ball in ball_dict.items():  # Lặp qua ball (thường chỉ có 1)
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))  # Vẽ tam giác cho ball


            # Vẽ thông tin kiểm soát bóng của đội
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)  # Vẽ thông tin kiểm soát bóng

            output_video_frames.append(frame)  # Thêm frame đã vẽ vào danh sách output

        return output_video_frames  # Trả về danh sách các frame đã vẽ chú thích