import cv2

def read_video(video_path):  # Hàm này nhận vào đường dẫn video và trả về danh sách các khung hình (frames).
    cap = cv2.VideoCapture(video_path)  # Mở video từ đường dẫn `video_path` để đọc.
    frames = []  # Tạo một danh sách rỗng để lưu trữ các khung hình video.
    
    while True:  # Vòng lặp để đọc từng khung hình trong video.
        ret, frame = cap.read()  # Đọc một khung hình từ video. ret sẽ là True nếu đọc thành công, frame là khung hình đó.
        if not ret:  # Nếu không đọc được khung hình (ret == False), tức là đã đến cuối video.
            break  # Thoát khỏi vòng lặp.
        frames.append(frame)  # Thêm khung hình vào danh sách `frames`.
    
    return frames  # Trả về danh sách các khung hình video.


def save_video(ouput_video_frames, output_video_path):  # Hàm này nhận vào danh sách khung hình và đường dẫn để lưu video.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Định dạng video codec sử dụng XVID.
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))  
    # Tạo đối tượng `VideoWriter` để lưu video với codec XVID, tốc độ 24 khung hình mỗi giây, kích thước video lấy từ khung hình đầu tiên.
    
    for frame in ouput_video_frames:  # Lặp qua từng khung hình trong danh sách `ouput_video_frames`.
        out.write(frame)  # Ghi khung hình vào video.
    
    out.release()  # Giải phóng tài nguyên video sau khi đã lưu xong.
