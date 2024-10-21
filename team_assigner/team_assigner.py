from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # Dictionary lưu màu sắc đại diện cho mỗi đội
        self.player_team_dict = {}  # Dictionary lưu đội của từng cầu thủ

    def get_clustering_model(self, image):
        # Chuyển đổi hình ảnh thành mảng 2D (mỗi pixel là một điểm dữ liệu với 3 giá trị màu)
        image_2d = image.reshape(-1, 3)

        # Sử dụng K-means để phân cụm với 2 cụm (2 đội)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Lấy phần ảnh trong bounding box của cầu thủ
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Chỉ lấy nửa trên của bounding box để xác định màu áo cầu thủ
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Áp dụng K-means để phân cụm màu sắc trong vùng ảnh
        kmeans = self.get_clustering_model(top_half_image)

        # Lấy nhãn cụm cho mỗi pixel
        labels = kmeans.labels_

        # Định hình lại nhãn thành kích thước của ảnh
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Xác định cụm đại diện cho màu nền (cụm không phải của cầu thủ)
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # Cụm còn lại đại diện cho cầu thủ
        player_cluster = 1 - non_player_cluster

        # Lấy màu của cầu thủ từ cụm trung tâm
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        # Lưu màu của từng cầu thủ
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Phân cụm màu sắc của các cầu thủ thành 2 đội bằng K-means
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        # Lưu kết quả K-means và màu sắc đại diện cho mỗi đội
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # Kiểm tra xem cầu thủ đã được gán đội chưa
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Lấy màu của cầu thủ từ bounding box
        player_color = self.get_player_color(frame, player_bbox)

        # Dự đoán đội của cầu thủ dựa trên màu sắc
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Tăng giá trị team_id lên 1 để có đội 1 và đội 2

        # Quy định cầu thủ có id 91 luôn thuộc đội 1
        if player_id == 91:
            team_id = 1

        # Lưu thông tin về đội của cầu thủ
        self.player_team_dict[player_id] = team_id

        return team_id
