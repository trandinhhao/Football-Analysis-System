def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox  # Giải nén các giá trị từ bbox
    return int((x1+x2)/2),int((y1+y2)/2)  # Trả về tọa độ trung tâm của bbox

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]  # Trả về chiều rộng của bbox (x2 - x1)

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5  # Tính khoảng cách Euclide giữa hai điểm

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]  # Trả về khoảng cách theo trục x và y giữa hai điểm

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox  # Giải nén các giá trị từ bbox
    return int((x1+x2)/2),int(y2)  # Trả về tọa độ điểm chân (giữa cạnh dưới) của bbox