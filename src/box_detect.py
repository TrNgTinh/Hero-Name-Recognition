import cv2
import numpy as np
import os

class CircleDetector:
    def __init__(self):
        pass
        
            
    @staticmethod
    def detect_circles(image_path):
        # Đọc ảnh từ file
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        image = image[:, :width // 2, :]
        # Chuyển đổi ảnh sang ảnh xám để giảm chiều sâu màu
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Áp dụng GaussianBlur để giảm nhiễu và làm tăng chất lượng detect
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Sử dụng Hough Circle Transform để detect các đường tròn trong ảnh
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=20,
            param2=20,
            minRadius=10,
            maxRadius=300
        )

        bounding_boxes = []

        # Nếu có các đường tròn được detect
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])  # Tọa độ tâm
                radius = i[2]  # Bán kính

                # Tính toán tọa độ của bounding box
                x_min = center[0] - radius
                y_min = center[1] - radius
                x_max = center[0] + radius
                y_max = center[1] + radius

                bounding_boxes.append(((int(x_min), int(y_min)), (int(x_max), int(y_max))))
                
                image = image[max(0,y_min - 3) :min(y_max + 3,width // 2) , max(0,x_min - 3):min(x_max + 3, height), :]
                return image
        else:
            print("zo day")
            return image
                              
                
def process_folder(input_folder, output_folder, circle_detect):
        # Đảm bảo thư mục đầu ra tồn tại
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Lấy danh sách tất cả các tệp trong thư mục đầu vào
        files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

        # Lặp qua từng tệp và xử lý
        for file in files:
            input_path = os.path.join(input_folder, file)
            output_name = file.split('_')[0]  # Lấy tên trước dấu _
            output_path = os.path.join(output_folder,output_name, f"{output_name}_{file.split('_')[1]}.png")
            # Thực hiện detect và lưu vào tệp mới
            try:
                processed_image = circle_detect.detect_circles(input_path)
                cv2.imwrite(output_path, processed_image)    
            except:
                pass

#circle_detect = CircleDetector()
#input_folder = '../test_images/'
#output_folder = 'Champions/'
#process_folder(input_folder, output_folder, circle_detect)

