import os
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image
import cv2

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

def augment_images_in_folders(main_folder, output_folder, num_augmented_per_image=4):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    seq = iaa.Sequential([
        sometimes(
        iaa.Affine(rotate=(-25, 25)),  # Xoay ảnh trong khoảng -45 đến 45 độ
        ),
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        sometimes(
        iaa.EdgeDetect(alpha=0.5)
        ),
        sometimes(
        iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
        ),
        sometimes(
        iaa.Cutout(nb_iterations=2)
        ),
        sometimes(
        iaa.Crop(percent=(0, 0.2)),  
        ),
        sometimes(
        iaa.Dropout(p=(0, 0.1)),  
        ),
        sometimes(
        iaa.AddToHueAndSaturation((-20, 20)),  
        ),
        sometimes(
        iaa.GaussianBlur(sigma=(0, 3.0))
        ),
    ])

    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)

        # Kiểm tra xem nó có phải là thư mục không
        if os.path.isdir(folder_path):
            # Lấy đường dẫn của ảnh trong thư mục con
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

            
            for img in image_files:
                image_filename = img
                image_path = os.path.join(folder_path, image_filename)
                img = cv2.imread(str(image_path))

                ori_file = f"{os.path.splitext(image_filename)[0]}.png"
                ori = os.path.join(output_folder, folder_name, ori_file)
                cv2.imwrite(ori, img)
                
                for i in range(num_augmented_per_image):
                    
                    augmented_img = seq.augment_image(img)
                    new_filename = f"{os.path.splitext(image_filename)[0]}_augmented_{i+1}.png"
                    new_path = os.path.join(output_folder, folder_name, new_filename)

                    
                    if not os.path.exists(os.path.join(output_folder, folder_name)):
                        os.makedirs(os.path.join(output_folder, folder_name))
                        
                    cv2.imwrite(new_path, augmented_img)

# Thư mục input
main_folder_path = "./Data/Champions"

# Thư mục output
output_folder_path = "./Data/aug_Champions"

# Số lượng ảnh tăng cường cho mỗi ảnh gốc
num_augmented_per_image = 20

augment_images_in_folders(main_folder_path, output_folder_path, num_augmented_per_image)
