from face_emb import ArcfaceInference
from box_detect import CircleDetector
import numpy as np
import faiss
import os
import cv2
import argparse

def build_faiss_index(emb_model, path_folder):
    # Get a list of all subdirectories in the folder
    subdirectories = [d for d in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, d))]

    # Initialize Faiss index
    index_dimension = 512  # Change this dimension based on your face embedding model
    index = faiss.IndexFlatIP(index_dimension)

    # Lists to store face embeddings and corresponding file names
    embeddings = []
    image_names = []

    # Iterate through subdirectories
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(path_folder, subdirectory)
        
        # Get a list of all image files in the subdirectory
        image_files = [f for f in os.listdir(subdirectory_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Iterate through image files in the subdirectory
        for image_file in image_files:
            image_path = os.path.join(subdirectory_path, image_file)
            embedding = emb_model.inference(image_path)
            faiss.normalize_L2(embedding)
            index.add(embedding)
            
            # Save face embedding and corresponding file name (using the subdirectory name)
            embeddings.append(embedding)
            image_names.append(subdirectory)

    # Convert the lists to NumPy arrays for Faiss
    face_embeddings = np.vstack(embeddings)
    
    # Save the image names and face embeddings for later reference
    index_info = {"image_names": image_names, "embeddings": embeddings}
    return index, index_info

    
def relatedness_fn(index_build, query_embedding,  k=1):
        faiss.normalize_L2(query_embedding)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        # Use Faiss to find the k-nearest neighbors
        D, I = index_build.search(query_embedding.reshape(1, -1), k)
        return D[0], I[0]


def main():
    parser = argparse.ArgumentParser(description="Process images and calculate relatedness.")
    parser.add_argument("--folder_path", default='./test_images', help="Path to the folder containing image files.")
    args = parser.parse_args()

    circle_detector = CircleDetector()
    face_emb = ArcfaceInference()

    FaceBank = './Data/Champions/'  # Đặt đường dẫn đến thư mục chứa các tệp hình ảnh (có thể thay đổi)

    index, index_info = build_faiss_index(face_emb, FaceBank)

    output_file = 'test.txt'  # Specify the name of the output text file

    with open(output_file, 'w') as file:
        for filename in os.listdir(args.folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Chọn chỉ các tệp hình ảnh
                image_path = os.path.join(args.folder_path, filename)

                # Thực hiện các bước xử lý cho mỗi hình ảnh trong thư mục
                img1 = circle_detector.detect_circles(image_path)
                emb1 = face_emb.inference(img1)
                cos, ind = relatedness_fn(index, emb1)

                # Ghi kết quả vào file
                result_line = f"{filename}\t{index_info['image_names'][ind[0]]}\n"
                file.write(result_line)

                # In kết quả (tùy chọn)
                print(result_line.strip())

if __name__ == "__main__":
    main()
