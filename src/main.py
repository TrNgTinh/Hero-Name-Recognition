from face_emb import ArcfaceInference
from box_detect import CircleDetector
import numpy as np
import faiss
import os
import cv2
from clip import CLIPImageSimilarity
from train_mobilenet import MobileNetV3Inference, ResNetInference


def build_faiss_index(emb_model, path_folder):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(path_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Initialize Faiss index
    index_dimension = 512  # Change this dimension based on your face embedding model
    index = faiss.IndexFlatIP(index_dimension)

    # Lists to store face embeddings and corresponding file names
    embeddings = []
    image_names = []

    # Extract face embeddings and add to Faiss index
    for image_file in image_files:
        image_path = os.path.join(path_folder, image_file)
        embedding = emb_model.inference(image_path)
        faiss.normalize_L2(embedding)
        index.add(embedding)
        # Save face embedding and corresponding file name
        embeddings.append(embedding)
        image_names.append(image_file)

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


'''
mobile_net_infer =  ResNetInference()
#cicle_detector = CircleDetector()
#path_1 = ('../test_images/Amumu_142330807963265_round2_Amumu_06-14-2021.mp4_24_0.jpg')
#img1 = cicle_detector.detect_circles(path_1)
#cv2.imwrite('test.png',img1)
print(mobile_net_infer.predict_single_image('./aug_Champions/Akali/Akali_augmented_1.png'))

clip = CLIPImageSimilarity()

index, index_info = build_faiss_index(clip, '../Champions/')

img1 = cicle_detector.detect_circles(path_1)
cv2.imwrite('test.jpg',img1)
emb1 = clip.calculate_emb(img1)
cos, index = relatedness_fn(index, emb1)
print(cos, index)

print(index_info['image_names'][index[0]])
print(index_info['image_names'][index[1]])
print(index_info['image_names'][index[2]])


'''
cicle_detector = CircleDetector()
face_emb = ArcfaceInference()

def distance(vector1,vector2):
    sim = vector1 @ vector2.T
    sim = sim / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return sim
    
folder_path = '../test_images'  # Đặt đường dẫn đến thư mục chứa các tệp hình ảnh

index, index_info = build_faiss_index(face_emb, '../Champions/')

for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Chọn chỉ các tệp hình ảnh
        image_path = os.path.join(folder_path, filename)

        # Thực hiện các bước xử lý cho mỗi hình ảnh trong thư mục
        print(image_path)
        img1 = cicle_detector.detect_circles(image_path)
        emb1 = face_emb.inference(img1)
        cos, ind = relatedness_fn(index, emb1)

        # In kết quả
        print(f"Image: {filename}, Cosine Similarity: {cos}, Related Image: {index_info['image_names'][ind[0]]}")


