from arcface_torch.backbones import get_model
import torch
import cv2
import numpy as np

class ArcfaceInference:
    def __init__(self, model_name='r50', model_weights_path='arcface_torch/work_dirs/ms1mv2_r50/model.pt'):
        self.model = get_model(name=model_name)
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.eval()

    @torch.no_grad()
    def inference(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.resize(img, (112, 112))
        else:
            img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)

        self.model.eval()
        feat = self.model(img).squeeze(axis=1).numpy()
        return feat
        

if __name__ == "__main__":
    arcface_module = ArcfaceInference()
    arcface_module.inference(img='../test_images/Katarina_167856652018987_round3_Annie_06-10-2021.mp4_15_1.jpg')
