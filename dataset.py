from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config
from torchvision import transforms

class MapDataset(Dataset):
    def __init__(self , root_dir):
        super().__init__()

        self.root_dir = root_dir

        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        
        img_file = self.list_files[index]

        img_path = os.path.join(self.root_dir , img_file)

        img = np.array(Image.open(img_path))

        input_img = img[: , int(img.shape[1]/2): , :]
        target_img = img[: , :int(img.shape[1]/2) , :]

        # augmentations = config.both_transform(image = input_img , image0 = target_img)
        # input_img , target_img = augmentations["image"] , augmentations["image0"]

        input_img = config.transform_only_input(image = input_img)["image"]
        target_img = config.transform_only_target(image = target_img)["image"]

        return input_img , target_img

if __name__ == "__main__":
    m = MapDataset("D:/PROGRAMMING/PYTHON/watermarkRemover/train2")
    i , t = m[1]

    input_img_pil = transforms.ToPILImage()(i*0.5 + 0.5)
    target_img_pil = transforms.ToPILImage()(t)

    target_img_pil.show(title="Input Image")