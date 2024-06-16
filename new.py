import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torchvision import transforms
from utils import load_checkpoint  # Make sure load_checkpoint is correctly implemented

# Assuming DEVICE is defined somewhere in your config.py or utils.py
from config import DEVICE, LR, CHECKPOINT_GEN
from generalGen import Generator  # Assuming this is where your Generator class is defined


def create_img(gen, img_path , filename):
    # Define transformation (only to tensor in this case)
    transform = A.Compose([
        A.Normalize(
        mean = [0.5,0.5,0.5],
        std = [0.5,0.5,0.5],
        max_pixel_value = 255.0
        ),
        A.Resize(2048,2048),
        ToTensorV2()
    ])

    # Load and preprocess the input image
    original_size = Image.open(img_path).size
    
    img = np.array(Image.open(img_path))
    img = transform(image=img)["image"]
    img = img.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to DEVICE

    img = img.to(torch.float32).to(DEVICE)

    # Generate image
    with torch.no_grad():
        gen_img = gen(img)  # Generate the image using the Generator

    resize_transform = transforms.Resize(original_size[::-1])
    gen_img = resize_transform(gen_img)
    
    # Save the generated image
    save_image(gen_img*0.5 + 0.5, f"D:/PROGRAMMING/PYTHON/watermarkRemover/created/{filename}.jpg")  # Adjust the path as needed

    print("removed watermark")

# Example usage:
if __name__ == "__main__":
    
    gen = Generator(3).to(DEVICE)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LR)
    create_img(gen, "C:/Users/kamal/Downloads/val.jpg" , "hello")

