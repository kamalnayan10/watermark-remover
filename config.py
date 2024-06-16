import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-5
BATCH_SIZE = 16
NUM_WORKERS = 2
IMG_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
EPOCHS = 1000
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"


transform_only_input = A.Compose([
    A.Normalize(
        mean = [0.5,0.5,0.5],
        std = [0.5,0.5,0.5],
        max_pixel_value = 255.0,
        ),
    A.Resize(width = 512 , height = 512),
    A.HorizontalFlip(p = 0.5),
    ToTensorV2()
])


transform_only_target = A.Compose([
    A.Normalize(
        mean = [0.5,0.5,0.5],
        std = [0.5,0.5,0.5],
        max_pixel_value = 255.0
        ),
    A.Resize(width = 512 , height = 512),
    A.HorizontalFlip(p = 0.5),
    ToTensorV2()
])