import torch
from utils import *
import torch.nn as nn
import torch.optim as optim
from config import *
from dataset import MapDataset
# from generator import Generator
from generalGen import Generator
from discriminator import Dicriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from new import create_img


def train(disc , gen , loader , opt_disc , opt_gen , l1 , bce , g_scaler , d_scaler):
    loop = tqdm(loader , leave=True)

    for idx , (x,y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            d_real = disc(x,y)
            d_fake = disc(x,y_fake.detach())

            d_real_loss = bce(d_real , torch.ones_like(d_real))
            d_fake_loss = bce(d_fake , torch.zeros_like(d_fake))

            d_loss = (d_real_loss + d_fake_loss)/2

        disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            d_fake = disc(x , y_fake)
            g_fake_loss = bce(d_fake , torch.ones_like(d_fake))
            L1 = l1(y_fake , y)* L1_LAMBDA

            g_loss = g_fake_loss + L1
        
        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


def main():
    disc = Dicriminator(3).to(DEVICE)
    gen = Generator(3).to(DEVICE)

    opt_disc = optim.Adam(params=disc.parameters() , lr = LR , betas=(0.5 , 0.999))
    opt_gen = optim.Adam(params=gen.parameters() , lr = LR , betas=(0.5 , 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_GEN , gen , opt_gen , LR)
        load_checkpoint(CHECKPOINT_DISC , disc , opt_disc , LR)

    train_dataset = MapDataset(root_dir="train2")

    train_loader = DataLoader(train_dataset , batch_size=BATCH_SIZE , shuffle=True , num_workers= NUM_WORKERS)
    val_loader = DataLoader(train_dataset , batch_size=1 , shuffle=True , num_workers= NUM_WORKERS)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        train(disc , gen , train_loader , opt_disc , opt_gen , L1_LOSS , BCE , g_scaler , d_scaler)

        if epoch % 5 == 0 and SAVE_MODEL:
            save_checkpoint(gen , opt_gen , CHECKPOINT_GEN) 
            save_checkpoint(disc , opt_disc , CHECKPOINT_DISC) 

        if epoch % 10 == 0:
            save_some_examples(gen , val_loader , epoch , "evaluation")
            create_img(gen,"C:/Users/kamal/Downloads/battlecreek-coffee-roasters-rsnzc-8dVs0-unsplash.jpg"
                       , f"created_{epoch}")



if __name__ == "__main__":
    main()