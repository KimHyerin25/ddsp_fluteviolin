import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
import pandas as pd
import IPython.display as ipd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import yaml

from model import AutoEncoder
from dataset import MusicNetDataSet
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data_dir')
args = parser.parse_args()
config = yaml.safe_load(open('config.yaml'))

cur_time = time.strftime("%Y%m%d-%H%M%S")
class WeightedSpecLoss:
  def __init__(self, fft_size=1024, sr=16000, device='cuda'):
    self.weight = torch.ones(fft_size//2).to(device)
    self.weight[:fft_size//4] = torch.linspace(10,1,fft_size//4)

  def __call__(self, pred, target):
    mse = (pred-target)**2
    mse *= self.weight[:, None]
    return mse.mean()
  

num_epochs = config["model"]["num_epochs"]
device = 'cuda'
model = AutoEncoder(**config["autoencoder"])
model.to(device)

train_dataset = MusicNetDataSet(args.data_dir)
train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=config["model"]["shuffle"], num_workers=config["model"]["num_workers"])
loss_calculator = WeightedSpecLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["lr"])
for epoch in tqdm(range(num_epochs)):
  for batch in train_loader:
    audio, pitch = batch
    recon_spec, spec = model(audio.to(device), pitch.to(device))
    loss = loss_calculator(recon_spec, spec)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
# mkdir if no exists /weights

Path('weights').mkdir(exist_ok=True)
torch.save(model.state_dict(), f'./autoencoder/weights/autoencoder_{cur_time}.pt')
