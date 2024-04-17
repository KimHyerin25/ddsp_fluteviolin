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

class SpecModel(nn.Module):
  def __init__(self, n_fft, hop_length):
    super().__init__()
    self.spec_converter = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    self.db_converter = torchaudio.transforms.AmplitudeToDB(stype='power')

  def forward(self, audio_sample):
    spec = self.spec_converter(audio_sample)
    db_spec = self.db_converter(spec)
    return db_spec

class Conv2dNormPool(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    self.batch_norm = nn.BatchNorm2d(out_channels)
    self.activation = nn.LeakyReLU(0.1)
    
  def forward(self, x):
    x = self.conv(x)
    x = self.batch_norm(x)
    x = self.activation(x)
    return x
  
class Conv2dNormTransposePool(Conv2dNormPool):
  def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
    super().__init__(in_channels, out_channels, kernel_size, padding, stride)
    self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

  
class AutoEncoder(nn.Module):
  def __init__(self, n_fft, hop_length, hidden_size=256):
    super().__init__()
    self.spec_model = SpecModel(n_fft, hop_length)
    self.encoder = nn.Sequential()
    self.pitch_embedder = nn.Embedding(121, hidden_size//2)
    self.num_channels = [1] + [128] * 3 + [256] * 3 + [512] * 2 + [1024]
    i = 0
    self.encoder.add_module(f"conv_norm{i}", Conv2dNormPool(self.num_channels[i], self.num_channels[i+1], (5,5), 2, (2,2) ))
    for i in range(1,7):
      self.encoder.add_module(f"conv_norm{i}", Conv2dNormPool(self.num_channels[i], self.num_channels[i+1], (4,4), 1, (2,2) ))
    i = 7
    self.encoder.add_module(f"conv_norm{i}", Conv2dNormPool(self.num_channels[i], self.num_channels[i+1], (2,2), 0, (2,2) ))
    i = 8
    self.encoder.add_module(f"conv_norm{i}", Conv2dNormPool(self.num_channels[i], self.num_channels[i+1], (1,1), 0, (1,1) ))

    self.final_layer = nn.Linear(hidden_size * 2, hidden_size) 
    self.decoder = nn.Sequential(      
        Conv2dNormTransposePool(in_channels=self.num_channels[-1] + hidden_size//2, out_channels=self.num_channels[-2], kernel_size=(2,1), padding=0, stride=(2,2))
    )
    i = 0
    self.decoder.add_module(f"conv_norm{i}", Conv2dNormTransposePool(self.num_channels[-2-i], self.num_channels[-3-i], (2,2), 0, (2,2)))
    for i in range(1,7):
      self.decoder.add_module(f"conv_norm{i}", Conv2dNormTransposePool(self.num_channels[-2-i], self.num_channels[-3-i], (4,4), 1, (2,2)))
    self.decoder.add_module("final_module",  nn.ConvTranspose2d(in_channels=self.num_channels[1], out_channels=1, kernel_size=(4,4), padding=1, stride=(2,2)),)


  def forward(self, x, pitch):
    spec = self.spec_model(x)
    spec = spec[:,:,:-1, :] # to match 512
    spec /= 80
    spec = nn.functional.pad(spec, (2,3), value=torch.min(spec))
    out = self.encoder(spec)
    # print(out.shape, out.view(out.shape[1]//2, -1).shape)
    latent = self.final_layer(out.view(-1, out.shape[1]//2))
    latent = torch.cat([latent, self.pitch_embedder(pitch)], dim=-1)
    latent = latent.view(latent.shape[0], -1, 1, 1)
    recon_spec = self.decoder(latent)
    return recon_spec, spec