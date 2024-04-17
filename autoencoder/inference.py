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

from autoencoder.model import AutoEncoder
from autoencoder.dataset import MusicNetDataSet
import time
import argparse
cur_time = time.strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('--pt_dir', type=str, default='./autoencoder')
parser.add_argument('--test_dir', type=str, default='data_dir')
# 제일 최근 weight 저장한 것 불러오기
args = parser.parse_args()

config = yaml.safe_load(open('config.yaml'))

model = AutoEncoder(**config["autoencoder"])
weight_path = sorted(Path(args.pt_dir).rglob('*.pt'))[-1]

def network_output_to_audio(spec): #GriffenLim 추가
  rescaled_spec = spec * 80
  padded_spec = nn.functional.pad(rescaled_spec, (0,0, 0,1), value=-100)
  magnitude_spec = torchaudio.functional.DB_to_amplitude(padded_spec, ref=1, power=1)
  griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256, n_iter=100)
  spec_recon_audio = griffin_lim(magnitude_spec)
  return spec_recon_audio

def get_embedding(model, x):
  with torch.inference_mode():
    spec = model.spec_model(x)
    spec = spec[:,:,:-1] # to match 512
    spec /= 80
    spec = nn.functional.pad(spec, (2,3), value=torch.min(spec))
    out = model.encoder(spec)

    latent = model.final_layer(out.view(out.shape[0], -1))
  return latent

def decoding(model, latent, pitch):
  if isinstance(pitch, int):
    pitch = torch.tensor([pitch], dtype=torch.long)
  with torch.inference_mode():
    latent = torch.cat([latent, model.pitch_embedder(pitch)], dim=-1)
    latent = latent.view(latent.shape[0], -1, 1, 1)
    recon_spec = model.decoder(latent)
  return recon_spec


# args.pt_dir 없으면 weight_path로 load
if args.pt_dir:
    pretrained_weights = model.load_state_dict(torch.load(args.pt_dir))
else:
    pretrained_weights = model.load_state_dict(torch.load(weight_path))

model.load_state_dict(pretrained_weights)
model.eval()

test_loader = DataLoader(args.test_dir, batch_size=config["model"]["batch_size"], shuffle=config["model"]["shuffle"], num_workers=config["model"]["num_workers"])
test_batch = next(iter(test_loader))
audio, pitch = test_batch
with torch.no_grad():
  recon_spec, spec = model(audio, pitch)
recon_audio = network_output_to_audio(recon_spec[1])

idx1 = config["inference"]["idx_1"]
idx2 = config["inference"]["idx_2"]
audio, pitch = test_batch

sound_a = audio[idx1:idx1+1] 
sound_b = audio[idx2:idx2+1]
embedding_a = get_embedding(model, sound_a)
embedding_b = get_embedding(model, sound_b)

mixed_embedding = (embedding_a + embedding_b)/2
mixed_spec = decoding(model, mixed_embedding, pitch=config["inference"]["pitch"]) # You can change pitch here
mixed_audio = network_output_to_audio(mixed_spec[0])
# save mixed_audio

torchaudio.save(f'mixed_audio_{cur_time}.wav', mixed_audio, config["inference"]["sr"])
print('save done!')