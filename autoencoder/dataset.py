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
import pretty_midi


class MusicNetDataSet:
  def __init__(self, path, sr=16000):
    if isinstance(path, str):
      path = Path(path)
    self.path = path
    self.file_list = list(self.path.rglob('*.wav'))
    self.sr = sr

  def __getitem__(self, idx):
    fn = self.file_list[idx]
    csvs = str(self.path) + '/' + fn.stem + '.f0.csv'
    df = pd.read_csv(csvs)
    audio, sr = torchaudio.load(fn)
    audio_sec = audio.size(1)/self.sr
    start_sec = random.uniform(0, audio_sec-4)
    end_sec = start_sec + 4
    start_sample = int(start_sec * self.sr)
    end_sample = int(end_sec * self.sr)
    extract_tensor = audio[:, start_sample:end_sample]
    extracted_crepe = df[(df['time'] >= start_sec ) & (df['time'] <= end_sec)]
    frequency_crepe = torch.tensor(extracted_crepe['frequency'].values, dtype=torch.float)
    frequency = np.repeat(frequency_crepe, extract_tensor.shape[1] // len(frequency_crepe))
    notenum = np.array([int(pretty_midi.hz_to_note_number(f)) for f in frequency])
    frequency = torch.tensor(notenum, dtype=torch.long)
    return extract_tensor, frequency[0]
  
  def __len__(self):
    return len(self.file_list)