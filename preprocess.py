from concurrent.futures import process
import csv
import os
import torch
import random
from tqdm.auto import tqdm
import torchaudio

SAMPLE_RATE = 22000
MIN_DURATION_SECONDS = 4
MIN_DURATION_SAMPLES = SAMPLE_RATE * MIN_DURATION_SECONDS

def load_metadata():
  METADATA = "data/LJSpeech-1.1/metadata.csv"
  AUDIO = "data/LJSpeech-1.1/wavs/"
  with open(METADATA, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    ret = []
    for row in reader:
      ret.append((os.path.join(AUDIO, row[0]+'.wav'), row[1]))
  return ret

def transform_audio(waveform, sample_rate):
  if waveform.shape[1] >= MIN_DURATION_SAMPLES:
    waveform = waveform[0, :MIN_DURATION_SAMPLES]
    # We extracted 80-channel filterbanks features computed from a 25ms window with a stride of 10ms.
    hop_length = int(sample_rate/(1000/10))
    win_length = int(sample_rate/(1000/25))
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=win_length, win_length=win_length, hop_length=hop_length, n_mels=80)(waveform)
    return mel_transform
  else:
    return None

def load_and_transform(x, transform=True):
  waveform, sample_rate = torchaudio.load(x, normalize=True)
  if transform:
    return transform_audio(waveform, sample_rate)
  else:
    return waveform

def extract_audios(data, transform=True):
  ret = []
  for i in tqdm(data):
    audio = load_and_transform(i[0], transform)
    if audio != None:
      ret.append(audio)
  return ret

if __name__ == "__main__":
  #data = load_metadata()
  data = load_metadata()[:100]
  print(f"Got {len(data)} audio files")
  random.shuffle(data)
  audios = extract_audios(data, True)
  print(f"Usable audio files {len(audios)}")  
  print(audios[0].shape)
  #torch.save(audios, "data/full_audio.pt")

