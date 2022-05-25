from concurrent.futures import process
import csv
import os
import torch
import random
from tqdm.auto import tqdm
import torchaudio

SAMPLE_RATE = 22000
MAX_DURATION_SECONDS = 5
MAX_DURATION_SAMPLES = SAMPLE_RATE * MAX_DURATION_SECONDS

def load_metadata():
  METADATA = "data/LJSpeech-1.1/metadata.csv"
  AUDIO = "data/LJSpeech-1.1/wavs/"
  with open(METADATA, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    ret = []
    for row in reader:
      ret.append((os.path.join(AUDIO, row[0]+'.wav'), row[1]))
  return ret

def load_audio(x):
  waveform, sample_rate = torchaudio.load(x, normalize=True)
  # We extracted 80-channel filterbanks features computed from a 25ms window with a stride of 10ms.
  hop_length = int(sample_rate/(1000/10))
  win_length = int(sample_rate/(1000/25))
  mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=win_length, win_length=win_length, hop_length=hop_length, n_mels=80)(waveform)
  return mel_transform[0].T

def process_audio_file(x):
  x = load_audio(x)
  if x.shape[1] <= MAX_DURATION_SAMPLES:
    return x
  else:
    return None

def extract_audios(data):
  ret = []
  for i in tqdm(data):
    audio = process_audio_file(i[0])
    if audio != None:
      ret.append(audio)
  ret = torch.nn.utils.rnn.pad_sequence(ret, batch_first=True)
  return ret


if __name__ == "__main__":
  data = load_metadata()
  # data = load_metadata()[:100]
  print(f"Got {len(data)} audio files")
  random.shuffle(data)
  audios = extract_audios(data)
  print(f"Usable audio files {len(audios)}")
  dev = audios[:200]
  
  torch.save(audios, "data/full_audio.pt")
  torch.save(dev, "data/dev_audio.pt")
  