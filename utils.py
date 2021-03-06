import torch
from torch import nn
import torchaudio
from torch.utils.data import Dataset


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

def generator_loss(netD, fake_audios):
    out, hidden  = netD(fake_audios)
    out = out.to(fake_audios.device)
    tgt = torch.full([fake_audios.shape[0]], fake_label).to(fake_audios.device)
    loss = -((torch.log(out)).mean())
    
    return loss

def discriminator_loss(netD, real_audios, fake_audios):
    real_audios = real_audios.reshape(real_audios.shape[0], 1012, 80)
    tgt_real = torch.full([real_audios.shape[0]], real_label)
    tgt_fake = torch.full([fake_audios.shape[0]], fake_label)
    D_real, hidden  = netD(real_audios)
    D_fake, hidden  = netD(fake_audios)
    D_real = D_real.to(fake_audios.device).view(-1)
    D_fake = D_fake.to(fake_audios.device).view(-1)
    d_loss_real = -((torch.log(D_real)).mean())
    d_loss_fake = -((torch.log(1-D_fake)).mean())
    
    return d_loss_real, D_real.mean(), d_loss_fake, D_fake.mean()

class AudioDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample.unsqueeze(0)).squeeze() # For some reason TimeMasking needs extra dimension

        return sample

def load_data(fn):
  print("Loading data")
  #data = torch.load(f'/content/drive/MyDrive/ColabNotebooks/iTalk/data/{fn}') for google colab
  data = torch.load(f'data/{fn}') # For local computing
  print("Data loaded")
  return data

def generate_img(generator):
    with torch.no_grad():
        noise = torch.randn(1, generator.mel_len, 10, device=generator.device)
        img, hidden = generator(noise)
    return img

#def untransform(audio):


if __name__ == "__main__":
    train_audio_transforms = nn.Sequential(
        # 80 is the full thing
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        # 256 is the hop size, so 86 is one second
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    )
    device = torch.device('cpu')
    data = load_data('full_audio_no_pad.pt')
    ds = AudioDataset(data, train_audio_transforms)
    print(ds[0])
