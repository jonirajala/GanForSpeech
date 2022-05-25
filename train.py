import torch
from tqdm.auto import tqdm

from cnnModels import CnnDiscriminator, CnnGenerator
from rnnModels import RnnDiscriminator, RnnGenerator
from utils import generator_loss, discriminator_loss, AudioDataset

def load_data(fn):
  print("Loading data")
  #data = torch.load(f'/content/drive/MyDrive/ColabNotebooks/iTalk/data/{fn}') for google colab
  data = torch.load(f'/data/{fn}') # For local computing
  print("Data loaded")
  return data

def train(data):
  epochs = 10
  lr = 0.01
  batch_size = 64
  mel_length = 1012
  ds = AudioDataset(data.squeeze().to(device))
  dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

  netG = RnnGenerator()
  netD = RnnDiscriminator()

  netD = netD.to(device)
  netG = netG.to(device)

  #gen_optim = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
  #disc_optim = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
  gen_optim = torch.optim.Adam(netG.parameters(), lr=lr, weight_decay=1e-5)
  disc_optim = torch.optim.Adam(netD.parameters(), lr=lr, weight_decay=1e-5)
  
  netD.train()
  netG.train()
  for epoch in range(epochs):
      gen_loss = 0
      dis_loss = 0
      print(f"Epoch: {epoch}")
      for idx, x in enumerate(dataloader):
          x = x.to(device)
          noise = torch.randn(batch_size, mel_length, 10, device=device)
          fake_imgs, hidden = netG(noise)
          
          gen_optim.zero_grad()
          g_loss = generator_loss(netD, fake_imgs)
          g_loss.backward()
          gen_optim.step()
          gen_loss += g_loss.item()
          
          disc_optim.zero_grad()
          d_loss_real, D_real, d_loss_fake, D_fake = discriminator_loss(netD, x, fake_imgs.detach())
          d_loss_tot = d_loss_real + d_loss_fake
          d_loss_tot.backward()
          disc_optim.step()
          dis_loss += d_loss_tot.item()
          
          if idx % 100 == 0:
              print(f"D_real: {D_real}, D_fake: {D_fake}, Generator loss: {gen_loss/(idx+1)}, Disc loss: {dis_loss/(idx+1)}")
  torch.save(netG.state_dict(), '1_dcgan_g.pth')
  torch.save(netD.state_dict(), '1_dcgan_d.pth')


if __name__ == "__main__":
  #device = torch.device('cuda:0')
  device = torch.device('cpu')

  data = load_data('full_audio.pt')
  train(data)
