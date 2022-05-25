import torch
from torch.utils.data import Dataset

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

def generator_loss(netD, fake_audios):
    """Loss computed to train the GAN generator.

    Args:
      netD: The discriminator whose forward function takes inputs of shape (batch_size, nc, 28, 28)
         and produces outputs of shape (batch_size, 1).
      fake_audios of shape (batch_size, nc, 28, 28): Fake images produces by the generator.

    Returns:
      loss: The mean of the binary cross-entropy losses computed for all the samples in the batch.

    Notes:
    - Make sure that you process on the device given by `fake_audios.device`.
    - Use values of global variables `real_label`, `fake_label` to produce the right targets.
    """
    out, hidden  = netD(fake_audios)
    out = out.to(fake_audios.device)
    tgt = torch.full([fake_audios.shape[0]], fake_label).to(fake_audios.device)
    loss = -((torch.log(out)).mean())
    
    return loss

def discriminator_loss(netD, real_audios, fake_audios):
    """Loss computed to train the GAN discriminator.

    Args:
      netD: The discriminator.
      real_audios of shape (batch_size, nc, 28, 28): Real images.
      fake_audios of shape (batch_size, nc, 28, 28): Fake images produces by the generator.

    Returns:
      d_loss_real: The mean of the binary cross-entropy losses computed on the real_audios.
      D_real: Mean output of the discriminator for real_audios. This is useful for tracking convergence.
      d_loss_fake: The mean of the binary cross-entropy losses computed on the fake_audios.
      D_fake: Mean output of the discriminator for fake_audios. This is useful for tracking convergence.

    Notes:
    - Make sure that you process on the device given by `fake_audios.device`.
    - Use values of global variables `real_label`, `fake_label` to produce the right targets.
    """
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
            sample = self.transform(sample)

        return sample