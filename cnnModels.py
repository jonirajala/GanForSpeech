import torch.nn as nn
import torch


class CnnGenerator(nn.Module):
    def __init__(self, nz=10, ngf=64, nc=1):
        super(CnnGenerator, self).__init__()
        self.convt_1 = nn.ConvTranspose2d(
            in_channels=nz, out_channels=4*ngf, kernel_size=4, stride=2, bias=False, padding=0)
        self.convt_2 = nn.ConvTranspose2d(
            in_channels=4*ngf, out_channels=2*ngf, kernel_size=4, stride=2, bias=False, padding=0)
        self.convt_3 = nn.ConvTranspose2d(
            in_channels=2*ngf, out_channels=ngf, kernel_size=4, stride=2, bias=False, padding=0)
        self.convt_4 = nn.ConvTranspose2d(
            in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, bias=False, padding=1)
            

        self.bn1 = nn.BatchNorm2d(num_features=4*ngf)
        self.bn2 = nn.BatchNorm2d(num_features=2*ngf)
        self.bn3 = nn.BatchNorm2d(num_features=ngf)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z, verbose=False):


        x = self.relu(self.bn1(self.convt_1(z)))
        x = self.relu(self.bn2(self.convt_2(x)))
        x = self.relu(self.bn3(self.convt_3(x)))
        x = self.tanh(self.convt_4(x))

        return x


class CnnDiscriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(CnnDiscriminator, self).__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=nc, out_channels=ndf, kernel_size=4, stride=3, bias=False, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=ndf, out_channels=2 *
                                ndf, kernel_size=4, stride=4, bias=False, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=2*ndf, out_channels=4 *
                                ndf, kernel_size=4, stride=4, bias=False, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=4*ndf, out_channels=1, kernel_size=4, stride=4, bias=False, padding=1)

        self.bn3 = nn.BatchNorm2d(num_features=4*ndf)
        self.bn2 = nn.BatchNorm2d(num_features=2*ndf)
        self.bn1 = nn.BatchNorm2d(num_features=ndf)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, verbose=False):
        x = self.relu(self.bn1(self.conv_1(x)))
        x = self.relu(self.bn2(self.conv_2(x)))
        x = self.relu(self.bn3(self.conv_3(x)))
        x = self.conv_4(x).reshape(-1)

        return torch.sigmoid(x)


if __name__ == "__main__":
  device = torch.device('cpu')
  nz = 10
  #402
  z = torch.randn(144, nz, 23, 3, device=device)
  print(z.shape)
  netG = CnnGenerator()
  netD = CnnDiscriminator()
  samples = netG(z)
  l = netD(samples)
  print(l.shape)
  # target ([100, 1, 1012, 80])
  print(samples.shape)