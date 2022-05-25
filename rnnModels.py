import torch.nn as nn
import torch


class RnnGenerator(nn.Module):
    def __init__(self, input_size=10, hidden_size=350):
        super(RnnGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=2, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size, 80)
        self.bn = nn.BatchNorm1d(num_features=1012) 

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z, verbose=False):
        x, hidden = self.lstm(z)
        x = self.fc1(self.bn(x))
        return x, hidden
    
    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)


class RnnDiscriminator(nn.Module):
    def __init__(self, input_size=80, hidden_size=350):
        super(RnnDiscriminator, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        self.relu = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(hidden_size*1012*2, 1)
        self.bn = nn.BatchNorm1d(num_features=hidden_size*1012*2)

    def forward(self, x, verbose=False):

        # YOUR CODE HERE 1012, 80

        x, hidden = self.lstm(x)
        x = self.fc1(self.bn(x.reshape(x.shape[0], -1)))
        return torch.sigmoid(x), hidden


if __name__ == "__main__":
  device = torch.device('cpu')
  nz = 10
  bs = 32
  mel_length = 1012

  z = torch.randn(bs, mel_length, 10, device=device)
  netG = RnnGenerator()
  netD = RnnDiscriminator()
  samples, hiddden = netG(z)
  samples, hidden = netD(samples)
  # target ([100, 1, 1012, 80])
