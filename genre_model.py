import torch
from torch import nn
from torch.utils.data import Dataset

LATENT_DIM = 16

class GenreDataset(Dataset):
    def __init__(self, matrix):
        self.X = torch.tensor(matrix, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx]

class GenreAutoencoder(nn.Module):
    def __init__(self, num_genres):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genres, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_DIM)
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, num_genres),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z