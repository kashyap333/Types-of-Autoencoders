import sys, torch
sys.path.append('..')
import argparse
from torch import nn
from helper import helper
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=10)
ap.add_argument('--batch_sz', type=int, default=4)
ap.add_argument('--le', type=float, default=1e-2)
args = ap.parse_args()

helper.make_dir()
device = helper.device()

# Loading the dataset
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.FashionMNIST(root='', download=True, train=True, transform=transform)
test = datasets.FashionMNIST(root='', download=True, train=False, transform=transform)
train_load = DataLoader(train, batch_size=args.batch_sz, shuffle=True)
test_load = DataLoader(test, batch_size=args.batch_sz, shuffle=False)

image, label = next(iter(train_load))
print(image.size())
print(len(train_load))

#creating the model
class variational_encoder(nn.Module):
    def __init__(self, latent_dims):
        super(variational_encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.normal = torch.distributions.Normal(0, 1)
        self.k1 = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.normal.sample(mu.shape)
        self.k1 = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z
    

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = torch.nn.functional.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = variational_encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

# Training
def train(autoencoder, data, epochs=args.epochs):
    opt = torch.optim.Adam(autoencoder.parameters())
    vae.train()
    train_loss=0.0
    for epoch in tqdm(range(epochs)):
        for x, _ in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.k1
            loss.backward()
            opt.step()
            train_loss+=loss.item()
    return train_loss/len(data)

vae = VariationalAutoencoder(4).to(device) # GPU
training_loss = train(vae, train_load)

plt.plot(training_loss)

torch.save(vae.state_dict(), f'output/models.pth')
