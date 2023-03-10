import sys
sys.path.append("..")
import argparse
from helper import helper
from torchvision import transforms
import random, torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Creating a parser
ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=10)
ap.add_argument('--lr', type=float, default=1e-1)
ap.add_argument('--batch_sz', type=int, default=4)
ap.add_argument('--weight_decay', type=float, default=1e-5)
args = vars(ap.parse_args())

epochs = args['epochs']
lr = args['lr']
batch_sz = args['batch_sz']
weight_decay = args['weight_decay']

helper.make_dir()
device = helper.device()


train_paths = helper.get_path('../Denoising_AE/denoising-dirty-documents/train/','*.png')
test_paths = helper.get_path('../Denoising_AE/denoising-dirty-documents/test/', '*.png')
cleaned_paths = helper.get_path('../Denoising_AE/denoising-dirty-documents/train_cleaned/', '*.png')

# show a couple of images
i = random.randint(0,len(train_paths)-1)
#helper.show_image(train_paths[i])
#helper.show_image(cleaned_paths[i])

#read_image
train = helper.read_image(train_paths)
test = helper.read_image(test_paths)
cleaned_image = helper.read_image(cleaned_paths)

#transforms on image
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.functional.rgb_to_grayscale,
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# Create Dataset
class image_data(Dataset):
    def __init__(self, image, labels=None, transforms=None):

        self.image = image
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        X = self.image[idx]
        X = X.astype(np.uint8)

        if self.transforms:
            X = self.transforms(X)

        if self.labels is not None:
            y = self.labels[idx]
            y = y.astype(np.uint8)
            y = self.transforms(y)

            return (X,y)
        else:
            return X
        
train_data = image_data(train, cleaned_image, transforms=transforms)
test_data = image_data(test, labels=None, transforms=transforms)

train_loader = DataLoader(train_data, batch_size=batch_sz, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_sz, shuffle=False)

# The Autoencoder Network Architecture
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, 2, padding=1, stride=2), #  (N, 64, 129, 129)
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 256, 2, padding=1, stride=2), # (N, 256, 33, 33)
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, padding=2, stride=2, output_padding=1), #  (N, 64, 64, 64)
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 2, padding=1, stride=2), # (N, 16, 256, 256)
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):

        x, indices1 = F.max_pool2d(self.encoder1(x), 2, 2, return_indices=True) #  (N, 64, 64, 64)
        x = self.encoder2(x) 
        x = self.decoder1(x)
        x = self.decoder2(F.max_unpool2d(x,indices1, 3, 2)) ## for unpool (N, 64, 129, 129)

        return x

    

model = AutoEncoder().to(device)
#print(model)

# Define the Loss Function and Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training loop
def Train(model, dataloader, epochs):
    print('---------------------------------------------Training started--------------------------------------------------')
    training_loss = []
    model.train()
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
        
        loss = running_loss/len(dataloader)
        training_loss.append(loss)
         
        if epoch % 5 == 0:
            print(f' Epoch {epoch} has Loss : {loss: .3f}')
            helper.save_decoded_image(output.detach().cpu(), f'../Denoising_AE/output/images/denoise_{epoch}.png')

    return training_loss


#Testing Loop
def test(model, dataloader):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i < 10:
                img = data.to(device)
                output = model(img)
                helper.save_decoded_image(output.detach().cpu(), f'../Denoising_AE/output/images/test_image_{i}.png')

# Training and Testing
training_loss = Train(model, train_loader, epochs=epochs)
test(model, test_loader)

# Saving model
torch.save(model.state_dict(), f'../Denoising_AE/output/models/model_epochs:{epochs}')

# Plotting
plt.plot(training_loss)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.savefig('../Denoising_AE/output/images/loss.png')
plt.show()
