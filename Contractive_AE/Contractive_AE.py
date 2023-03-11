import argparse as ag
import sys
sys.path.append('..')
from helper import helper
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
import torch
import time
from tqdm import tqdm
# Argument Parser
ap = ag.ArgumentParser()
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--lr', type=float, default=1e-2)
ap.add_argument('--batch_sz', type=int, default=4)
ap.add_argument('--lamda', type=float, default=1e-3, help='lambda*Frobenious norm of Jacobian. How much importance should be given to the loss')
args = ap.parse_args()

# Create folders
helper.make_dir()
device = helper.device()

# Load Data
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST(root='',train=True, download=True, transform=transform)
testset = datasets.FashionMNIST(root='', download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=args.batch_sz, shuffle=True)
test_loader = DataLoader(testset, batch_size=args.batch_sz, shuffle=False)

#Plotting a random image and its size
#image, _ = next(iter(train_loader))
#image = image[0].permute(1,2,0).numpy()
#img = plt.imshow(image)
#plt.show()
#print(f'image shape: {image.shape}')

# Creating model
class Contractive_AE(nn.Module):
    def __init__(self):
        super(Contractive_AE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), # N,16,14,14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), # N,32,7,7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7), # N,64,1,1
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7), # N,32,7,7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,padding=1, output_padding=1), # N,16,14,14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,padding=1, output_padding=1), # N,1,28,28
            nn.ReLU()
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)

        return enc, dec

loss_fn = nn.MSELoss()

# Defining the Contractive loss
def contractive_loss(input, hidden_img, output, lamda):
    mse = loss_fn(output, input)

    hidden_img.backward(torch.ones(hidden_img.size()).to(device), retain_graph=True)
    con_loss = torch.sqrt(torch.sum(torch.pow(input.grad, 2)))
    input.grad.data.zero_()

    loss = mse + (lamda * con_loss)

    return loss

model = Contractive_AE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(model, train_loader, epoch):
    print('--------------------Training Started----------------------------')
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total=int(len(trainset)/train_loader.batch_size)):

        counter += 1
        img,_ = data
        img = img.to(device)
        img.requires_grad_(True)
        img.retain_grad()
        
        hidden_img, output = model(img)
        loss = contractive_loss(img, hidden_img, output, args.lamda)
        img.requires_grad_(False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss/counter
    print(f'train Loss in epoch {epoch}: {loss:.3f}')

    # save for every 5 epochs
    if epoch % 5 == 0:
        helper.save_decoded_image(output, f'output/images/train_{epoch}.png')
    return epoch_loss

# Testing Loop
def test(model, test_data, epoch):
    print('--------------------testing-------------------')
    model.eval()
    test_runningloss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_data), total=int(len(testset)/test_data.batch_size)):
            counter+=1
            img,_ = data
            img = img.to(device)
            _, output = model(img)
            loss = loss_fn(output, img)
            test_runningloss += loss.item()

        epoch_loss = test_runningloss/counter
        print(f'test loss at epoch {epoch}: {loss:.3f}')

        if epoch%5 == 0:
            helper.save_decoded_image(output, f'output/images/test_{epoch}.png') 
        
        return epoch_loss
    
# running training and testing
train_loss = []
test_loss = []
start = time.time()

for epoch in range(args.epochs):
    print(f'Epoch {epoch}')
    train_epoch_loss = train(model, train_loader, epoch)
    test_epoch_loss = test(model, test_loader, epoch)
    train_loss.append(train_epoch_loss)
    test_loss.append(test_epoch_loss)

end = time.time()

print(f'Total training and testing time: {(end-start)/60} minutes')

#saving the model

torch.save(model.state_dict(), f'output/sparse_ae{args.epochs}.pth')

#plotting
plt.figure(figsize=(10,7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(test_loss, color='red', label='test loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('output/loss.png')
plt.show()




                          



