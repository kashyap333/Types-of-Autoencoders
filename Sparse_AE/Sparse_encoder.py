from torch import nn
import torch, argparse
import helper
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=10,
                help='Number of epochs')
ap.add_argument('--reg_parameter', type=float, default=0.001,
                help='sparsity loss regularisation value')
ap.add_argument('--add_sparse', type=bool, default=True,
                help='True or False for sparsity')
ap.add_argument('--batch_sz', type=int, default=16,
                help='batch size of dataloader')
ap.add_argument('--lr', type=float, default=1e-1,
                help='learning rate')
args = vars(ap.parse_args())

epochs = args['epochs']
reg_param = args['reg_parameter']
add_sparsity = args['add_sparse']
learning_rate = args['lr']
batch_size = args['batch_sz']
helper.make_dir()
#Data Loading
transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.FashionMNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
)
testset = datasets.FashionMNIST(
    root='../input/data',
    train=False,
    download=True,
    transform=transform
)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Sparse_AutoEncoder
class Sparse_AE(nn.Module):
    def __init__(self):
        super(Sparse_AE, self).__init__()

    #encoder
        self.encoder_decoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2,padding=1), # N,16,14,14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2,padding=1), # N,32,7,7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7), # N,64,1,1
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7), # N,32,7,7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,padding=1, output_padding=1), # N,16,14,14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,padding=1, output_padding=1), # N,1,28,28
            nn.ReLU()
        )

    def forward(self,x):

        return self.encoder_decoder(x)
        
device = helper.device()
model = Sparse_AE().to(device)
model_children = list(model.children())
        
# Loss_fn and Optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Sparse_loss function
def sparse_loss(image):
    loss = 0
    values = image
    for i in range(len(model_children[0])):
        if 1%2 == 0:
            values = model_children[0][i](values)
            loss += torch.mean(torch.abs(values))

    return loss

#Training loop
def train(model, train_data, epochs):
    print('--------------------Training Started----------------------------')
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=int(len(trainset)/trainloader.batch_size)):

        counter += 1
        img,_ = data
        img = img.to(device)
        output = model(img)
        mse_loss = loss_fn(output, img)
        if add_sparsity:
            l1_loss = sparse_loss(img)
            loss = mse_loss + reg_param*l1_loss
        else:
            loss = mse_loss

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
def test(model, test_data, epochs):
    print('--------------------testing-------------------')
    model.eval()
    test_runningloss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_data), total = int(len(testset)/test_data.batch_size)):
            counter+=1
            img,_ = data
            img = img.to(device)
            output = model(img)
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

for epoch in range(epochs):
    print(f'Epoch {epoch}')
    train_epoch_loss = train(model, trainloader, epoch)
    test_epoch_loss = test(model, testloader, epoch)
    train_loss.append(train_epoch_loss)
    test_loss.append(test_epoch_loss)

end = time.time()

print(f'Total training and testing time: {(end-start)/60} minutes')

#saving the model

torch.save(model.state_dict(), f'output/sparse_ae{epochs}.pth')

#plotting
plt.figure(figsize=(10,7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(test_loss, color='red', label='test loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('output/loss.png')
plt.show()












