import torch, os
from torchvision.utils import save_image


def device():
    return ('cuda' if torch.cuda.is_available() else 'cpu')

def make_dir():
    image_dir = '../output/images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

def save_decoded_image(img, name):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, name)
