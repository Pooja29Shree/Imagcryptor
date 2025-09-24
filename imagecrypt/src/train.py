import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Generator, Discriminator, Reconstructor
import torch.nn as nn


transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # or 3 channels if RGB
])
dataset = datasets.ImageFolder("../data/", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
z_dim = 100       # latent dimension for generator
key_dim = 16      # dimension of key vector
img_channels = 3  # RGB images

# Instantiate models
G = Generator(z_dim, key_dim, img_channels).to(device)
D = Discriminator(img_channels).to(device)
R = Reconstructor(img_channels, key_dim).to(device)

# Optimizers
optG = torch.optim.Adam(
    list(G.parameters()) + list(R.parameters()),
    lr=2e-4, betas=(0.5, 0.999)
)
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Loss functions
bce = nn.BCEWithLogitsLoss().to(device)  # adversarial loss
l1  = nn.L1Loss().to(device)             # reconstruction loss