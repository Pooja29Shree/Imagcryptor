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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)  # Set to 0 for Windows compatibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
z_dim = 100       # latent dimension for generator
key_dim = 16      # dimension of key vector
img_channels = 3  # RGB images

# Instantiate models
G = Generator(z_dim, key_dim, img_channels).to(device)
D = Discriminator(img_channels).to(device)
R = Reconstructor(key_dim, img_channels).to(device)

# Optimizers
optG = torch.optim.Adam(
    list(G.parameters()) + list(R.parameters()),
    lr=2e-4, betas=(0.5, 0.999)
)
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Loss functions
bce = nn.BCEWithLogitsLoss().to(device)  # adversarial loss
l1  = nn.L1Loss().to(device)             # reconstruction loss

epochs = 50  # adjust as needed

for epoch in range(epochs):
    for real, _ in dataloader:
        real = real.to(device)
        bsz = real.size(0)

        # latent + key
        z = torch.randn(bsz, z_dim, device=device)
        key = torch.randn(bsz, key_dim, device=device)

        # ==== Train Discriminator ====
        fake = G(z, key)
        d_real = D(real)
        d_fake = D(fake.detach())
        lossD = bce(d_real, torch.ones_like(d_real)) + \
                bce(d_fake, torch.zeros_like(d_fake))
        optD.zero_grad()
        lossD.backward()
        optD.step()

        # ==== Train Generator + Reconstructor ====
        d_fake = D(fake)
        adv_loss = bce(d_fake, torch.ones_like(d_fake))
        recon = R(fake, key)
        recon_loss = l1(recon, real)
        lossG = adv_loss + 10.0 * recon_loss
        optG.zero_grad()
        lossG.backward()
        optG.step()

    print(f"Epoch [{epoch+1}/{epochs}]  LossD: {lossD.item():.4f}  "
          f"Adv: {adv_loss.item():.4f}  Recon: {recon_loss.item():.4f}")