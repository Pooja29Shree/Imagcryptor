import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # or 3 channels if RGB
])
dataset = datasets.ImageFolder("../data/", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

