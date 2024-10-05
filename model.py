from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Greyscale and Normalization
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

#----------

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7)                       # (B, 64, 1, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),             # (B, 32, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 16, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, 32, 32)
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#------------

model = Autoencoder().to(device)

#loss function
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# add noise

def add_noise(img):
    noise = torch.randn_like(img) * 0.2
    noisy_img = img + noise
    return noisy_img


#-----------------

# Model Training
num_pass = 15
for epoch in range(num_pass):
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        noisy_img = add_noise(img).to(device)
        
        # Forward pass
        output = model(noisy_img)
        loss = criterion(output, img)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Pass [{epoch+1}/{num_pass}], Loss: {loss.item():.4f}')

#------------------

torch.save(model.state_dict(), 'trained.pth')
print('Model trained and saved')
