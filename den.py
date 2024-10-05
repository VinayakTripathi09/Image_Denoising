from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7)                       
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),             
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Autoencoder().to(device)
model.load_state_dict(torch.load('trained.pth'))


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
#test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

#--------

def add_noise(img):
    noise = torch.randn_like(img) * 0.2
    noisy_img = img + noise
    return noisy_img

#----------
# Test images

'''

model.eval()
with torch.no_grad():
    for i in range(2):
        img, _ = test_dataset[i]
        img = img.unsqueeze(0).to(device)
        noisy_img = add_noise(img).to(device)
        
        output = model(noisy_img)
        
        img = img.cpu().numpy()
        noisy_img = noisy_img.cpu().numpy()
        output = output.cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(np.squeeze(img), cmap='gray')
        axes[0].set_title('Original')
        axes[1].imshow(np.squeeze(noisy_img), cmap='gray')
        axes[1].set_title('With Noise')
        axes[2].imshow(np.squeeze(output), cmap='gray')
        axes[2].set_title('Denoised')
        plt.show()
'''

#------------------------------------------------
# My Images


def l_and_p_image(img_path, transform, image_size=(32, 32)):
    img = Image.open(img_path).convert('L')  
    img = img.resize(image_size)  
    img = transform(img)  
    return img

img_path = 'dog1.jpeg'

# Load and preprocess image
custom_image = l_and_p_image(img_path, transform)
custom_image = custom_image.unsqueeze(0).to(device)

# Add noise to the custom image
noisy_custom_image = add_noise(custom_image)

# Evaluate the model on the custom image
model.eval()
with torch.no_grad():
    denoised_image = model(noisy_custom_image)

# Convert images to numpy for visualization
custom_image_np = custom_image.cpu().numpy()
noisy_custom_image_np = noisy_custom_image.cpu().numpy()
denoised_image_np = denoised_image.cpu().numpy()

# Display the original, noisy, and denoised images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(np.squeeze(custom_image_np), cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(np.squeeze(noisy_custom_image_np), cmap='gray')
axes[1].set_title('Noisy')
axes[2].imshow(np.squeeze(denoised_image_np), cmap='gray')
axes[2].set_title('Denoised')
plt.show()
