from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform to grayscale and normalize
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


#--------

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

model = Autoencoder().to(device)


#------------

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def add_noise(img):
    noise = torch.randn_like(img) * 0.2
    noisy_img = img + noise
    return noisy_img

#----------------
print('Starting model training')

num_epochs = 5
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        noisy_img = add_noise(img).to(device)
        
        output = model(noisy_img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Pass [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Model Training Successful')
#----------

print('Starting the evaluation')
#test images
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
        axes[1].set_title('Noisy')
        axes[2].imshow(np.squeeze(output), cmap='gray')
        axes[2].set_title('Denoised')
        plt.show()

#------------------------------------------------

def l_and_p_image(image_path, transform, image_size=(32, 32)):
    image = Image.open(image_path).convert('L') 
    image = image.resize(image_size)  
    image = transform(image)  
    return image

image_path = 'dog1.jpeg'

custom_image = l_and_p_image(image_path, transform)
custom_image = custom_image.unsqueeze(0).to(device)


noisy_custom_image = add_noise(custom_image)


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