import numpy as np
import matplotlib.pyplot as plt
def add_elliptic_gaussian(image, centers, axes, angles, amplitudes):
    """
    Adds multiple elliptic Gaussians to the given image.
    :param image: 2D numpy array representing the image
    :param centers: Array of (x, y) coordinates for the centers of the Gaussians
    :param axes: Array of (a, b) for the major and minor axes of the Gaussians
    :param angles: Array of angles in degrees for the rotation of the Gaussians
    :param amplitudes: Array of amplitudes for the Gaussian peaks
    """
    # Precompute indices, cosines, and sines
    y_indices, x_indices = np.indices(image.shape)
    cos_theta = np.cos(np.radians(angles))
    sin_theta = np.sin(np.radians(angles))
    
    for (x, y), (a, b), cos_t, sin_t, amplitude in zip(centers, axes, cos_theta, sin_theta, amplitudes):
        x0 = x_indices - x
        y0 = y_indices - y
        
        x_rot = cos_t * x0 + sin_t * y0
        y_rot = -sin_t * x0 + cos_t * y0
        
        gauss = amplitude * np.exp(-((x_rot**2 / (2*a**2)) + (y_rot**2 / (2*b**2))))
        image += gauss

def spectral_noise(image_size, peak_frequency):
    # Identical as previous implementation
    noise = np.random.normal(size=image_size)
    noise_fft = np.fft.fftshift(np.fft.fft2(noise))
    cy, cx = image_size[0] // 2, image_size[1] // 2
    y, x = np.ogrid[-cy:image_size[0]-cy, -cx:image_size[1]-cx]
    mask = np.exp(-((x**2 + y**2) - peak_frequency**2)**2 / (2*(peak_frequency/2)**2))
    filtered_fft = noise_fft * mask
    filtered_noise = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))
    return filtered_noise

def create_image():
    image_size = (256, 256)
    image = spectral_noise(image_size, 16)
    image_foreground = np.zeros(image_size)

    # Parameters for multiple Gaussians
    num_gaussians = 20
    centers = np.random.randint(128 - 70, 128 + 70, (num_gaussians, 2))
    axes = np.random.randint(3, 15, (num_gaussians, 2))
    angles = np.random.randint(0, 360, num_gaussians)
    amplitudes = (1 - np.random.power(5, num_gaussians)) * 2

    # Add Gaussians to the images
    add_elliptic_gaussian(image, centers, axes, angles, amplitudes)
    add_elliptic_gaussian(image_foreground, centers, axes, angles, amplitudes)
    
    return image, image_foreground


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*8*8, 12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 256*8*8),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def create_image_tensor():
    # Use the previously defined create_image() function here
    image, image_forground = create_image()  # assuming create_image returns a normalized 2D numpy array
    return [torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(image_forground, dtype=torch.float32).unsqueeze(0)] # Add channel dimension

class DynamicImageDataset(Dataset):
    def __init__(self, num_images):
        """
        :param num_images: Number of images to generate (size of the dataset).
        """
        self.num_images = num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return create_image_tensor()

# Assuming the autoencoder and other necessary imports and functions are defined


def train(model, dataloader, epochs, optimizer, device, criterion=nn.MSELoss()):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            img_noisy, img_clean = data
            inputs = img_noisy.to(device)
            ref = img_clean.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, ref)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

