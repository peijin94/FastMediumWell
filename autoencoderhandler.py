import torch

def add_elliptic_gaussian(image, centers, axes, angles, amplitudes):
    """
    Adds multiple elliptic Gaussians to the given image using PyTorch.
    """
    device = image.device  # Ensure all operations are on the same device as the image
    y_indices, x_indices = torch.meshgrid(torch.arange(image.size(0), device=device), 
                                          torch.arange(image.size(1), device=device), indexing='ij')
    cos_theta = torch.cos((angles))
    sin_theta = torch.sin((angles))
    
    for (x, y), (a, b), cos_t, sin_t, amplitude in zip(centers, axes, cos_theta, sin_theta, amplitudes):
        x0 = x_indices - x
        y0 = y_indices - y
        
        x_rot = cos_t * x0 + sin_t * y0
        y_rot = -sin_t * x0 + cos_t * y0
        
        gauss = amplitude * torch.exp(-((x_rot**2 / (2 * a**2)) + (y_rot**2 / (2 * b**2))))
        image += gauss


def spectral_noise(image_size, peak_frequency):
    """
    Generate noise with a specific peak frequency in the Fourier domain using PyTorch.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise = torch.randn(image_size, device=device)
    noise_fft = torch.fft.fftshift(torch.fft.fft2(noise))
    cy, cx = image_size[0] // 2, image_size[1] // 2
    y, x = torch.meshgrid(torch.arange(-cy, image_size[0]-cy, device=device), 
                          torch.arange(-cx, image_size[1]-cx, device=device), indexing='ij')
    mask = torch.exp(-((x**2 + y**2) - peak_frequency**2)**2 / (2 * (peak_frequency/2)**2))
    filtered_fft = noise_fft * mask
    filtered_noise = torch.real(torch.fft.ifft2(torch.fft.ifftshift(filtered_fft)))
    return filtered_noise


def create_image():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = (256, 256)
    #image = spectral_noise(image_size, 16).to(device)*1.5
    image = torch.zeros(image_size, device=device)
    image_foreground = torch.zeros(image_size, device=device)

    # Parameters for multiple Gaussians
    num_gaussians = 20
    centers = torch.randint(int(image_size[0]*0.15), int(image_size[0]*0.85), (num_gaussians, 2), device=device)
    axes = ( torch.rand( (num_gaussians, 2), device=device) *4+2) *( torch.rand((num_gaussians, 1), device=device) +0.5)

    angles = torch.randint(0, 360, (num_gaussians,), device=device)/180 * 3.14159
    amplitudes = (torch.randn(num_gaussians, device=device)**2) *6 +2

    # Add Gaussians to the images
    add_elliptic_gaussian(image, centers, axes, angles, amplitudes)
    #add_elliptic_gaussian(image_foreground, centers, axes, angles, amplitudes)
    
    imagmax = torch.max(image)
    image /= imagmax
    image_foreground /= imagmax
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
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 1 x 256 x 256
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # Output: 8 x 128 x 128
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # Output: 16 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: 64 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # Output: 128 x 8 x 8
            nn.ReLU(),
            nn.Flatten(-3,-1),  # Flatten to form the bottleneck
            nn.Linear(64 * 8 * 8, 64)  # Bottleneck layer with 16 units
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 64 * 8 * 8),  # Expand from the bottleneck
            nn.Unflatten(-1, (64, 8, 8)),  # Unflatten to the shape (128, 8, 8)
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 16 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 8 x 128 x 128
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 1 x 256 x 256
            nn.Sigmoid()  # Ensure output is in the range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder2, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 1 x 256 x 256
            nn.Conv2d(1, 8, 3, stride=2, padding=1),  # Output: 16 x 128 x 128
            nn.BatchNorm2d(8),
            nn.LeakyReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # Output: 16 x 64 x 64
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # b, 32, 32, 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # b, 64, 8, 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.Flatten(-3,-1),  # Flatten to form the bottleneck
            nn.Linear(128 * 8 * 8, 256),  
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * 8 * 8),  # Expand from the bottleneck
            nn.ReLU(True),
            nn.Unflatten(-1, (128, 8, 8)),  # Unflatten to the shape (128, 8, 8)
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # b, 64, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # b, 32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # b, 16, 64, 64
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # b, 1, 128, 128
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),  # b, 1, 256, 256
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder3(nn.Module):
    def __init__(self):
        super(Autoencoder3, self).__init__()
        self.encoder = nn.Sequential(
            # input (nc) x 256 x 256
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            # input (nfe) x 64 x 64
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(True),
            # input (nfe*2) x 32 x 32
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(True),
            # input (nfe*4) x 16 x 16
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(True),
            # input (nfe*8) x 8 x 8
            nn.Conv2d(64 * 8, 64 * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 16),
            nn.LeakyReLU(True),
            # input (nfe*16) x 4 x 4
            nn.Conv2d(64 * 16, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(True)
            # output (nz) x 1 x 1
        )

        self.decoder = nn.Sequential(
            # input (nz) x 1 x 1
            nn.ConvTranspose2d(1024, 64 * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 16),
            nn.ReLU(True),
            # input (nfd*16) x 4 x 4
            nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # input (nfd*8) x 8 x 8
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # input (nfd*4) x 16 x 16
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # input (nfd*2) x 32 x 32
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # input (nfd) x 64 x 64
            nn.ConvTranspose2d(64,   32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # input (nc) x 128 x 128
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # output (nc) x 128 x 128
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder4(nn.Module):
    def __init__(self):
        super(Autoencoder4, self).__init__()
        self.encoder = nn.Sequential(
            # input (nc) x 128 x 128
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # input (nfe) x 64 x 64
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # input (nfe*2) x 32 x 32
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # input (nfe*4) x 16 x 16
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # input (nfe*8) x 8 x 8
            nn.Conv2d(64 * 8, 64 * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 16),
            nn.ReLU(True),
            # input (nfe*16) x 4 x 4
            nn.Conv2d(64 * 16, 2048, 4, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
            # output (nz) x 1 x 1
        )

        self.decoder = nn.Sequential(
            # input (nz) x 1 x 1
            nn.ConvTranspose2d(2048, 64 * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 16),
            nn.ReLU(True),
            # input (nfd*16) x 4 x 4
            nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # input (nfd*8) x 8 x 8
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # input (nfd*4) x 16 x 16
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # input (nfd*2) x 32 x 32
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # input (nfd) x 64 x 64
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # output (nc) x 128 x 128
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
    return [image.unsqueeze(0).clone().detach(), 
            image_forground.unsqueeze(0).clone().detach()] # Add channel dimension

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

import time

def train(model, dataloader, epochs, optimizer, device, criterion=nn.MSELoss(),
          print_epoch=True,file_epoch=False, save_every=None, save_path=None, save_name_base=None):
    model.train()

    # time of the training
    t0 = time.time()
    loss_list = []

    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            img_noisy, img_clean = data
            inputs = img_noisy.to(device)
            #ref = img_clean.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if print_epoch:
            print('Epoch [{}/{}], Loss: {:.10f}'.format(epoch+1, epochs, total_loss / len(dataloader))+ ' # Time(s): '+ str(time.time()-t0))
        if file_epoch:
            with open('losses.txt','a') as f:
                f.write('Epoch [{}/{}], Loss: {:.10f}'.format(epoch+1, epochs, total_loss / len(dataloader))+ ' # Time(s): '+ str(time.time()-t0)+'\n')
        if save_every and save_path and save_name_base:
            if (epoch+1) % save_every == 0:
                print(f'Saving model at epoch {epoch+1}')
                torch.save(model.state_dict(), f'{save_path}{save_name_base}_epoch{epoch+1}.pth')
        
        loss_list.append(total_loss / len(dataloader))
    
    np.save('losses.npy', np.array(loss_list))
