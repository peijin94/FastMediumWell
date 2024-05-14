import numpy as np
import matplotlib.pyplot as plt
import torch
from autoencoderhandler import *



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Model, loss, and optimizer
model = Autoencoder2().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-3)

# Dataset and DataLoader setup
num_train_images = 4096  # Specify how many images you want for training
train_dataset = DynamicImageDataset(num_train_images)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Train the model
train(model, train_loader, epochs=6000, device=device, criterion=criterion,
       optimizer=optimizer, file_epoch=True, save_every=300, save_path='modelfile/', save_name_base='model2')

#torch.save(model.state_dict(), 'modelepoch800bn256model2.pth')