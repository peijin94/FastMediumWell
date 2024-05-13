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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train the model
train(model, train_loader, epochs=400, device=device, criterion=criterion,
       optimizer=optimizer, file_epoch=True)

torch.save(model.state_dict(), 'modelepoch500bn256model2.pth')