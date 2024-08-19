############Combining all the above steps into 1##########
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm


# Configuration
epochs      = 100
batch_size  = 64
sample_size = 100    # Number of random values to sample
g_lr        = 1.0e-4 # Generator's learning rate
d_lr        = 1.0e-4 # Discriminator's learning rate


# DataLoader for MNIST
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)


# Generator Network
class Generator(nn.Sequential):
    def __init__(self, sample_size: int):
        super().__init__(
            nn.Linear(sample_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 784),
            nn.Sigmoid())

        # Random value vector size
        self.sample_size = sample_size

    def forward(self, batch_size: int):
        # Generate randon values
        z = torch.randn(batch_size, self.sample_size)

        # Generator output
        output = super().forward(z)

        # Convert the output into a greyscale image (1x28x28)
        generated_images = output.reshape(batch_size, 1, 28, 28)
        return generated_images


# Discriminator Network
class Discriminator(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1))

    def forward(self, images: torch.Tensor, targets: torch.Tensor):
        prediction = super().forward(images.reshape(-1, 784))
        loss = F.binary_cross_entropy_with_logits(prediction, targets)
        return loss


# To save images in grid layout
def save_image_grid(epoch: int, images: torch.Tensor, ncol: int):
    image_grid = make_grid(images, ncol)     # Images in a grid
    image_grid = image_grid.permute(1, 2, 0) # Move channel last
    image_grid = image_grid.detach().numpy()    # To Numpy
    plt.imshow(image_grid)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(f'generated_{epoch:03d}.jpg')
    plt.show()


# Real and fake labels
real_targets = torch.ones(batch_size, 1)
fake_targets = torch.zeros(batch_size, 1)


# Generator and Discriminator networks
generator = Generator(sample_size)
discriminator = Discriminator()


# Optimizers
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)


# Training loop
for epoch in range(epochs):

    d_losses = []
    g_losses = []

    for images, labels in tqdm(dataloader):
        #===============================
        # Discriminator Network Training
        #===============================

        # Loss with MNIST image inputs and real_targets as labels
        discriminator.train()
        d_loss = discriminator(images, real_targets)

        # Generate images in eval mode
        generator.eval()
        with torch.no_grad():
            generated_images = generator(batch_size)

        # Loss with generated image inputs and fake_targets as labels
        d_loss += discriminator(generated_images, fake_targets)

        # Optimizer updates the discriminator parameters
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        #===============================
        # Generator Network Training
        #===============================

        # Generate images in train mode
        generator.train()
        generated_images = generator(batch_size)

        # Loss with generated image inputs and real_targets as labels
        discriminator.eval() # eval but we still need gradients
        g_loss = discriminator(generated_images, real_targets)

        # Optimizer updates the generator parameters
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Keep losses for logging
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    # Print average losses
    print(epoch, np.mean(d_losses), np.mean(g_losses))

    # Save images
    save_image_grid(epoch, generator(batch_size), ncol=8)