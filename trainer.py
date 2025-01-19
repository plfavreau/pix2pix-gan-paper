import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, generator, discriminator, train_loader, val_loader, config):
        self.G = generator
        self.D = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))

        self.bce_criterion = nn.BCELoss()
        self.L1_criterion = nn.L1Loss()

        self.G_losses = []
        self.D_losses = []
        self.G_GAN_losses = []
        self.G_L1_losses = []

        self.fixed_x, self.fixed_y = next(iter(self.train_loader))
        self.fixed_x = self.fixed_x.to(self.config.DEVICE)
        self.fixed_y = self.fixed_y.to(self.config.DEVICE)

        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)

    def train(self):
        for ep in range(self.config.EPOCHS):
            for i, data in enumerate(self.train_loader):
                y, x = data
                x = x.to(self.config.DEVICE)
                y = y.to(self.config.DEVICE)

                b_size = x.shape[0]

                real_class = torch.ones(b_size, 1, 30, 30).to(self.config.DEVICE)
                fake_class = torch.zeros(b_size, 1, 30, 30).to(self.config.DEVICE)

                # Train D
                self.D.zero_grad()
                real_patch = self.D(y, x)
                real_gan_loss = self.bce_criterion(real_patch, real_class)

                fake = self.G(x)
                fake_patch = self.D(fake.detach(), x)
                fake_gan_loss = self.bce_criterion(fake_patch, fake_class)

                D_loss = real_gan_loss + fake_gan_loss
                D_loss.backward()
                self.D_optimizer.step()

                # Train G
                self.G.zero_grad()
                fake_patch = self.D(fake, x)
                fake_gan_loss = self.bce_criterion(fake_patch, real_class)

                L1_loss = self.L1_criterion(fake, y)
                G_loss = fake_gan_loss + self.config.L1_LAMBDA * L1_loss
                G_loss.backward()
                self.G_optimizer.step()

                if (i + 1) % 200 == 0:
                    print(
                        f"Epoch [{ep}/{self.config.EPOCHS}], Step [{i + 1}/{len(self.train_loader)}], "
                        f"d_loss: {D_loss.item():.4f}, g_loss: {G_loss.item():.4f}, "
                        f"D(real): {real_patch.mean():.2f}, D(fake):{fake_patch.mean():.2f}, "
                        f"g_loss_gan:{fake_gan_loss.item():.4f}, g_loss_L1:{L1_loss.item():.4f}"
                    )
                    self.G_losses.append(G_loss.item())
                    self.D_losses.append(D_loss.item())
                    self.G_GAN_losses.append(fake_gan_loss.item())
                    self.G_L1_losses.append(L1_loss.item())

                    self.generate_and_save_images(ep, i)

            torch.save(self.G.state_dict(), f"{self.config.CHECKPOINT_DIR}/generator_epoch_{ep}.pth")
            torch.save(self.D.state_dict(), f"{self.config.CHECKPOINT_DIR}/discriminator_epoch_{ep}.pth")

    def generate_and_save_images(self, epoch, step):
        with torch.no_grad():
            self.G.eval()
            fake = self.G(self.fixed_x).detach().cpu()
            self.G.train()

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.title("Conditional Image (x)")
        plt.imshow(np.transpose(torchvision.utils.make_grid(self.fixed_x.cpu(), nrow=1, padding=5, normalize=True), (1, 2, 0)))

        plt.subplot(1, 3, 2)
        plt.axis("off")
        plt.title("Generated Image")
        plt.imshow(np.transpose(torchvision.utils.make_grid(fake, nrow=1, padding=5, normalize=True), (1, 2, 0)))

        plt.subplot(1, 3, 3)
        plt.axis("off")
        plt.title("Ground Truth (y)")
        plt.imshow(np.transpose(torchvision.utils.make_grid(self.fixed_y.cpu(), nrow=1, padding=5, normalize=True), (1, 2, 0)))

        plt.savefig(f"{self.config.RESULTS_DIR}/epoch_{epoch}_step_{step}.png")
        plt.close()
