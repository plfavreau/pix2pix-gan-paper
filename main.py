import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from data_loader import Data
from data_visu import display_images, load_image, split_image
from models.discriminator import Discriminator
from models.generator import Generator

batch_size = 8
workers = 10
epochs = 100
gf_dim = 64
df_dim = 64
L1_lambda = 100.0
in_w = in_h = 256
c_dim = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = Data()
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

path = "facades/train/"
val_dataset = Data(path=path)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

if __name__ == "__main__":
    image_path = f"{path}91.jpg"
    image = load_image(image_path)

    image_real, image_cond = split_image(image)

    display_images([image_real, image_cond], ["Real", "Condition"], figsize=(18, 6))

    G = Generator().to(device)
    D = Discriminator().to(device)

    G_optimizer = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    bce_criterion = nn.BCELoss()
    L1_criterion = nn.L1Loss()

    G_losses = []
    D_losses = []
    G_GAN_losses = []
    G_L1_losses = []
    img_list = []

    fixed_x, fixed_y = next(iter(train_loader))
    fixed_x = fixed_x.to(device)
    fixed_y = fixed_y.to(device)

    for ep in range(epochs):
        for i, data in enumerate(train_loader):
            y, x = data
            x = x.to(device)
            y = y.to(device)

            b_size = x.shape[0]

            real_class = torch.ones(b_size, 1, 30, 30).to(device)
            fake_class = torch.zeros(b_size, 1, 30, 30).to(device)

            # Train D
            D.zero_grad()
            real_patch = D(y, x)
            real_gan_loss = bce_criterion(real_patch, real_class)

            fake = G(x)
            fake_patch = D(fake.detach(), x)
            fake_gan_loss = bce_criterion(fake_patch, fake_class)

            D_loss = real_gan_loss + fake_gan_loss
            D_loss.backward()
            D_optimizer.step()

            G.zero_grad()
            fake_patch = D(fake, x)
            fake_gan_loss = bce_criterion(fake_patch, real_class)

            L1_loss = L1_criterion(fake, y)
            G_loss = fake_gan_loss + L1_lambda * L1_loss
            G_loss.backward()
            G_optimizer.step()

            if (i + 1) % 5 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(real): {:.2f}, D(fake):{:.2f}, g_loss_gan:{:.4f}, g_loss_L1:{:.4f}".format(
                        ep,
                        epochs,
                        i + 1,
                        len(train_loader),
                        D_loss.item(),
                        G_loss.item(),
                        real_patch.mean(),
                        fake_patch.mean(),
                        fake_gan_loss.item(),
                        L1_loss.item(),
                    )
                )
                G_losses.append(G_loss.item())
                D_losses.append(D_loss.item())
                G_GAN_losses.append(fake_gan_loss.item())
                G_L1_losses.append(L1_loss.item())

                with torch.no_grad():
                    G.eval()
                    fake = G(fixed_x).detach().cpu()
                    G.train()

                fig = plt.figure(figsize=(10, 10))
                plt.subplot(1, 3, 1)
                plt.axis("off")
                plt.title("conditional image (x)")
                plt.imshow(
                    np.transpose(
                        torchvision.utils.make_grid(
                            fixed_x.cpu(), nrow=1, padding=5, normalize=True
                        ),
                        (1, 2, 0),
                    )
                )

                plt.subplot(1, 3, 2)
                plt.axis("off")
                plt.title("fake image")
                plt.imshow(
                    np.transpose(
                        torchvision.utils.make_grid(
                            fake, nrow=1, padding=5, normalize=True
                        ),
                        (1, 2, 0),
                    )
                )

                plt.subplot(1, 3, 3)
                plt.axis("off")
                plt.title("ground truth (y)")
                plt.imshow(
                    np.transpose(
                        torchvision.utils.make_grid(
                            fixed_y.cpu(), nrow=1, padding=5, normalize=True
                        ),
                        (1, 2, 0),
                    )
                )

                plt.savefig(f"results/epoch_{ep}_step_{i}.png")
                plt.close()
                img_list.append(fig)

        torch.save(G.state_dict(), f"checkpoints/generator_epoch_{ep}.pth")
        torch.save(D.state_dict(), f"checkpoints/discriminator_epoch_{ep}.pth")
