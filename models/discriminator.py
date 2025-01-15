import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cnn_block


class Discriminator(nn.Module):
    def __init__(self, c_dim=3, df_dim=64, instance_norm=False):  # input: 256x256
        super(Discriminator, self).__init__()
        self.conv1 = cnn_block(
            c_dim * 2, df_dim, 4, stride=2, padding=1, first_layer=True
        )  # 128x128
        self.conv2 = cnn_block(df_dim, df_dim * 2, 4, stride=2, padding=1)  # 64x64
        self.conv3 = cnn_block(df_dim * 2, df_dim * 4, 4, stride=2, padding=1)  # 32x32
        self.conv4 = cnn_block(df_dim * 4, df_dim * 8, 4, stride=1, padding=1)  # 31x31
        self.conv5 = cnn_block(
            df_dim * 8, 1, 4, stride=1, padding=1, first_layer=True
        )  # 30x30

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        Forward pass of the discriminator
        Args:
            x: target image
            y: input image
        Returns:
            Probability map of size 30x30
        """
        out = torch.cat([x, y], dim=1)
        out = F.leaky_relu(self.conv1(out), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(out), 0.2)
        out = F.leaky_relu(self.conv4(out), 0.2)
        out = self.conv5(out)

        return self.sigmoid(out)
