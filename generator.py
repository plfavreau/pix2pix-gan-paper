import torch.nn as nn
import torch.nn.functional as F

from utils import cnn_block


class Generator(nn.Module):
    def __init__(self, c_dim=3, gf_dim=64, instance_norm=False):
        super(Generator, self).__init__()

        self.e1 = cnn_block(c_dim, gf_dim, 4, 2, 1, first_layer=True)
        self.e2 = cnn_block(gf_dim, gf_dim * 2, 4, 2, 1)
        self.e3 = cnn_block(gf_dim * 2, gf_dim * 4, 4, 2, 1)
        self.e4 = cnn_block(gf_dim * 4, gf_dim * 8, 4, 2, 1)
        self.e5 = cnn_block(gf_dim * 8, gf_dim * 8, 4, 2, 1)
        self.e6 = cnn_block(gf_dim * 8, gf_dim * 8, 4, 2, 1)
        self.e7 = cnn_block(gf_dim * 8, gf_dim * 8, 4, 2, 1)
        self.e8 = cnn_block(gf_dim * 8, gf_dim * 8, 4, 2, 1, first_layer=True)

    def forward(self, x):
        e1 = self.e1(x)
        e1_bis = F.leaky_relu(e1, 0.2)

        e2 = self.e2(e1_bis)
        e2_bis = F.leaky_relu(e2, 0.2)

        e3 = self.e3(e2_bis)
        e3_bis = F.leaky_relu(e3, 0.2)

        e4 = self.e4(e3_bis)
        e4_bis = F.leaky_relu(e4, 0.2)

        e5 = self.e5(e4_bis)
        e5_bis = F.leaky_relu(e5, 0.2)

        e6 = self.e6(e5_bis)
        e6_bis = F.leaky_relu(e6, 0.2)

        e7 = self.e7(e6_bis)
        e7_bis = F.leaky_relu(e7, 0.2)

        e8 = self.e8(e7_bis)
        e8_bis = F.relu(e8)

        return e8_bis
