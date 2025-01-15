import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cnn_block, tcnn_block


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

        self.d1 = tcnn_block(gf_dim * 8, gf_dim * 8, 4, 2, 1)
        self.d2 = tcnn_block(gf_dim * 8 * 2, gf_dim * 8, 4, 2, 1)
        self.d3 = tcnn_block(gf_dim * 8 * 2, gf_dim * 8, 4, 2, 1)
        self.d4 = tcnn_block(gf_dim * 8 * 2, gf_dim * 8, 4, 2, 1)
        self.d5 = tcnn_block(gf_dim * 8 * 2, gf_dim * 4, 4, 2, 1)
        self.d6 = tcnn_block(gf_dim * 4 * 2, gf_dim * 2, 4, 2, 1)
        self.d7 = tcnn_block(gf_dim * 2 * 2, gf_dim * 1, 4, 2, 1)
        self.d8 = tcnn_block(gf_dim * 1 * 2, c_dim, 4, 2, 1, first_layer=True)

        self.tanh = nn.Tanh()

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

        d1 = torch.cat([F.dropout(self.d1(e8_bis), 0.5, training=True), e7], dim=1)
        d1_bis = F.relu(d1)

        d2 = torch.cat([F.dropout(self.d2(d1_bis), 0.5, training=True), e6], dim=1)
        d2_bis = F.relu(d2)

        d3 = torch.cat([F.dropout(self.d3(d2_bis), 0.5, training=True), e5], dim=1)
        d3_bis = F.relu(d3)

        d4 = torch.cat([self.d4(d3_bis), e4], dim=1)
        d4_bis = F.relu(d4)

        d5 = torch.cat([self.d5(d4_bis), e3], dim=1)
        d5_bis = F.relu(d5)

        d6 = torch.cat([self.d6(d5_bis), e2], dim=1)
        d6_bis = F.relu(d6)

        d7 = torch.cat([self.d7(d6_bis), e1], dim=1)
        d7_bis = F.relu(d7)

        d8 = self.d8(d7_bis)
        d8_bis = self.tanh(d8)

        return d8_bis
