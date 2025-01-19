from torch.utils.data import DataLoader

import config
from data_loader import Data
from models.discriminator import Discriminator
from models.generator import Generator
from trainer import Trainer


def main():
    # Data loading
    train_dataset = Data(path=config.TRAIN_PATH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
    )

    val_dataset = Data(path=config.VAL_PATH)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
    )

    # Model initialization
    G = Generator(c_dim=config.C_DIM, gf_dim=config.GF_DIM).to(config.DEVICE)
    D = Discriminator(c_dim=config.C_DIM, df_dim=config.DF_DIM).to(config.DEVICE)

    # Trainer initialization and training
    trainer = Trainer(G, D, train_loader, val_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
