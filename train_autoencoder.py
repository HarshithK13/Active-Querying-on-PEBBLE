#!/usr/bin/env python3
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



class PreferenceSegmentDataset(Dataset):
    """
    Offline dataset for segments from the preference .npz file.

    Each sample is a single segment flattened to a 1D vector:
        x.shape = (T * (ds + da),)
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)

        seg1 = data["seg1"]   # (N, T, D)
        seg2 = data["seg2"]   # (N, T, D)
        print(seg1.shape)
        print(seg2.shape)
        # print("seg1: ", seg1)
        # print("seg2: ", seg2)

        # stack both as independent segments: (2N, T, D)
        segments = np.concatenate([seg1, seg2], axis=0).astype(np.float32)
        # print("segments: ", segments)
        print("segments.shape: ", segments.shape)
        self.N, self.T, self.D = segments.shape
        self.flat_segments = segments.reshape(self.N, self.T * self.D)
        # print("flat_segments: ", self.flat_segments)
        print("flat_segments.shape: ", self.flat_segments.shape)
        print(f"[Dataset] Loaded {self.N} segments from {npz_path}")
        print(f"[Dataset] Segment length T = {self.T}, feature dim = {self.D}")
        print(f"[Dataset] Flattened input dim = {self.flat_segments.shape[1]}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.flat_segments[idx]
        return torch.from_numpy(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()


        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z



# ---------- Training ----------

def train_autoencoder(
    prefs_path,
    output_dir,
    latent_dim=64,
    batch_size=256,
    lr=1e-3,
    num_epochs=50,
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    dataset = PreferenceSegmentDataset(prefs_path)
    input_dim = dataset.flat_segments.shape[1]

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon, z = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.size(0)
            global_step += 1

        avg_loss = running_loss / len(dataset)
        print(f"[Epoch {epoch:03d}] recon_loss = {avg_loss:.6f}")

    # save encoder + decoder
    save_path = os.path.join(output_dir, "autoencoder.pth")
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "segment_len": dataset.T,
            "feature_dim": dataset.D,
        },
        save_path,
    )
    print(f"[Save] Autoencoder saved to {save_path}")


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Train segment autoencoder from preference dataset")
    parser.add_argument("--prefs_path", type=str, required=True,
                        help="Path to prefs_seedXXXX.npz")
    parser.add_argument("--output_dir", type=str, default="latent_models",
                        help="Where to save the trained autoencoder")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_autoencoder(
        prefs_path=args.prefs_path,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
    )
