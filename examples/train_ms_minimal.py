#!/usr/bin/env python3
"""
Minimal MS training loop template using MSDataset.

Edit paths and hyperparameters before real training.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import create_ms_dataloader


class TinyMsSR(nn.Module):
    """Very small baseline model for quick wiring tests."""

    def __init__(self, num_frames: int = 4, num_bands: int = 8, hidden: int = 48):
        super().__init__()
        in_channels = num_frames * num_bands
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_bands, kernel_size=3, padding=1),
        )

    def forward(self, lr_frames):
        # lr_frames is a list of tensors, each [B, C, H, W]
        x = torch.cat(lr_frames, dim=1)
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update these paths for your environment.
    hr_dir = Path("data/input_ms")
    config_path = Path("configs/default_config.yaml")
    global_stats_path = Path("configs/combined_stats.yaml")

    loader = create_ms_dataloader(
        hr_image_dir=hr_dir,
        config_path=config_path,
        global_stats_path=global_stats_path if global_stats_path.exists() else None,
        batch_size=2,
        num_workers=0,
        shuffle=True,
        augment=True,
        seed=42,
    )

    first_batch = next(iter(loader))
    num_frames = len(first_batch["lr"])
    num_bands = first_batch["hr"].shape[1]

    model = TinyMsSR(num_frames=num_frames, num_bands=num_bands).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    max_steps = 20

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break

        lr_frames = [x.to(device) for x in batch["lr"]]
        hr = batch["hr"].to(device)

        # Match output resolution to HR for this tiny baseline.
        pred = model(lr_frames)
        pred = nn.functional.interpolate(pred, size=hr.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(pred, hr)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"step={step} loss={loss.item():.6f}")

    print("Finished minimal MS template run.")


if __name__ == "__main__":
    main()
