#!/usr/bin/env python3
"""
Train class-conditional BW GAN (16x16) and export INT8 generator for Arduino Uno.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


LATENT = 16
H1 = 48
H2 = 64
OUT = 256  # 16x16
CLASSES = 10


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(LATENT + CLASSES, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, OUT)
        self.c1 = nn.Embedding(CLASSES, H1)
        self.c2 = nn.Embedding(CLASSES, H2)
        self.c3 = nn.Embedding(CLASSES, OUT)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_oh = F.one_hot(y, num_classes=CLASSES).float()
        x = torch.cat([z, y_oh], dim=-1)
        x = F.leaky_relu(self.fc1(x) + self.c1(y), 0.2)
        x = F.leaky_relu(self.fc2(x) + self.c2(y), 0.2)
        return torch.tanh(self.fc3(x) + self.c3(y))


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(OUT + CLASSES, H2)
        self.fc2 = nn.Linear(H2, H1)
        self.fc3 = nn.Linear(H1, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_oh = F.one_hot(y, num_classes=CLASSES).float()
        x = torch.cat([x, y_oh], dim=-1)
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        return self.fc3(h)


def quantize(arr: np.ndarray):
    max_abs = float(np.max(np.abs(arr)))
    scale = max_abs / 127.0 if max_abs > 1e-8 else 1.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale


def c_arr(name: str, arr: np.ndarray) -> str:
    flat = arr.reshape(-1)
    lines = []
    for i in range(0, flat.size, 16):
        lines.append("  " + ", ".join(str(int(v)) for v in flat[i : i + 16]))
    body = ",\n".join(lines)
    return f"const int8_t PROGMEM {name}[{flat.size}] = {{\n{body}\n}};\n"


def export_header(g: Generator, out_path: Path) -> None:
    tensors = {
        "G_W1_Q": g.fc1.weight.detach().cpu().numpy().T.astype(np.float32),  # [26,48]
        "G_B1_Q": g.fc1.bias.detach().cpu().numpy().astype(np.float32),
        "G_W2_Q": g.fc2.weight.detach().cpu().numpy().T.astype(np.float32),  # [48,64]
        "G_B2_Q": g.fc2.bias.detach().cpu().numpy().astype(np.float32),
        "G_W3_Q": g.fc3.weight.detach().cpu().numpy().T.astype(np.float32),  # [64,256]
        "G_B3_Q": g.fc3.bias.detach().cpu().numpy().astype(np.float32),
        "G_C1_Q": g.c1.weight.detach().cpu().numpy().astype(np.float32),  # [10,48]
        "G_C2_Q": g.c2.weight.detach().cpu().numpy().astype(np.float32),  # [10,64]
        "G_C3_Q": g.c3.weight.detach().cpu().numpy().astype(np.float32),  # [10,256]
    }
    qd = {}
    sd = {}
    for k, v in tensors.items():
        q, s = quantize(v)
        qd[k] = q
        sd[f"S_{k}"] = s

    parts = [
        "#pragma once",
        "#include <Arduino.h>",
        "#include <avr/pgmspace.h>",
        "",
        f"#define GAN_LATENT {LATENT}",
        f"#define GAN_CLASSES {CLASSES}",
        f"#define GAN_H1 {H1}",
        f"#define GAN_H2 {H2}",
        "#define GAN_W 16",
        "#define GAN_H 16",
        "#define GAN_OUT 256",
        "",
    ]
    for k in ("G_W1_Q", "G_B1_Q", "G_W2_Q", "G_B2_Q", "G_W3_Q", "G_B3_Q", "G_C1_Q", "G_C2_Q", "G_C3_Q"):
        parts.append(c_arr(k, qd[k]))
    for sk, sv in sd.items():
        parts.append(f"const float {sk} = {sv:.10e}f;")

    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Exported {out_path} ({out_path.stat().st_size} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("gan_bw_arduino/mnist_bw_data"))
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-g", type=float, default=1.2e-3)
    ap.add_argument("--lr-d", type=float, default=9e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--header-out", type=Path, default=Path("gan_bw_arduino/tiny_bw_gan_model.h"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform = transforms.Compose(
        [
            transforms.Resize((16, 16), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t > 0.32).float()),  # binary target
            transforms.Lambda(lambda t: t * 2.0 - 1.0),  # [-1,1]
        ]
    )
    ds = datasets.MNIST(root=str(args.data_dir), train=True, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    g = Generator()
    d = Discriminator()
    opt_g = torch.optim.Adam(g.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    for ep in range(args.epochs):
        gl = []
        dlv = []
        for x, y in dl:
            x = x.view(x.size(0), -1)
            y = y.long()

            # discriminator
            z = torch.randn(x.size(0), LATENT)
            fake = g(z, y).detach()
            real_logit = d(x, y)
            fake_logit = d(fake, y)

            real_t = torch.empty_like(real_logit).uniform_(0.88, 1.0)  # label smoothing
            fake_t = torch.empty_like(fake_logit).uniform_(0.0, 0.12)

            d_loss = F.binary_cross_entropy_with_logits(real_logit, real_t)
            d_loss += F.binary_cross_entropy_with_logits(fake_logit, fake_t)

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
            opt_d.step()

            # generator
            z = torch.randn(x.size(0), LATENT)
            gen = g(z, y)
            gen_logit = d(gen, y)
            g_t = torch.ones_like(gen_logit)
            g_loss = F.binary_cross_entropy_with_logits(gen_logit, g_t)

            # tiny L1 prior to keep strokes compact
            g_loss = g_loss + 0.02 * gen.abs().mean()

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
            opt_g.step()

            gl.append(float(g_loss.item()))
            dlv.append(float(d_loss.item()))

        print(f"epoch={ep+1:02d} d={np.mean(dlv):.4f} g={np.mean(gl):.4f}")

    export_header(g, args.header_out)


if __name__ == "__main__":
    main()
