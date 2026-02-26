#!/usr/bin/env python3
"""
Capture generated BW images from Arduino serial protocol and save PNG files.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import serial
from PIL import Image


HEX = set("0123456789abcdef")


def parse_block(lines: list[str], w: int = 16, h: int = 16):
    if len(lines) != h:
        return None
    data = []
    for ln in lines:
        ln = ln.strip().lower()
        if len(ln) != w or any(ch not in HEX for ch in ln):
            return None
        for ch in ln:
            v = int(ch, 16)  # 0..15
            data.append(v * 17)  # 0..255
    return data


def save_png(pixels: list[int], out_path: Path, scale: int):
    img = Image.new("L", (16, 16))
    img.putdata(pixels)
    if scale > 1:
        img = img.resize((16 * scale, 16 * scale), Image.NEAREST)
    img.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbserial-A50285BI")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--digit", type=int, default=0, help="Target class 0..9")
    ap.add_argument("--out-dir", type=Path, default=Path("gan_bw_arduino/generated_images"))
    ap.add_argument("--scale", type=int, default=16, help="PNG upscale factor")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(1.2)
    ser.reset_input_buffer()

    got = 0
    while got < args.count:
        d = max(0, min(9, int(args.digit)))
        ser.write(f"{d}\n".encode("ascii"))
        ser.flush()
        lines = []
        in_block = False
        got_class = d
        t0 = time.time()
        while time.time() - t0 < 8.0:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if line == "IMG_BEGIN":
                in_block = True
                lines = []
                continue
            if in_block and line.startswith("CLASS="):
                try:
                    got_class = int(line.split("=", 1)[1].strip())
                except Exception:
                    got_class = d
                continue
            if line == "IMG_END" and in_block:
                px = parse_block(lines)
                if px is not None:
                    out = args.out_dir / f"digit{got_class}_{got:03d}.png"
                    save_png(px, out, args.scale)
                    print(f"saved {out}")
                    got += 1
                break
            if in_block:
                lines.append(line)

    ser.close()
    print(f"done: {got} images")


if __name__ == "__main__":
    main()
