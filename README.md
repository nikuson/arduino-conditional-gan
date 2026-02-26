# Conditional BW GAN on Arduino Uno (16x16)

Tiny class-conditional GAN that generates grayscale digit images directly on **Arduino Uno** and streams them to PC as image blocks for saving to PNG.

## Highlights

- **Runs on Arduino Uno** (ATmega328P, 2KB SRAM / 32KB Flash).
- **Class-conditional generation**: request a specific digit `0..9`.
- **Grayscale output (16 levels)** instead of binary-only output.
- **End-to-end pipeline**:
  1. Train cGAN on MNIST (Python/PyTorch)
  2. Quantize generator to INT8 and export Arduino header
  3. Run inference on Uno
  4. Capture serial output and save PNG files on PC
- **Near-max Uno optimization**:
  - Flash usage ~94% for higher-capacity model
  - SRAM remains safe for runtime buffers

## Architecture

Generator (exported to Uno):
- Input: latent noise + one-hot class (`10`)
- MLP: `26 -> 48 -> 64 -> 256`
- Conditional class embeddings injected at each layer
- Output: `16x16` image (grayscale quantized to 4-bit levels for protocol)

Discriminator (training only on PC):
- Conditional MLP using image + class one-hot.

## Files

- `gan_bw_arduino/train_and_export_bw_gan.py`  
  Train conditional GAN on MNIST and export INT8 header.
- `gan_bw_arduino/tiny_bw_gan.ino`  
  Arduino Uno inference code (conditional generation + serial protocol).
- `gan_bw_arduino/tiny_bw_gan_model.h`  
  Auto-generated quantized model for Uno.
- `gan_bw_arduino/capture_bw_gan_images.py`  
  Reads serial protocol and saves PNG images.
- `gan_bw_arduino/README.md`  
  Quickstart commands.

## Quickstart

### 1) Train + export model
```bash
python3 gan_bw_arduino/train_and_export_bw_gan.py --epochs 30
```

2) Build + flash Uno
```bash
mkdir -p gan_bw_arduino_sketch
cp gan_bw_arduino/tiny_bw_gan.ino gan_bw_arduino_sketch/gan_bw_arduino_sketch.ino
cp gan_bw_arduino/tiny_bw_gan_model.h gan_bw_arduino_sketch/
arduino-cli compile --fqbn arduino:avr:uno gan_bw_arduino_sketch
arduino-cli upload -p /dev/cu.usbserial-A50285BI --fqbn arduino:avr:uno gan_bw_arduino_sketch
```
3) Generate PNGs for a chosen digit
```bash
python3 gan_bw_arduino/capture_bw_gan_images.py \
  --port /dev/cu.usbserial-A50285BI \
  --digit 7 \
  --count 20 \
  --out-dir gan_bw_arduino/generated_images_cgan7 \
  --scale 16
```
