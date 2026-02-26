/*
 * Class-conditional BW GAN generator for Arduino Uno.
 * Input class digit 0..9 in serial, emits 16x16 grayscale protocol block.
 */

#include <Arduino.h>
#include <avr/pgmspace.h>
#include <math.h>
#include "tiny_bw_gan_model.h"

static float z[GAN_LATENT];
static float zin[GAN_LATENT + GAN_CLASSES];
static float h1[GAN_H1];
static float h2[GAN_H2];
static uint8_t pix4[GAN_OUT]; // 0..15
static uint8_t targetClass = 0;

static inline int8_t readQ(const int8_t *arr, uint16_t idx) {
  return (int8_t)pgm_read_byte_near(arr + idx);
}

static void matvec_q(const float *in, const int8_t *w, float s, uint8_t inD, uint8_t outD, float *out) {
  for (uint8_t o = 0; o < outD; o++) {
    float acc = 0.0f;
    for (uint8_t i = 0; i < inD; i++) {
      acc += in[i] * (float)readQ(w, (uint16_t)i * outD + o);
    }
    out[o] = acc * s;
  }
}

static char nibbleHex(uint8_t v) {
  v &= 0x0F;
  return (v < 10) ? ('0' + v) : ('a' + (v - 10));
}

static void generate_image(uint8_t cls) {
  if (cls >= GAN_CLASSES) cls = 0;
  for (uint8_t i = 0; i < GAN_LATENT; i++) {
    z[i] = (float)random(-1000, 1001) / 1000.0f;
    zin[i] = z[i];
  }
  for (uint8_t i = 0; i < GAN_CLASSES; i++) zin[GAN_LATENT + i] = (i == cls) ? 1.0f : 0.0f;

  matvec_q(zin, G_W1_Q, S_G_W1_Q, GAN_LATENT + GAN_CLASSES, GAN_H1, h1);
  for (uint8_t i = 0; i < GAN_H1; i++) {
    h1[i] += (float)readQ(G_B1_Q, i) * S_G_B1_Q;
    h1[i] += (float)readQ(G_C1_Q, (uint16_t)cls * GAN_H1 + i) * S_G_C1_Q;
    if (h1[i] < 0.0f) h1[i] *= 0.2f;
  }

  matvec_q(h1, G_W2_Q, S_G_W2_Q, GAN_H1, GAN_H2, h2);
  for (uint8_t i = 0; i < GAN_H2; i++) {
    h2[i] += (float)readQ(G_B2_Q, i) * S_G_B2_Q;
    h2[i] += (float)readQ(G_C2_Q, (uint16_t)cls * GAN_H2 + i) * S_G_C2_Q;
    if (h2[i] < 0.0f) h2[i] *= 0.2f;
  }

  // Stream-like compute for output layer, store 4-bit grayscale in pix4[].
  for (uint16_t o = 0; o < GAN_OUT; o++) {
    float acc = 0.0f;
    for (uint8_t i = 0; i < GAN_H2; i++) {
      acc += h2[i] * (float)readQ(G_W3_Q, (uint16_t)i * GAN_OUT + o);
    }
    acc = acc * S_G_W3_Q + (float)readQ(G_B3_Q, o) * S_G_B3_Q;
    acc += (float)readQ(G_C3_Q, (uint16_t)cls * GAN_OUT + o) * S_G_C3_Q;
    float v = tanhf(acc); // [-1,1]
    int16_t q = (int16_t)((v + 1.0f) * 7.5f + 0.5f); // 0..15
    if (q < 0) q = 0;
    if (q > 15) q = 15;
    pix4[o] = (uint8_t)q;
  }
}

static void print_ascii_preview() {
  const char shades[] = " .:-=+*#%@";
  for (uint8_t r = 0; r < GAN_H; r++) {
    for (uint8_t c = 0; c < GAN_W; c++) {
      uint8_t v = pix4[r * GAN_W + c];
      uint8_t idx = (uint8_t)((uint16_t)v * 9 / 15);
      Serial.print(shades[idx]);
    }
    Serial.println();
  }
}

static void emit_protocol() {
  // Protocol:
  // IMG_BEGIN
  // CLASS=<0..9>
  // 16 rows, each 16 chars hex nibble ('0'..'f')
  // IMG_END
  Serial.println(F("IMG_BEGIN"));
  Serial.print(F("CLASS="));
  Serial.println(targetClass);
  for (uint8_t r = 0; r < GAN_H; r++) {
    for (uint8_t c = 0; c < GAN_W; c++) {
      Serial.print(nibbleHex(pix4[r * GAN_W + c]));
    }
    Serial.println();
  }
  Serial.println(F("IMG_END"));
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  randomSeed(analogRead(A0));

  Serial.println(F("Conditional BW GAN 16x16 on Arduino Uno"));
  Serial.println(F("Send digit 0..9 then Enter"));
  Serial.println();
}

void loop() {
  if (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c >= '0' && c <= '9') {
      targetClass = (uint8_t)(c - '0');
    }
    if (c == '\n' || c == '\r') {
      generate_image(targetClass);
      print_ascii_preview();
      emit_protocol();
      Serial.println();
    }
  }
}
