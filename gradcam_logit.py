#!/usr/bin/env python3
"""
Class-averaged Grad-CAM for Original CNN + STFT.

Fixes three problems of the pipeline's Grad-CAM:
  1. Uses pre-softmax LOGITS (softmax saturates at 100% confidence
     -> zero gradients -> blank heatmaps for NB and TB).
  2. Averages over N correctly classified test images per class
     (one image is anecdotal).
  3. Adds a frequency profile with the y-axis in Hz, assuming the
     224-row STFT image spans 0 Hz (bottom) to 5000 Hz (top).

Outputs:
  figures/gradcam_logit_classavg.png
  results/gradcam_logit_summary.json
"""
import os, json, random
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

BASE    = "/home/r4/sharma/new_acoustic_boiling_result"
MODEL   = f"{BASE}/models/Original_CNN_STFT_best.h5"
TEST    = f"{BASE}/outputSpectrogram/Split_Final/STFT/test"
OUT_PNG = f"{BASE}/figures/gradcam_logit_classavg.png"
OUT_JSON = f"{BASE}/results/gradcam_logit_summary.json"
CLASSES = ['BKG', 'NB', 'Pre-CHF', 'TB']
BANDS   = {'BKG': (0, 100), 'NB': (200, 250), 'Pre-CHF': (200, 400), 'TB': (400, 500)}
FMAX, SIZE, N_PER_CLASS = 5000, 224, 30

model = tf.keras.models.load_model(MODEL, compile=False)
convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
last_conv = convs[-1].name
head = model.layers[-1]                       # Dense(4, softmax)
print(f"Last conv layer: {last_conv} | head: {head.name}")


def gradcam_logit(img, ci):
    x = tf.convert_to_tensor(img[None], tf.float32)
    with tf.GradientTape() as tape:
        h, co = x, None
        for layer in model.layers[:-1]:
            h = layer(h, training=False)
            if layer.name == last_conv:
                co = h
        logits = tf.matmul(h, head.kernel) + head.bias
        target = logits[:, ci]
    g = tape.gradient(target, co)
    w = tf.reduce_mean(g, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(co[0] * w, axis=-1))
    cam = tf.image.resize(cam[..., None], (SIZE, SIZE)).numpy()[..., 0]
    probs = tf.nn.softmax(logits).numpy()[0]
    return cam, probs


def load(path):
    return img_to_array(load_img(path, target_size=(SIZE, SIZE))) / 255.0


row_freq = FMAX * (1 - (np.arange(SIZE) + 0.5) / SIZE)   # row 0 = top = 5000 Hz
random.seed(42)
summary, panels = {}, {}

for ci, cls in enumerate(CLASSES):
    files = sorted(os.listdir(os.path.join(TEST, cls)))
    random.shuffle(files)
    cams, example = [], None
    for f in files:
        img = load(os.path.join(TEST, cls, f))
        cam, probs = gradcam_logit(img, ci)
        if np.argmax(probs) != ci or cam.max() <= 0:
            continue
        cams.append(cam / cam.max())
        if example is None:
            example = img
        if len(cams) >= N_PER_CLASS:
            break
    mean_cam = np.mean(cams, axis=0)
    profile = mean_cam.mean(axis=1)
    profile = profile / (profile.max() + 1e-8)
    peak_hz = float(row_freq[np.argmax(profile)])
    lo, hi = BANDS[cls]
    in_band = float(mean_cam[(row_freq >= lo) & (row_freq <= hi)].sum() / mean_cam.sum())
    below_1k = float(mean_cam[row_freq <= 1000].sum() / mean_cam.sum())
    summary[cls] = {'n_images': len(cams), 'peak_frequency_hz': round(peak_hz, 1),
                    'fraction_in_nominal_band': round(in_band, 3),
                    'nominal_band_hz': [lo, hi],
                    'fraction_below_1kHz': round(below_1k, 3)}
    panels[cls] = (example, mean_cam, profile)
    print(f"{cls:8s}: n={len(cams):2d}  peak={peak_hz:7.1f} Hz  "
          f"in {lo}-{hi} Hz: {in_band:.1%}  below 1 kHz: {below_1k:.1%}")

fig, axes = plt.subplots(4, 3, figsize=(14, 16),
                         gridspec_kw={'width_ratios': [1, 1, 0.8]})
for r, cls in enumerate(CLASSES):
    ex, cam, prof = panels[cls]
    s = summary[cls]
    axes[r, 0].imshow(ex, aspect='auto', extent=[0, 1, 0, FMAX])
    axes[r, 0].set_ylabel(f"{cls}\nFrequency (Hz)", fontweight='bold')
    axes[r, 0].set_xlabel('Time (s)')
    axes[r, 1].imshow(ex, aspect='auto', extent=[0, 1, 0, FMAX])
    axes[r, 1].imshow(cam, cmap='jet', alpha=0.5, aspect='auto',
                      extent=[0, 1, 0, FMAX], vmin=0, vmax=1)
    axes[r, 1].set_xlabel('Time (s)')
    axes[r, 2].plot(prof, row_freq, color='k')
    lo, hi = BANDS[cls]
    axes[r, 2].axhspan(lo, hi, color='red', alpha=0.25, label=f'nominal {lo}-{hi} Hz')
    axes[r, 2].axhline(s['peak_frequency_hz'], color='b', ls='--',
                       label=f"peak {s['peak_frequency_hz']:.0f} Hz")
    axes[r, 2].set_ylim(0, FMAX)
    axes[r, 2].set_xlabel('Mean activation')
    axes[r, 2].legend(fontsize=8, loc='upper right')
    if r == 0:
        axes[r, 0].set_title('Example STFT spectrogram')
        axes[r, 1].set_title(f'Class-averaged Grad-CAM (n={s["n_images"]}) overlay')
        axes[r, 2].set_title('Activation profile vs frequency')
fig.suptitle('Logit-based, class-averaged Grad-CAM — Original CNN + STFT',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor='white')
json.dump(summary, open(OUT_JSON, 'w'), indent=2)
print(f"Saved: {OUT_PNG}\nSaved: {OUT_JSON}")
