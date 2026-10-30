#!/usr/bin/env python3
"""Pre-CHF threshold: choose on VALIDATION, report on TEST (Table 8)."""
import os, json
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, precision_score, recall_score

BASE  = "/home/r4/sharma/new_acoustic_boiling_result"
MODEL = f"{BASE}/models/Original_CNN_STFT_best.h5"
SPLIT = f"{BASE}/outputSpectrogram/Split_Final/STFT"
OUT   = f"{BASE}/results/threshold_val_tuned.json"
CLASSES = ['BKG', 'NB', 'Pre-CHF', 'TB']
PC = CLASSES.index('Pre-CHF')

print("Loading model...")
model = tf.keras.models.load_model(MODEL, compile=False)

def get_probs(split):
    g = ImageDataGenerator(rescale=1./255).flow_from_directory(
        f"{SPLIT}/{split}", target_size=(224, 224), batch_size=32,
        class_mode='categorical', shuffle=False, classes=CLASSES)
    return model.predict(g, verbose=1), g.classes

def predict_at(p, t):
    rest = p.copy(); rest[:, PC] = -1
    return np.where(p[:, PC] >= t, PC, np.argmax(rest, axis=1))

def scores_at(p, y, t):
    yp = predict_at(p, t)
    yt_b = (y == PC).astype(int)
    yp_b = (yp == PC).astype(int)
    return {
        'threshold': round(float(t), 2),
        'accuracy':  round(float(np.mean(yp == y) * 100), 2),
        'precision': round(float(precision_score(yt_b, yp_b, zero_division=0)), 4),
        'recall':    round(float(recall_score(yt_b, yp_b, zero_division=0)), 4),
        'f1':        round(float(f1_score(yt_b, yp_b, zero_division=0)), 4),
    }

print("Predicting on validation set...")
pv, yv = get_probs('validation')
print("Predicting on test set...")
pt, yt = get_probs('test')

grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
val_curve = [scores_at(pv, yv, t) for t in grid]
best = max(val_curve, key=lambda r: (r['f1'], r['recall']))
t_star = best['threshold']

out = {
    'protocol': 'threshold chosen on validation set (59 Pre-CHF), reported on test set (61 Pre-CHF)',
    'chosen_threshold': t_star,
    'validation_at_chosen': best,
    'test_at_chosen':       scores_at(pt, yt, t_star),
    'test_at_default_0.50': scores_at(pt, yt, 0.50),
    'validation_curve': val_curve,
}
with open(OUT, 'w') as f:
    json.dump(out, f, indent=2)

print(f"\nChosen threshold (val best F1) : {t_star}")
print(f"Validation at chosen t         : {best}")
print(f"Test at chosen t               : {out['test_at_chosen']}")
print(f"Test at default 0.50           : {out['test_at_default_0.50']}")
print(f"Saved: {OUT}")
