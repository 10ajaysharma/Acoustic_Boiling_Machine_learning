#!/usr/bin/env python3
"""
Point 3 - what does the CNN add over a simple classifier?

Logistic regression and random forest on hand-crafted spectral features of the
ORIGINAL clips, using exactly the same train / validation / test clips as the
CNN (the clip lists are read from the cached image folders of Split_Final).
No augmentation; class imbalance handled with class_weight='balanced'.

Features per 1-s clip (53 numbers):
  48 log-spaced Welch PSD bands, 50 Hz - 5 kHz, in dB (level-normalised by the
     clip's total power so the classifier sees spectral SHAPE, not loudness),
  total level (dB), spectral centroid, spectral bandwidth, roll-off (85 %),
  zero-crossing rate.
Also a second feature set: the 224x224 STFT image downsampled to 28x28 (784
numbers) - the same picture the CNN sees, fed to a linear model.

Run:
  python baselines_simple.py --clips-root /path/to/original_clips \
        --split-images /path/to/Split_Final/STFT --out results_baselines

Outputs: baselines.csv (accuracy, macro F1, Pre-CHF P/R/F1 on validation and
test, per model, 3 seeds for RF), confusion_<model>.csv, and the per-clip
test predictions for a paired comparison with the CNN.
"""
import argparse, os, glob, json
import numpy as np, pandas as pd
import soundfile as sf, librosa
from scipy import signal
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

SR = 10000
CLASSES = ['BKG', 'NB', 'Pre-CHF', 'TB']
CLASS_DIRS = {'BKG': ['BKG', 'Background', 'background'], 'NB': ['NB', 'Nucleate', 'nucleate_boiling'],
              'Pre-CHF': ['Pre-CHF', 'PreCHF', 'Pre_CHF', 'prechf'], 'TB': ['TB', 'Transition', 'transition_boiling']}
EDGES = np.logspace(np.log10(50), np.log10(5000), 49)


def find_class_dir(root, cls):
    for d in CLASS_DIRS[cls]:
        hits = [p for p in glob.glob(os.path.join(root, '**', d), recursive=True) if os.path.isdir(p)]
        if hits: return hits[0]
    raise FileNotFoundError(f'no folder for class {cls} under {root}; edit CLASS_DIRS')


def wav_index(root):
    idx = {}
    for cls in CLASSES:
        for w in glob.glob(os.path.join(find_class_dir(root, cls), '**', '*.wav'), recursive=True):
            idx[(cls, os.path.splitext(os.path.basename(w))[0])] = w
    return idx


def split_items(split_images, split):
    out = []
    for cls in CLASSES:
        for p in glob.glob(os.path.join(find_class_dir(os.path.join(split_images, split), cls), '*.png')):
            out.append((cls, os.path.splitext(os.path.basename(p))[0], p))
    return out


def load_clip(path):
    y, sr = sf.read(path, dtype='float32')
    if y.ndim > 1: y = y.mean(1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def spectral_features(y):
    f, P = signal.welch(y, fs=SR, nperseg=1024, noverlap=512)
    total = P.sum() + 1e-20
    bands = [P[(f >= lo) & (f < hi)].sum() / total + 1e-12 for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    cen = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=512, hop_length=256).mean()
    bw = librosa.feature.spectral_bandwidth(y=y, sr=SR, n_fft=512, hop_length=256).mean()
    ro = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=512, hop_length=256, roll_percent=0.85).mean()
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256).mean()
    return np.array([*(10 * np.log10(bands)), 10 * np.log10(total), cen, bw, ro, zcr], dtype='float32')


def image_features(png):
    return (np.asarray(Image.open(png).convert('L').resize((28, 28)), dtype='float32') / 255.0).ravel()


def build(items, idx, kind):
    X, y = [], []
    for cls, stem, png in items:
        if kind == 'spectral':
            if (cls, stem) not in idx: continue
            X.append(spectral_features(load_clip(idx[(cls, stem)])))
        else:
            X.append(image_features(png))
        y.append(cls)
    return np.array(X), np.array(y)


def evaluate(name, clf, Xtr, ytr, Xva, yva, Xte, yte, out, preds):
    clf.fit(Xtr, ytr); rows = []
    for split, X, yy in [('validation', Xva, yva), ('test', Xte, yte)]:
        p = clf.predict(X)
        pr, rc, f1, _ = precision_recall_fscore_support(yy, p, labels=CLASSES, zero_division=0)
        rows.append({'model': name, 'split': split, 'n': len(yy), 'accuracy': accuracy_score(yy, p), 'macro_F1': f1_score(yy, p, average='macro'),
                     'PreCHF_precision': pr[2], 'PreCHF_recall': rc[2], 'PreCHF_F1': f1[2], 'NB_recall': rc[1], 'TB_F1': f1[3]})
        if split == 'test':
            pd.DataFrame(confusion_matrix(yy, p, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(os.path.join(out, f'confusion_{name}.csv'))
            preds[name] = p
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips-root', required=True); ap.add_argument('--split-images', required=True); ap.add_argument('--out', default='results_baselines')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    idx = wav_index(a.clips_root)
    tr, va, te = (split_items(a.split_images, s) for s in ('train', 'validation', 'test'))
    if not va: va = split_items(a.split_images, 'val')
    print(f'train {len(tr)}, val {len(va)}, test {len(te)} images found')
    m = sum((c, s_) in idx for c, s_, _ in te)
    print(f'{m} of {len(te)} test images matched to a WAV file by name')
    if m == 0: raise SystemExit('No test image matched a WAV file by name; the PNG stem must equal the WAV stem.')
    rows, preds = [], {}
    for kind in ('spectral', 'image'):
        print('features:', kind)
        Xtr, ytr = build(tr, idx, kind); Xva, yva = build(va, idx, kind); Xte, yte = build(te, idx, kind)
        rows += evaluate(f'LogReg_{kind}', make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced')), Xtr, ytr, Xva, yva, Xte, yte, a.out, preds)
        for seed in (42, 123, 456):
            rows += evaluate(f'RF_{kind}_seed{seed}', RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', min_samples_leaf=2, n_jobs=-1, random_state=seed), Xtr, ytr, Xva, yva, Xte, yte, a.out, preds)
        if kind == 'spectral':
            pd.DataFrame({'class': yte, 'stem': [s for c, s, p in te if (c, s) in idx], **{k: v for k, v in preds.items()}}).to_csv(os.path.join(a.out, 'test_predictions_spectral.csv'), index=False)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, 'baselines.csv'), index=False)
    pd.set_option('display.width', 250); print(df.round(4).to_string(index=False))
    print('CNN reference (same test clips): accuracy 99.75 %, Pre-CHF F1 0.951 (best run); 99.26 ± 0.29 %, 0.878 ± 0.057 over five seeds')
