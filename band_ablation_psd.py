#!/usr/bin/env python3
"""
Point 2 - support the physics with the actual sound data.

Two things in one script, both on the ORIGINAL (non-augmented) clips:

  (A) Welch power spectral density per class from the raw WAV clips
      (mean and 10-90 % band, log-log), the Pre-CHF/NB ratio in dB, and the
      Minnaert / Fritz marker frequencies (0.28, 0.5, 1.35, 2.8 kHz).
      -> figure psd_by_class.png, table psd_bands.csv

  (B) Band ablation on the TEST clips: a band is removed from the WAVEFORM
      (zero-phase Butterworth band-stop), the clip is re-rendered with the
      same STFT image pipeline as training, and the published model predicts
      again.  Also the reverse: keep ONLY one band (band-pass).
      -> table band_ablation.csv, figure band_ablation.png

Run on skuta (same env as complete_pipeline.py):

  python band_ablation_psd.py \
      --clips-root  /path/to/original_clips          # <root>/<class>/*.wav  (or nested)
      --split-images /path/to/Split_Final/STFT       # <split>/<class>/<stem>.png -> which clips are test
      --model /home/r4/sharma/new_acoustic_boiling_result/models/Original_CNN_STFT_best.h5 \
      --out results_band --selfcheck 40

--selfcheck N renders N test clips and compares them pixel by pixel with the
cached training images; it must print max diff 0.0 before the ablation
numbers are trusted.  If the renderer below is not pixel-identical, copy the
image function from complete_pipeline.py / transient_prechf_test.py into
render_stft() (same settings: SR 10 kHz, n_fft 512, hop 256, fmax 5 kHz,
dB relative to clip max, viridis, 2.24 in x 100 dpi = 224 px).
"""
import argparse, os, sys, glob, json, io
import numpy as np, pandas as pd
import soundfile as sf
import librosa, librosa.display
from scipy import signal
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({
    'font.size': 15, 'font.weight': 'bold',
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.labelsize': 16, 'axes.labelweight': 'bold',
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 13, 'lines.linewidth': 2.4,
    'axes.linewidth': 1.4, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
})

SR, NFFT, HOP, FMAX = 10000, 512, 256, 5000
CLASSES = ['BKG', 'NB', 'Pre-CHF', 'TB']
CLASS_DIRS = {'BKG': ['BKG', 'Background', 'background'], 'NB': ['NB', 'Nucleate', 'nucleate_boiling'],
              'Pre-CHF': ['Pre-CHF', 'PreCHF', 'Pre_CHF', 'prechf'], 'TB': ['TB', 'Transition', 'transition_boiling']}
BANDS = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 5000)]
MARKERS = {'NB peak 0.28 kHz': 280, 'Pre-CHF 2nd peak 0.5 kHz (~12 mm)': 500,
           'Pre-CHF peak 1.35 kHz (~5 mm)': 1350, 'Fritz bubble 2.3 mm -> 2.8 kHz': 2800}


# ----------------------------------------------------------------------------- rendering
def render_stft(y):
    """224x224 RGB uint8, meant to be pixel-identical to the training images."""
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    D = librosa.amplitude_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    librosa.display.specshow(D, sr=SR, hop_length=HOP, fmax=FMAX, cmap='viridis', ax=ax)
    buf = io.BytesIO(); fig.savefig(buf, dpi=100, format='png'); plt.close(fig); buf.seek(0)
    return np.asarray(Image.open(buf).convert('RGB').resize((224, 224)))


def load_renderer(spec):
    """--renderer module:function, e.g. transient_prechf_test:render_image.
    The function must take a 1-D float32 waveform at 10 kHz and return a 224x224x3 uint8 array.
    Only use a module whose top level does not start training when imported."""
    import importlib
    mod, fn = spec.split(':')
    f = getattr(importlib.import_module(mod), fn)
    print(f'using renderer {mod}.{fn}')
    return f


# ----------------------------------------------------------------------------- data discovery
def find_class_dir(root, cls):
    for d in CLASS_DIRS[cls]:
        hits = [p for p in glob.glob(os.path.join(root, '**', d), recursive=True) if os.path.isdir(p)]
        if hits: return hits[0]
    raise FileNotFoundError(f'no folder for class {cls} under {root}; edit CLASS_DIRS')


def wav_index(root):
    idx = {}
    for cls in CLASSES:
        d = find_class_dir(root, cls)
        for w in glob.glob(os.path.join(d, '**', '*.wav'), recursive=True):
            idx[(cls, os.path.splitext(os.path.basename(w))[0])] = w
    return idx


def split_stems(split_images, split):
    out = []
    for cls in CLASSES:
        d = find_class_dir(os.path.join(split_images, split), cls)
        for p in glob.glob(os.path.join(d, '*.png')):
            out.append((cls, os.path.splitext(os.path.basename(p))[0], p))
    return out


def load_clip(path):
    y, sr = sf.read(path, dtype='float32')
    if y.ndim > 1: y = y.mean(1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


# ----------------------------------------------------------------------------- (A) PSD
def psd_analysis(train, out):
    f = None; acc = {c: [] for c in CLASSES}
    for cls, stem, _ in train:
        y = load_clip(train_idx[(cls, stem)])
        f, P = signal.welch(y, fs=SR, nperseg=1024, noverlap=512, window='hann', scaling='density')
        acc[cls].append(P)
    stats = {}
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), layout='constrained')
    cols = {'BKG': '#4c72b0', 'NB': '#e0a526', 'Pre-CHF': '#2ca02c', 'TB': '#d62728'}
    for cls in CLASSES:
        A = 10 * np.log10(np.array(acc[cls]) + 1e-20)
        m, lo, hi = A.mean(0), np.percentile(A, 10, 0), np.percentile(A, 90, 0)
        stats[cls] = m
        ax[0].plot(f, m, color=cols[cls], lw=2.8, label=f'{cls} (n={len(acc[cls])})'); ax[0].fill_between(f, lo, hi, color=cols[cls], alpha=0.15)
    ax[0].set_xscale('log'); ax[0].set_xlim(30, 5000); ax[0].set_xlabel('Frequency (Hz)'); ax[0].set_ylabel('PSD (dB re 1 Pa²/Hz, arbitrary offset)')
    ax[0].set_title('(a) Welch PSD of the original clips, mean and 10–90 % band'); ax[0].legend(fontsize=13, loc='lower left', prop={'weight': 'bold'})
    ax[1].plot(f, stats['Pre-CHF'] - stats['NB'], color='k', lw=2.8, label='Pre-CHF − NB'); ax[1].plot(f, stats['TB'] - stats['NB'], color='#d62728', lw=2.8, ls='--', label='TB − NB')
    ax[1].axhline(0, color='grey', lw=0.9); ax[1].set_xscale('log'); ax[1].set_xlim(30, 5000); ax[1].set_xlabel('Frequency (Hz)'); ax[1].set_ylabel('Level difference (dB)')
    ax[1].set_title('(b) Spectral difference relative to NB'); ax[1].legend(fontsize=13, prop={'weight': 'bold'})
    for a in ax:
        a.tick_params(width=1.2, length=6)
        for lab in a.get_xticklabels() + a.get_yticklabels():
            lab.set_fontweight('bold')
        for lab, fm in MARKERS.items():
            a.axvline(fm, color='grey', ls=':', lw=1.1)
            a.text(fm, a.get_ylim()[1], lab.split(' ')[0] + ' ' + lab.split(' ')[1], rotation=90, va='top', ha='right', fontsize=11, fontweight='bold', color='dimgrey')
    fig.savefig(os.path.join(out, 'psd_by_class.png'), dpi=300)
    rows = []
    for lo, hi in BANDS:
        sel = (f >= lo) & (f < hi)
        r = {'band_Hz': f'{lo}-{hi}'}
        for cls in CLASSES: r[cls + '_dB'] = round(float(np.mean(stats[cls][sel])), 2)
        r['PreCHF_minus_NB_dB'] = round(r['Pre-CHF_dB'] - r['NB_dB'], 2); rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'psd_bands.csv'), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# ----------------------------------------------------------------------------- (B) band ablation
def band_filter(y, lo, hi, mode):
    """mode 'remove' = band-stop, 'keep' = band-pass. Edge bands use high/low-pass."""
    nyq = SR / 2
    if mode == 'remove':
        if lo <= 0: sos = signal.butter(4, hi / nyq, btype='highpass', output='sos')
        elif hi >= nyq: sos = signal.butter(4, lo / nyq, btype='lowpass', output='sos')
        else: sos = signal.butter(4, [lo / nyq, hi / nyq], btype='bandstop', output='sos')
    else:
        if lo <= 0: sos = signal.butter(4, hi / nyq, btype='lowpass', output='sos')
        elif hi >= nyq: sos = signal.butter(4, lo / nyq, btype='highpass', output='sos')
        else: sos = signal.butter(4, [lo / nyq, hi / nyq], btype='bandpass', output='sos')
    return signal.sosfiltfilt(sos, y).astype('float32')


def predict(model, imgs):
    x = np.stack(imgs).astype('float32') / 255.0
    return model.predict(x, batch_size=64, verbose=0)


def metrics(y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    d = {'accuracy': accuracy_score(y_true, y_pred)}
    for i, c in enumerate(CLASSES): d[f'recall_{c}'] = r[i]
    d['precision_Pre-CHF'] = p[2]; d['F1_Pre-CHF'] = f1[2]
    d['PreCHF_to_NB'] = int(sum(1 for a, b in zip(y_true, y_pred) if a == 'Pre-CHF' and b == 'NB'))
    d['NB_to_PreCHF'] = int(sum(1 for a, b in zip(y_true, y_pred) if a == 'NB' and b == 'Pre-CHF'))
    return d


def ablation(test, model, out, render):
    ys = {}; labels = []
    for cls, stem, _ in test:
        ys[(cls, stem)] = load_clip(test_idx[(cls, stem)]); labels.append(cls)
    rows = []
    conds = [('none', None, None)] + [(f'remove {lo}-{hi} Hz', (lo, hi), 'remove') for lo, hi in BANDS] + [(f'keep only {lo}-{hi} Hz', (lo, hi), 'keep') for lo, hi in BANDS]
    for name, band, mode in conds:
        imgs = []
        for (cls, stem) in ys:
            y = ys[(cls, stem)] if band is None else band_filter(ys[(cls, stem)], band[0], band[1], mode)
            imgs.append(render(y))
        P = predict(model, imgs); pred = [CLASSES[i] for i in P.argmax(1)]
        m = metrics(labels, pred); m['condition'] = name
        m['mean_p_PreCHF_on_PreCHF'] = float(P[np.array(labels) == 'Pre-CHF', 2].mean())
        rows.append(m); print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != 'condition'})
    df = pd.DataFrame(rows).set_index('condition'); df.to_csv(os.path.join(out, 'band_ablation.csv'))
    # figure
    rem = df[df.index.str.startswith('remove')]; keep = df[df.index.str.startswith('keep')]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), layout='constrained', sharey=True)
    x = np.arange(len(BANDS)); w = 0.38; lab = [f'{lo/1000:g}–{hi/1000:g}' for lo, hi in BANDS]
    for a, d, t in [(ax[0], rem, '(a) Band removed from the waveform'), (ax[1], keep, '(b) Only this band kept')]:
        a.bar(x - w / 2, 100 * d['recall_Pre-CHF'], w, color='#2ca02c', label='Pre-CHF recall')
        a.bar(x + w / 2, 100 * d['recall_NB'], w, color='#e0a526', label='NB recall')
        a.axhline(100 * df.loc['none', 'recall_Pre-CHF'], color='#2ca02c', ls=':', lw=1.4); a.axhline(100 * df.loc['none', 'recall_NB'], color='#e0a526', ls=':', lw=1.4)
        a.set_xticks(x); a.set_xticklabels(lab, fontweight='bold'); a.set_xlabel('Band (kHz)'); a.set_title(t, loc='left'); a.set_ylim(0, 105)
        a.tick_params(width=1.2, length=6)
        for lab2 in a.get_yticklabels():
            lab2.set_fontweight('bold')
    ax[0].set_ylabel('Recall on original test clips (%)'); ax[0].legend(fontsize=13, loc='lower left', prop={'weight': 'bold'})
    fig.savefig(os.path.join(out, 'band_ablation.png'), dpi=300)


# ----------------------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips-root', required=True); ap.add_argument('--split-images', required=True)
    ap.add_argument('--model', required=True); ap.add_argument('--out', default='results_band')
    ap.add_argument('--selfcheck', type=int, default=40); ap.add_argument('--skip-psd', action='store_true')
    ap.add_argument('--renderer', default=None, help='module:function of the verified training renderer (optional)')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    render = load_renderer(a.renderer) if a.renderer else render_stft
    train_idx = test_idx = wav_index(a.clips_root)
    train = split_stems(a.split_images, 'train'); test = split_stems(a.split_images, 'test')
    train = [t for t in train if (t[0], t[1]) in train_idx]; test = [t for t in test if (t[0], t[1]) in test_idx]
    print(f'{len(train)} train clips, {len(test)} test clips matched to WAV files')
    if not test or not train:
        sys.exit('No image matched a WAV file by name. The PNG stem must equal the WAV stem; check one example of each and edit split_stems()/wav_index().')
    if a.selfcheck:
        rng = np.random.default_rng(0); diffs = []
        for cls, stem, png in [test[i] for i in rng.choice(len(test), min(a.selfcheck, len(test)), replace=False)]:
            ref = np.asarray(Image.open(png).convert('RGB').resize((224, 224))).astype(int)
            diffs.append(np.abs(render(load_clip(test_idx[(cls, stem)])).astype(int) - ref).max())
        print(f'selfcheck: max pixel diff over {len(diffs)} clips = {max(diffs)}  (must be 0; small values = anti-aliasing differences, inspect before trusting)')
    if not a.skip_psd: psd_analysis(train, a.out)
    import tensorflow as tf
    model = tf.keras.models.load_model(a.model, compile=False)
    ablation(test, model, a.out, render)
    print('done ->', a.out)
