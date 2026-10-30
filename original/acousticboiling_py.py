#!/usr/bin/env python3
"""
COMPLETE BOILING PIPELINE - JOURNAL REVISION VERSION
Applied Thermal Engineering Submission

INTEGRATES:
  1. Original pipeline (WAV→Spec→Split→Augment→Train)
  2. analyze_results.py (grades, figures, findings)
  3. Missing figures generator (dataset/pipeline/examples)
  4. ALL journal revision additions:
     - Ablation study (5 experiments)
     - Augmentation sensitivity analysis
     - Statistical significance (t-test + Wilcoxon)
     - Calibration analysis + ECE
     - ROC curves (all classes + Pre-CHF vs NB binary)
     - 95% Bootstrap confidence intervals
     - Hard-negative analysis (NB vs Pre-CHF)
     - Grad-CAM with physics validation
     - Contribution section auto-generator

Run: python3 acousticboiling.py --steps all
  or: python3 acousticboiling.py --steps train analysis figures
  or: python3 acousticboiling.py --steps calibration roc ci gradcam

Optional flags: --epochs 50  --n_seeds 6  --ablation_seeds 3
                --n_bootstrap 1000  --noise_levels 50 60 70
                --kfold_splits 5  --stop_on_error

REVISION-FIX CHANGELOG (this version)
  * Checkpoints saved as native .keras (legacy .h5 frequently failed
    to reload nested ResNet/EfficientNet models -> every post-hoc
    step crashed).  safe_load_model() falls back between formats.
  * Resumed-from-checkpoint runs no longer write fake 0.0 metadata;
    history is recovered from history_*.json or marked None/resumed.
  * TF seeded in the main training loop (was python/numpy only).
  * Grad-CAM works on Keras 3 (layer.output.shape) and on nested
    pretrained bases; falls back gracefully instead of blank panels.
  * K-fold CV: originals only (no augmented leakage across folds) and
    streamed via tf.data (was ~30 GB float32 in RAM).
  * Augmentation sensitivity uses PER-CLASS targets so 1x/5x/... no
    longer truncates NB/BKG/TB down to the Pre-CHF count.
  * Ablation 'A_None' records the actual (imbalanced) train counts.
  * Bootstrap CI uses labels= so a resample missing a rare class
    cannot raise; ~10x faster than per-iteration classification_report.
  * Hard-negative 2x2 matrix no longer counts BKG/TB predictions as
    Pre-CHF ('Other' column added).
  * PR baseline = true prevalence (was hardcoded 2.5%).
  * 5-seed CI uses t-distribution (n=5 -> 2.776, not 1.96).
  * Wilcoxon: warns when n_seeds < 6 (cannot reach p<0.05).
  * Contribution text is computed from result JSONs, not asserted.
  * fig_dataset reads split_index.json instead of hardcoded counts.
  * main(): each step wrapped so one failure doesn't abort the rest;
    'contributions' runs last so it sees all result files.
  * cv2 dependency removed (PIL resize).
"""

import os, json, time, shutil, random, warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50V2, EfficientNetB0, MobileNetV3Small
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image, ImageEnhance
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================================
# JSON ENCODER FOR NUMPY TYPES
# ============================================================
# BUG FIX: the original file called json.dump(..., cls=NumpyEncoder)
# in several places (e.g. step3_train_all) but never defined this
# class anywhere, causing a NameError right after training finished
# each experiment (i.e. AFTER the expensive part was already done,
# but before results ever got saved to disk). Adding it here fixes
# that crash without changing any other behavior.
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


random.seed(42)
np.random.seed(42)

# ============================================================
# PATHS  (Skuta cluster — Linux, paths are relative to your $HOME
#  so this works no matter what your exact home directory is)
# ============================================================
# On Skuta your data lives at:
#   ~/Dataset_of_fluid_boiling/Dataset_sound_of_boiling/<BKG|NB|Pre-CHF|TB>/*.wav
# (this is the Linux-side path behind the Z:\ drive you see on Windows)

# INPUT: raw WAV dataset (read-only)
DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "Dataset_of_fluid_boiling",
    "Dataset_sound_of_boiling"
)

# OUTPUT: everything this script generates (spectrograms, splits,
# augmented data, trained models, results, figures) goes here —
# kept completely separate from the raw input dataset, as a sibling
# folder in your home directory (== Z:\coustic_boiling_again on Windows)
BASE_DIR = os.path.join(
    os.path.expanduser("~"),
    "coustic_boiling_again"
)

WAV_DIR      = DATA_DIR
SPEC_DIR     = os.path.join(BASE_DIR, "outputSpectrogram",
                             "Spectrograms")
SPLIT_ORIG   = os.path.join(BASE_DIR, "outputSpectrogram",
                             "Split_Original")
SPLIT_FINAL  = os.path.join(BASE_DIR, "outputSpectrogram",
                             "Split_Final")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
FIGURES_DIR  = os.path.join(BASE_DIR, "figures")
PAPER_FIGS   = os.path.join(BASE_DIR, "paper_writing", "figures")
ABLATION_DIR = os.path.join(BASE_DIR, "ablation")

for d in [SPEC_DIR, SPLIT_ORIG, SPLIT_FINAL,
          MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
          PAPER_FIGS, ABLATION_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
FOLDERS     = ['BKG', 'NB', 'Pre-CHF', 'TB']
CLASS_NAMES = ['BKG', 'NB', 'Pre-CHF', 'TB']
SPEC_TYPES  = ['STFT', 'MEL', 'MFCC', 'CQT', 'CHROMA']
NUM_CLASSES = 4
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
TARGET_N    = 13000

SR      = 10000
N_FFT   = 512
HOP_LEN = 256
N_MELS  = 64
N_MFCC  = 40
N_CQT   = 72
FMAX    = SR // 2

SPEC_CONFIG = {
    'STFT'  : {'cmap': 'viridis' },
    'MEL'   : {'cmap': 'magma'   },
    'MFCC'  : {'cmap': 'coolwarm'},
    'CQT'   : {'cmap': 'inferno' },
    'CHROMA': {'cmap': 'plasma'  },
}
SPEC_LABELS = {
    'STFT'  : '(a) STFT\nLinear freq scale',
    'MEL'   : '(b) MEL\nPerceptual scale',
    'MFCC'  : '(c) MFCC\nCepstral coeff',
    'CQT'   : '(d) CQT\nLog freq scale',
    'CHROMA': '(e) CHROMA\nPitch-class',
}

REGIME_COLORS = {
    'BKG'    : '#1f77b4',
    'NB'     : '#FFA500',
    'Pre-CHF': '#00cc00',
    'TB'     : '#ff0000',
}
SPEC_COLORS = {
    'STFT'  : '#1f77b4',
    'MEL'   : '#ff7f0e',
    'MFCC'  : '#2ca02c',
    'CQT'   : '#d62728',
    'CHROMA': '#9467bd',
}

PHYSICS_BANDS = {
    'BKG'    : '< 100 Hz (natural convection)',
    'NB'     : '200-250 Hz (bubble growth/collapse)',
    'Pre-CHF': '200-400 Hz (broadband increase)',
    'TB'     : '400-500 Hz (microbubble emissions)',
}
PHYSICS_SHORT = {
    'BKG'    : '< 100 Hz\n(natural convection)',
    'NB'     : '200-250 Hz\n(bubble growth)',
    'Pre-CHF': '200-400 Hz\n(broadband noise)',
    'TB'     : '400-500 Hz\n(vapor film)',
}

# Original dataset counts (from paper)
ORIG_COUNTS = {
    'BKG': 2056, 'NB': 13367,
    'Pre-CHF': 399, 'TB': 460
}

# Checkpoint format. '.keras' is the native Keras-3 format and reliably
# round-trips nested functional models (ResNet50V2 / EfficientNet base
# wrapped in an outer Model). Legacy '.h5' frequently fails to reload.
MODEL_EXT = '.keras'


# ============================================================
# SHARED HELPERS
# ============================================================
def set_all_seeds(seed):
    """Seed python, numpy AND tensorflow (previously TF was unseeded
    in the main training loop, so results were not reproducible)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_mean(vals, default=float('nan')):
    return float(np.mean(vals)) if len(vals) else default


def safe_std(vals, default=0.0):
    return float(np.std(vals)) if len(vals) else default


def safe_load_model(path):
    """
    Load a checkpoint, transparently falling back between .keras/.h5
    if the recorded path does not exist. Returns None on failure
    instead of crashing every downstream analysis step.
    """
    candidates = [path]
    root, ext = os.path.splitext(path)
    for alt in ('.keras', '.h5'):
        if alt != ext:
            candidates.append(root + alt)
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            return tf.keras.models.load_model(p, compile=False)
        except Exception as e:
            print(f"  WARNING: could not load {p}: {e}")
            try:
                # Rebuild architecture from name and load weights only
                mn = os.path.basename(root).split('_')[0:2]
                mn = '_'.join(mn)
                if mn in BUILDERS:
                    m = BUILDERS[mn](input_shape=IMG_SIZE+(3,),
                                     num_classes=NUM_CLASSES)
                    m.load_weights(p)
                    return m
            except Exception as e2:
                print(f"  WARNING: weight-only reload failed: {e2}")
    print(f"  ERROR: no loadable checkpoint for {path}")
    return None


def load_results_df(results_json=None):
    """Load all_results.json into a DataFrame (or None)."""
    if results_json is None:
        results_json = os.path.join(BASE_DIR, 'all_results.json')
    if not os.path.exists(results_json):
        print(f"  ERROR: {results_json} not found")
        return None
    with open(results_json) as f:
        res = json.load(f)
    if not res:
        print(f"  ERROR: {results_json} is empty")
        return None
    df = pd.DataFrame(res)
    if 'prechf_f1' in df.columns:
        df['prechf_f1'] = df['prechf_f1'].fillna(0.0)
    else:
        df['prechf_f1'] = 0.0
    return df


def load_best_model_and_test(results_json=None):
    """
    Common preamble for every post-hoc analysis step:
    best row, loaded model, test generator, predictions.
    Returns dict or None.
    """
    df = load_results_df(results_json)
    if df is None:
        return None
    best  = df.loc[df['test_accuracy'].idxmax()]
    model = safe_load_model(best['model_path'])
    if model is None:
        return None
    spec_type = best['spec_type']
    test_dir  = os.path.join(SPLIT_FINAL, spec_type, 'test')
    if not os.path.exists(test_dir):
        print(f"  ERROR: {test_dir} missing")
        return None
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False, classes=CLASS_NAMES)
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=0)
    return {
        'df'        : df,
        'best'      : best,
        'model'     : model,
        'spec_type' : spec_type,
        'test_dir'  : test_dir,
        'test_gen'  : test_gen,
        'y_probs'   : y_pred_probs,
        'y_pred'    : np.argmax(y_pred_probs, axis=1),
        'y_true'    : test_gen.classes,
    }


def load_split_counts():
    """Read actual per-class split counts written by step1b.
    Returns {spec: {cls: {total,train,val,test}}} or None."""
    idx_f = os.path.join(BASE_DIR, 'split_index.json')
    if not os.path.exists(idx_f):
        return None
    with open(idx_f) as f:
        return json.load(f).get('stats')

# ============================================================
# GPU SETUP
# ============================================================
def setup_gpu():
    print("\n" + "="*55)
    print("GPU / CPU CHECK")
    print("="*55)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(
                    g, True)
            print(f"  GPU: {len(gpus)} device(s) found")
        except RuntimeError as e:
            print(f"  Warning: {e}")
    else:
        print("  No GPU found - running on CPU")
    print(f"  TensorFlow: {tf.__version__}")
    print("="*55)


# ============================================================
# GRADING FUNCTION
# ============================================================
def assign_grade(row):
    """Grade based on accuracy AND Pre-CHF F1."""
    acc  = row['test_accuracy']
    f1   = row.get('prechf_f1', 0)
    if f1 is None or (isinstance(f1, float) and np.isnan(f1)):
        f1 = 0.0

    if   acc >= 99.5 and f1 >= 0.90:
        return 'A+', 'Excellent - Publication Ready'
    elif acc >= 99.0 and f1 >= 0.80:
        return 'A',  'Very Good - Minor improvements needed'
    elif acc >= 98.0 and f1 >= 0.65:
        return 'B+', 'Good - Some improvements needed'
    elif acc >= 96.0 and f1 >= 0.50:
        return 'B',  'Acceptable - Needs improvement'
    elif acc >= 93.0 and f1 >= 0.30:
        return 'C',  'Below average - Significant issues'
    else:
        return 'D',  'Poor - Major problems'


# ============================================================
# MODEL BUILDERS
# ============================================================
def build_original_cnn(
        input_shape=(224, 224, 3), num_classes=4):
    """Lightweight CNN - 0.52M params."""
    return models.Sequential([
        layers.Conv2D(32, 3, activation='relu',
                      padding='same',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation='relu',
                      padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, activation='relu',
                      padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.30),

        layers.Conv2D(256, 3, activation='relu',
                      padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.30),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.50),
        layers.Dense(num_classes,
                     activation='softmax')
    ], name='Original_CNN')


def build_resnet_finetuned(
        input_shape=(224, 224, 3), num_classes=4):
    """ResNet50V2 with last-block fine-tuning."""
    base = ResNet50V2(
        weights='imagenet', include_top=False,
        input_shape=input_shape)
    base.trainable = False
    for layer in base.layers[-30:]:
        if not isinstance(layer,
                          layers.BatchNormalization):
            layer.trainable = True

    inp = tf.keras.Input(shape=input_shape)
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(512, activation='relu')(x)
    x   = layers.Dropout(0.50)(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes,
                       activation='softmax')(x)
    return tf.keras.Model(
        inp, out, name='ResNet50V2_FT')


def build_efficientnet(
        input_shape=(224, 224, 3), num_classes=4):
    base = EfficientNetB0(
        weights='imagenet', include_top=False,
        input_shape=input_shape)
    for layer in base.layers[:-20]:
        layer.trainable = False
    for layer in base.layers[-20:]:
        if not isinstance(layer,
                          layers.BatchNormalization):
            layer.trainable = True

    inp = tf.keras.Input(shape=input_shape)
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes,
                       activation='softmax')(x)
    return tf.keras.Model(
        inp, out, name='EfficientNetB0')


def build_mobilenet(
        input_shape=(224, 224, 3), num_classes=4):
    base = MobileNetV3Small(
        weights='imagenet', include_top=False,
        input_shape=input_shape)
    base.trainable = False

    inp = tf.keras.Input(shape=input_shape)
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes,
                       activation='softmax')(x)
    return tf.keras.Model(
        inp, out, name='MobileNetV3')


BUILDERS = {
    'Original_CNN'   : build_original_cnn,
    'ResNet50V2_FT'  : build_resnet_finetuned,
}

# ============================================================
# AUGMENTATION
# ============================================================
def augment_image(img, mode='full'):
    """
    mode options:
      'none'        – no augmentation
      'mild'        – brightness + shift only
      'full'        – all 6 transforms
      'no_freqmask' – full minus frequency masking
      'no_noise'    – full minus Gaussian noise
    """
    if mode == 'none':
        return img

    arr = np.array(img).astype(np.float32)

    if mode == 'mild':
        choice = random.randint(0, 1)
    elif mode == 'no_freqmask':
        choice = random.choice([0, 1, 3, 4, 5])
    elif mode == 'no_noise':
        choice = random.choice([0, 1, 2, 3, 5])
    else:
        choice = random.randint(0, 5)

    if choice == 0:
        f   = random.uniform(0.75, 1.25)
        enh = ImageEnhance.Brightness(img)
        return enh.enhance(f)
    elif choice == 1:
        shift = random.randint(-15, 15)
        arr   = np.roll(arr, shift, axis=1)
    elif choice == 2:
        h  = arr.shape[0]
        mh = random.randint(5, 15)
        ms = random.randint(0, max(0, h - mh))
        arr[ms: ms + mh, :] = arr.mean()
    elif choice == 3:
        w  = arr.shape[1]
        mw = random.randint(5, 15)
        ms = random.randint(0, max(0, w - mw))
        arr[:, ms: ms + mw] = arr.mean()
    elif choice == 4:
        noise = np.random.normal(0, 8, arr.shape)
        arr   = np.clip(arr + noise, 0, 255)
    elif choice == 5:
        f   = random.uniform(0.8, 1.2)
        enh = ImageEnhance.Contrast(img)
        return enh.enhance(f)

    return Image.fromarray(arr.astype(np.uint8))


# ============================================================
# BUILD ARBITRARY SPLIT (for ablation / sensitivity)
# ============================================================
def build_augmented_split(spec_type, target_n,
                           aug_mode='full',
                           out_dir=None,
                           per_class_target=None):
    """
    Build train/val/test split with given augmentation.
    Val and test are ALWAYS original-only copies.

    target_n         : default per-class train target (int)
    per_class_target : optional {class: target_n} overriding
                       target_n for specific classes. Used by the
                       sensitivity study so that scaling Pre-CHF by
                       1x/5x/... does NOT also truncate NB/BKG/TB
                       (the previous behaviour, which confounded the
                       'augmentation factor' with 'dataset size').

    aug_mode == 'none': originals only, never augmented; a class is
    subsampled only if it has MORE originals than its target.
    """
    if out_dir is None:
        out_dir = os.path.join(
            ABLATION_DIR,
            f"{aug_mode}_{target_n}")

    random.seed(42)
    np.random.seed(42)

    per_class_target = per_class_target or {}

    for split in ['train', 'validation', 'test']:
        for folder in FOLDERS:
            src = os.path.join(
                SPLIT_ORIG, spec_type, split, folder)
            dst = os.path.join(
                out_dir, spec_type, split, folder)
            os.makedirs(dst, exist_ok=True)

            if not os.path.exists(src):
                print(f"  WARNING: missing source {src}")
                continue

            orig = sorted([
                f for f in os.listdir(src)
                if f.endswith('.png')])

            # Val / Test: always original only
            if split in ['validation', 'test']:
                for f in orig:
                    dst_f = os.path.join(dst, f)
                    if not os.path.exists(dst_f):
                        shutil.copy2(
                            os.path.join(src, f),
                            dst_f)
                continue

            # Train: augment / subsample
            cls_target = int(per_class_target.get(folder, target_n))
            n_orig     = len(orig)

            existing = [f for f in os.listdir(dst)
                        if f.endswith('.png')]
            n_exist  = len(existing)

            # If a previous run left MORE files than we now want
            # (e.g. target changed), trim the augmented ones first.
            if n_exist > cls_target:
                aug_existing = sorted(
                    f for f in existing if f.startswith('aug'))
                orig_existing = [f for f in existing
                                 if not f.startswith('aug')]
                to_remove = aug_existing[
                    max(0, cls_target - len(orig_existing)):]
                for f in to_remove:
                    os.remove(os.path.join(dst, f))
                existing = [f for f in os.listdir(dst)
                            if f.endswith('.png')]
                n_exist  = len(existing)

            if n_exist >= cls_target and n_exist >= min(n_orig,
                                                        cls_target):
                continue

            # Copy originals
            for f in orig:
                dst_f = os.path.join(dst, f)
                if not os.path.exists(dst_f):
                    shutil.copy2(
                        os.path.join(src, f), dst_f)

            if aug_mode == 'none' or n_orig >= cls_target:
                # Subsample originals only if too many
                all_f = [
                    f for f in os.listdir(dst)
                    if f.endswith('.png')
                    and not f.startswith('aug')]
                if len(all_f) > cls_target:
                    random.shuffle(all_f)
                    for f in all_f[cls_target:]:
                        os.remove(
                            os.path.join(dst, f))
                continue

            count   = len([f for f in os.listdir(dst)
                           if f.endswith('.png')])
            aug_idx = count
            fails   = 0
            while count < cls_target:
                sf    = random.choice(orig)
                sp    = os.path.join(src, sf)
                nn    = f"aug{aug_idx:06d}_{sf}"
                dp    = os.path.join(dst, nn)
                if os.path.exists(dp):
                    count += 1; aug_idx += 1
                    continue
                try:
                    img = Image.open(sp).convert('RGB')
                    aug = augment_image(img,
                                        mode=aug_mode)
                    aug.save(dp)
                    count += 1; aug_idx += 1
                except Exception as e:
                    fails += 1
                    if fails > 100:
                        print(f"  ERROR: too many augmentation "
                              f"failures in {dst}: {e}")
                        break

    return out_dir


# ============================================================
# DATA GENERATORS
# ============================================================
def get_generators(spec_type, split_path=None):
    """Create train / val / test generators."""
    if split_path is None:
        split_path = SPLIT_FINAL

    base = os.path.join(split_path, spec_type)

    for split in ['train', 'validation', 'test']:
        p = os.path.join(base, split)
        if not os.path.exists(p):
            print(f"  ERROR: Missing {p}")
            return None, None, None

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
    ).flow_from_directory(
        os.path.join(base, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        classes=CLASS_NAMES)

    val_gen = ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        os.path.join(base, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=CLASS_NAMES)

    test_gen = ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        os.path.join(base, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=CLASS_NAMES)

    print(f"  Train: {train_gen.samples} | "
          f"Val: {val_gen.samples} | "
          f"Test: {test_gen.samples}")
    return train_gen, val_gen, test_gen


def get_class_weights(train_gen):
    cw = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes)
    cw_dict = dict(enumerate(cw))
    print(f"\n  Class Weights:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    {cls:<10}: {cw[i]:.3f}")
    return cw_dict


# ============================================================
# LOW-LEVEL TRAIN UTILITY
# ============================================================
def _train_model_on_split(model, split_path,
                           spec_type, epochs=50,
                           model_save_path=None,
                           verbose=1):
    """
    Train on any split_path.
    Returns (test_acc, y_true, y_pred,
             y_pred_probs, history) or None.
    """
    tg, vg, te = get_generators(
        spec_type, split_path=split_path)
    if tg is None:
        return None

    cw = get_class_weights(tg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    cbs = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=0),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5, patience=5,
            min_lr=1e-6, verbose=0)
    ]
    if model_save_path:
        cbs.append(ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0))

    hist = model.fit(
        tg, epochs=epochs,
        validation_data=vg,
        callbacks=cbs,
        class_weight=cw,
        verbose=verbose)

    te.reset()
    loss, acc = model.evaluate(te, verbose=0)
    te.reset()
    y_pred_probs = model.predict(te, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = te.classes

    return acc, y_true, y_pred, y_pred_probs, hist


# ============================================================
# STEP 1A: WAV → SPECTROGRAMS
# ============================================================
def step1a_generate_spectrograms():
    try:
        import librosa
        import librosa.display
    except ImportError:
        print("ERROR: pip install librosa")
        return None

    print("\n" + "="*60)
    print("STEP 1A: WAV → SPECTROGRAMS")
    print(f"  SR={SR} | N_FFT={N_FFT} | HOP={HOP_LEN}")
    print("="*60)

    def gen_stft(y):
        D = librosa.stft(
            y, n_fft=N_FFT, hop_length=HOP_LEN)
        return librosa.amplitude_to_db(
            np.abs(D), ref=np.max)

    def gen_mel(y):
        S = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT,
            hop_length=HOP_LEN,
            n_mels=N_MELS, fmax=FMAX)
        return librosa.power_to_db(S, ref=np.max)

    def gen_mfcc(y):
        return librosa.feature.mfcc(
            y=y, sr=SR, n_mfcc=N_MFCC,
            n_fft=N_FFT, hop_length=HOP_LEN,
            fmax=FMAX)

    def gen_cqt(y):
        C = librosa.cqt(
            y, sr=SR, hop_length=HOP_LEN,
            n_bins=N_CQT, bins_per_octave=12,
            fmin=librosa.note_to_hz('C1'))
        return librosa.amplitude_to_db(
            np.abs(C), ref=np.max)

    def gen_chroma(y):
        return librosa.feature.chroma_stft(
            y=y, sr=SR, n_fft=N_FFT,
            hop_length=HOP_LEN)

    generators = {
        'STFT': gen_stft, 'MEL': gen_mel,
        'MFCC': gen_mfcc, 'CQT': gen_cqt,
        'CHROMA': gen_chroma
    }

    def save_png(data, spec_type, path):
        fig, ax = plt.subplots(
            figsize=(2.24, 2.24), dpi=100)
        fig.subplots_adjust(
            left=0, right=1, top=1, bottom=0)
        cmap = SPEC_CONFIG[spec_type]['cmap']
        kw   = dict(sr=SR, hop_length=HOP_LEN,
                    x_axis='time', cmap=cmap, ax=ax)
        if   spec_type == 'STFT':
            librosa.display.specshow(
                data, y_axis='hz', fmax=FMAX, **kw)
        elif spec_type == 'MEL':
            librosa.display.specshow(
                data, y_axis='mel', fmax=FMAX, **kw)
        elif spec_type == 'MFCC':
            librosa.display.specshow(data, **kw)
        elif spec_type == 'CQT':
            librosa.display.specshow(
                data, y_axis='cqt_hz', **kw)
        elif spec_type == 'CHROMA':
            librosa.display.specshow(
                data, y_axis='chroma', **kw)
        ax.axis('off')
        plt.savefig(path, dpi=100,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)

    total = 0
    for folder in FOLDERS:
        wav_path = os.path.join(WAV_DIR, folder)
        if not os.path.exists(wav_path):
            print(f"  Missing: {wav_path}")
            continue
        wavs = sorted([
            f for f in os.listdir(wav_path)
            if f.lower().endswith('.wav')])
        if not wavs:
            continue
        print(f"\n  {folder}: {len(wavs)} WAV files")
        errs = 0
        for i, wf in enumerate(wavs):
            try:
                y, _ = librosa.load(
                    os.path.join(wav_path, wf),
                    sr=SR, mono=True)
            except Exception:
                errs += 1; continue

            for spec_type in SPEC_TYPES:
                out_dir = os.path.join(
                    SPEC_DIR, spec_type, folder)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(
                    out_dir,
                    Path(wf).stem + '.png')
                if os.path.exists(out_path):
                    continue
                try:
                    data = generators[spec_type](y)
                    save_png(data, spec_type,
                             out_path)
                    total += 1
                except Exception:
                    errs += 1

            pct = (i+1)/len(wavs)*100
            bar = ('█'*int(pct//5) +
                   '░'*(20-int(pct//5)))
            print(f"    [{bar}] {pct:5.1f}% "
                  f"({i+1}/{len(wavs)}) "
                  f"err={errs}",
                  end='\r')
        print(f"\n    Done | errors: {errs}")

    print(f"\n  Total PNGs: {total}")
    return total


# ============================================================
# STEP 1B: SPLIT ORIGINALS
# ============================================================
def step1b_split_originals(
        train=0.70, val=0.15, test=0.15):
    print("\n" + "="*60)
    print("STEP 1B: SPLIT ORIGINALS")
    print("  CRITICAL: Split BEFORE augmentation!")
    print(f"  Ratio: {int(train*100)}/"
          f"{int(val*100)}/{int(test*100)}")
    print("="*60)

    random.seed(42)
    split_stats = {}

    for spec in SPEC_TYPES:
        split_stats[spec] = {}
        print(f"\n  {spec}")
        for folder in FOLDERS:
            src = os.path.join(
                SPEC_DIR, spec, folder)
            if not os.path.exists(src):
                continue
            imgs = sorted([
                f for f in os.listdir(src)
                if f.endswith('.png')])
            if not imgs:
                continue
            random.shuffle(imgs)
            n     = len(imgs)
            n_tr  = int(n * train)
            n_val = int(n * val)
            splits = {
                'train'     : imgs[:n_tr],
                'validation': imgs[n_tr: n_tr+n_val],
                'test'      : imgs[n_tr+n_val:]
            }
            for sname, files in splits.items():
                dst = os.path.join(
                    SPLIT_ORIG, spec, sname, folder)
                os.makedirs(dst, exist_ok=True)
                for f in files:
                    dst_f = os.path.join(dst, f)
                    if not os.path.exists(dst_f):
                        shutil.copy2(
                            os.path.join(src, f),
                            dst_f)
            n_test = n - n_tr - n_val
            split_stats[spec][folder] = {
                'total': n,
                'train': n_tr,
                'val'  : n_val,
                'test' : n_test
            }
            print(f"    {folder:<10}: total={n} | "
                  f"train={n_tr} | "
                  f"val={n_val} | "
                  f"test={n_test}")

    idx_f = os.path.join(BASE_DIR, 'split_index.json')
    with open(idx_f, 'w') as f:
        json.dump({'ratios': {'train': train,
                              'val': val,
                              'test': test},
                   'seed': 42,
                   'stats': split_stats}, f, indent=2)
    print(f"\n  Split index saved: {idx_f}")
    return split_stats


# ============================================================
# STEP 1C: AUGMENT TRAINING SET ONLY
# ============================================================
def step1c_augment_train_only():
    """
    Build SPLIT_FINAL from SPLIT_ORIG: train augmented to TARGET_N per
    class ('full' mode), val/test copied as originals only.
    Delegates to build_augmented_split() so the main pipeline and the
    ablation/sensitivity splits are produced by identical code.
    """
    print("\n" + "="*60)
    print("STEP 1C: AUGMENT TRAINING SET ONLY")
    print(f"  Target: {TARGET_N}/class (train)")
    print("  Val/Test: NO augmentation")
    print("="*60)

    for spec in SPEC_TYPES:
        print(f"\n  {spec}")
        if not os.path.exists(os.path.join(SPLIT_ORIG, spec)):
            print(f"    missing {SPLIT_ORIG}/{spec} - skip")
            continue
        build_augmented_split(
            spec_type = spec,
            target_n  = TARGET_N,
            aug_mode  = 'full',
            out_dir   = SPLIT_FINAL)
        for folder in FOLDERS:
            p = os.path.join(SPLIT_FINAL, spec, 'train', folder)
            n = (len([f for f in os.listdir(p) if f.endswith('.png')])
                 if os.path.exists(p) else 0)
            print(f"    train/{folder}: {n}")


# ============================================================
# STEP 2: DATASET CHECK
# ============================================================
def step2_check():
    print("\n" + "="*60)
    print("STEP 2: DATASET INTEGRITY CHECK")
    print("="*60)

    ok = True
    for spec in SPEC_TYPES:
        print(f"\n  {spec}:")
        for split in ['train', 'validation', 'test']:
            total = 0
            for folder in FOLDERS:
                p = os.path.join(
                    SPLIT_FINAL, spec, split, folder)
                n = (len([f for f in os.listdir(p)
                           if f.endswith('.png')])
                     if os.path.exists(p) else 0)
                total += n
                if n == 0:
                    print(f"    EMPTY: {split}/{folder}")
                    ok = False
                if split in ['validation', 'test']:
                    aug_f = (
                        [f for f in os.listdir(p)
                         if f.startswith('aug')]
                        if os.path.exists(p) else [])
                    if aug_f:
                        print(f"    LEAKAGE: "
                              f"{split}/{folder} has "
                              f"{len(aug_f)} aug files!")
                        ok = False
            print(f"    {split}: total={total}")

    print(f"\n  Result: {'PASSED' if ok else 'FAILED'}")
    return ok


# ============================================================
# STEP 3: TRAIN ALL COMBINATIONS
# ============================================================
def step3_train_all(spec_types=None,
                    model_names=None,
                    epochs=50):
    if spec_types  is None: spec_types  = SPEC_TYPES
    if model_names is None: model_names = list(
        BUILDERS.keys())

    results_file = os.path.join(
        BASE_DIR, 'all_results.json')
    all_results  = []
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)

    done  = {(r['model_name'], r['spec_type'])
             for r in all_results}
    total = len(model_names) * len(spec_types)
    count = 0

    print(f"\n  Total: {total} experiments")

    for mn in model_names:
        for st in spec_types:
            count += 1
            if (mn, st) in done:
                print(f"\n  [{count}/{total}] "
                      f"SKIP: {mn}+{st}")
                continue

            print(f"\n{'='*60}")
            print(f"  [{count}/{total}] {mn}+{st}")
            print(f"{'='*60}")

            if mn not in BUILDERS:
                print(f"  ERROR: unknown model {mn}")
                continue

            save_path = os.path.join(
                MODELS_DIR,
                f"{mn}_{st}_best{MODEL_EXT}")
            hist_file = os.path.join(
                RESULTS_DIR, f"history_{mn}_{st}.json")

            set_all_seeds(42)

            tg, vg, te = get_generators(st)
            if tg is None:
                continue

            # RESUME SUPPORT: if a checkpoint from an earlier (possibly
            # crashed-before-saving-JSON) run already exists on disk,
            # load and evaluate it instead of retraining from scratch.
            # Metadata (val acc / epochs / time) is recovered from the
            # saved history JSON when available; otherwise it is
            # recorded as None + 'resumed': True so it cannot silently
            # pollute paper tables with 0.0 placeholders.
            model    = None
            hist     = None
            tt       = None
            resumed  = False
            val_acc_list = None
            acc_list     = None

            if os.path.exists(save_path) or os.path.exists(
                    os.path.splitext(save_path)[0] + '.h5'):
                print(f"  Found existing checkpoint: {save_path}")
                print(f"  Loading it instead of retraining...")
                model = safe_load_model(save_path)
                if model is not None:
                    resumed = True
                    if os.path.exists(hist_file):
                        with open(hist_file) as f:
                            h = json.load(f)
                        val_acc_list = h.get('val_accuracy')
                        acc_list     = h.get('accuracy')
                        tt           = h.get('_train_time_sec')
                else:
                    print("  Could not load checkpoint — "
                          "retraining from scratch.")

            if model is None:
                model = BUILDERS[mn](
                    input_shape=IMG_SIZE+(3,),
                    num_classes=NUM_CLASSES)

                cw = get_class_weights(tg)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        1e-3),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

                t0   = time.time()
                hist = model.fit(
                    tg, epochs=epochs,
                    validation_data=vg,
                    callbacks=[
                        EarlyStopping(
                            monitor='val_accuracy',
                            patience=10,
                            restore_best_weights=True,
                            verbose=0),
                        ModelCheckpoint(
                            save_path,
                            monitor='val_accuracy',
                            save_best_only=True,
                            verbose=0),
                        ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5, patience=5,
                            min_lr=1e-6, verbose=0)
                    ],
                    class_weight=cw, verbose=1)
                tt = time.time() - t0
                val_acc_list = [float(v) for v in
                                hist.history['val_accuracy']]
                acc_list     = [float(v) for v in
                                hist.history['accuracy']]

                # Save history immediately (before evaluation, so a
                # crash later cannot lose it)
                clean_hist = {k: [float(v) for v in vs]
                              for k, vs in hist.history.items()}
                clean_hist['_train_time_sec'] = tt
                with open(hist_file, 'w') as f:
                    json.dump(clean_hist, f, indent=2)

            te.reset()
            loss, acc = model.evaluate(te, verbose=0)
            te.reset()
            ypp = model.predict(te, verbose=0)
            yp  = np.argmax(ypp, axis=1)
            yt  = te.classes

            report = classification_report(
                yt, yp,
                target_names=CLASS_NAMES,
                output_dict=True, zero_division=0)
            pcf1 = {
                c: round(report[c]['f1-score'], 4)
                for c in CLASS_NAMES if c in report}

            # Inference time (single image, warm-up excluded)
            te.reset()
            sample = next(iter(te))[0][:1]
            model.predict(sample, verbose=0)      # warm-up
            t1     = time.time()
            for _  in range(50):
                model.predict(sample, verbose=0)
            inf_ms = (time.time()-t1)/50*1000

            result = {
                'model_name'    : mn,
                'spec_type'     : st,
                'test_accuracy' : round(acc*100, 2),
                'test_loss'     : round(loss, 4),
                'best_val_acc'  : (round(max(val_acc_list)*100, 2)
                                   if val_acc_list else None),
                'epochs_run'    : (len(acc_list)
                                   if acc_list else None),
                'train_time_min': (round(tt/60, 1)
                                   if tt is not None else None),
                'inf_time_ms'   : round(inf_ms, 2),
                'params_M'      : round(
                    model.count_params()/1e6, 2),
                'model_path'    : save_path,
                'per_class_f1'  : pcf1,
                'prechf_f1'     : pcf1.get(
                    'Pre-CHF', 0),
                'resumed'       : resumed,
            }
            all_results.append(result)
            with open(results_file, 'w') as f:
                json.dump(all_results, f,
                          indent=2, cls=NumpyEncoder)

            print(f"\n  Acc={result['test_accuracy']}%"
                  f" | Pre-CHF "
                  f"F1={result['prechf_f1']:.4f}"
                  f" | Time={result['train_time_min']}"
                  f"min"
                  f"{' (resumed)' if resumed else ''}")
            del model
            tf.keras.backend.clear_session()

    return all_results


# ============================================================
# STEP 4: ANALYZE RESULTS + GRADES
# ============================================================
def step4_analyze_results():
    """Load results, assign grades, print findings,
    generate analysis figures."""
    print("\n" + "="*60)
    print("STEP 4: ANALYZE RESULTS")
    print("="*60)

    df = load_results_df()
    if df is None:
        return None

    # Fill optional / possibly-None columns so formatting never sees NaN
    for col, default in [('best_val_acc', np.nan), ('epochs_run', 0),
                         ('train_time_min', np.nan),
                         ('inf_time_ms', np.nan), ('params_M', np.nan),
                         ('resumed', False)]:
        if col not in df.columns:
            df[col] = default
    df['prechf_f1'] = pd.to_numeric(df['prechf_f1'],
                                    errors='coerce').fillna(0.0)

    df = df.sort_values('test_accuracy',
                        ascending=False
                        ).reset_index(drop=True)

    df['grade'], df['grade_note'] = zip(
        *df.apply(assign_grade, axis=1))

    print(f"\n  Total experiments: {len(df)}")
    if df['resumed'].fillna(False).astype(bool).any():
        print("  NOTE: rows marked 'R' were evaluated from a resumed "
              "checkpoint; their Val%/epochs/time may be missing.")

    def _f(v, fmt):
        """Format a number or '-' if missing."""
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return '-'
            return format(v, fmt)
        except Exception:
            return '-'

    # ── Print table ──
    print("\n" + "="*95)
    print("COMPLETE RESULTS TABLE WITH GRADES")
    print("="*95)
    print(f"{'#':<3} {'Model':<18} {'Spec':<8} "
          f"{'Acc%':>7} {'Val%':>7} "
          f"{'PreCHF_F1':>10} {'Params':>7} "
          f"{'Inf(ms)':>8} {'Grade':>6}")
    print("-"*95)

    for i, row in df.iterrows():
        star = " ← BEST" if i == 0 else ""
        res  = " R" if bool(row.get('resumed', False)) else ""
        print(
            f"{i+1:<3}"
            f"{row['model_name']:<18}"
            f"{row['spec_type']:<8}"
            f"{_f(row['test_accuracy'], '.2f'):>7}"
            f"{_f(row['best_val_acc'], '.2f'):>7}"
            f"{_f(row['prechf_f1'], '.4f'):>10}"
            f"{_f(row['params_M'], '.2f'):>7}"
            f"{_f(row['inf_time_ms'], '.2f'):>8}"
            f"{row['grade']:>6}"
            f"{star}{res}")

    # ── Key findings ──
    best     = df.iloc[0]
    cnn_df   = df[df['model_name'] == 'Original_CNN']

    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    print(f"\n1. BEST MODEL:")
    print(f"   {best['model_name']} + "
          f"{best['spec_type']}")
    print(f"   Accuracy : {best['test_accuracy']}%")
    print(f"   Pre-CHF  : "
          f"{best.get('prechf_f1',0):.4f}")
    print(f"   Grade    : {best['grade']} - "
          f"{best['grade_note']}")

    print(f"\n2. SPECTROGRAM RANKING (Original CNN):")
    cnn_s = cnn_df.sort_values(
        'test_accuracy', ascending=False)
    for i, (_, row) in enumerate(cnn_s.iterrows()):
        print(f"   {i+1}. {row['spec_type']:<8}: "
              f"Acc={row['test_accuracy']:5.2f}%  "
              f"Pre-CHF_F1="
              f"{row.get('prechf_f1',0):.4f}  "
              f"Grade={row['grade']}")

    print(f"\n3. ARCHITECTURE COMPARISON:")
    for mn in df['model_name'].unique():
        sub    = df[df['model_name'] == mn]
        avg_a  = sub['test_accuracy'].mean()
        avg_f1 = sub.get(
            'prechf_f1',
            pd.Series([0]*len(sub))).mean()
        pm     = sub['params_M'].iloc[0]
        print(f"   {mn:<20}: "
              f"Avg_Acc={avg_a:5.2f}%  "
              f"Avg_PreCHF={avg_f1:.4f}  "
              f"Params={pm:.2f}M")

    print(f"\n4. PRE-CHF vs LITERATURE:")
    print(f"   Sinha et al. 2021 : 0.0000")
    print(f"   YOUR best         : "
          f"{df['prechf_f1'].max():.4f}")
    print(f"   Improvement       : "
          f"+{df['prechf_f1'].max():.4f}")
    print(f"   NOTE: Test set uses original "
          f"images only (no augmentation)")

    # ── Paper sentences ──
    best_cnn = (cnn_df.sort_values('test_accuracy')
                .iloc[-1] if len(cnn_df) > 0 else best)
    print("\n" + "="*60)
    print("PAPER-READY SENTENCES")
    print("="*60)
    stft_max_acc = df[df['spec_type']=='STFT'][
        'test_accuracy'].max() if len(
        df[df['spec_type']=='STFT'])>0 else 0
    stft_max_f1  = df[df['spec_type']=='STFT'][
        'prechf_f1'].max() if len(
        df[df['spec_type']=='STFT'])>0 else 0
    mel_max  = df[df['spec_type']=='MEL'][
        'test_accuracy'].max() if len(
        df[df['spec_type']=='MEL'])>0 else 0
    mfcc_max = df[df['spec_type']=='MFCC'][
        'test_accuracy'].max() if len(
        df[df['spec_type']=='MFCC'])>0 else 0
    cqt_max  = df[df['spec_type']=='CQT'][
        'test_accuracy'].max() if len(
        df[df['spec_type']=='CQT'])>0 else 0
    chr_max  = df[df['spec_type']=='CHROMA'][
        'test_accuracy'].max() if len(
        df[df['spec_type']=='CHROMA'])>0 else 0

    print(f"""
ABSTRACT:
  "The lightweight CNN ({best_cnn['params_M']:.2f}M params)
   achieved {best_cnn['test_accuracy']:.2f}% accuracy with
   {best_cnn['spec_type']} spectrogram and Pre-CHF F1
   of {best_cnn.get('prechf_f1',0):.4f}, compared to
   complete failure (F1=0.000) in prior work."

RESULTS:
  "STFT yielded the highest accuracy ({stft_max_acc:.2f}%)
   and Pre-CHF F1 ({stft_max_f1:.4f}), followed by MEL
   ({mel_max:.2f}%), MFCC ({mfcc_max:.2f}%), CQT
   ({cqt_max:.2f}%), and CHROMA ({chr_max:.2f}%)."

LIMITATION (MUST INCLUDE):
  "The Pre-CHF improvement is substantially attributed
   to augmentation-based class balancing (279 original
   → 13,000 samples). The test set comprised
   approximately 60 original Pre-CHF recordings.
   Further validation on additional experimental data
   is recommended."
""")

    # ── Figures ──
    _fig_complete_analysis(df)
    _fig_cnn_comparison(df)
    _fig_perclass_f1(df)

    # Save graded results
    df.to_csv(
        os.path.join(RESULTS_DIR,
                     'results_graded.csv'),
        index=False)
    print(f"\n  Graded CSV saved: "
          f"{RESULTS_DIR}/results_graded.csv")
    return df


def _fig_complete_analysis(df):
    """Complete analysis figure (heatmaps + scatter)."""
    ml = df['model_name'].unique().tolist()
    sl = SPEC_TYPES

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        'Complete Boiling Regime Classification\n'
        f'({len(df)} experiments | '
        'Split 70/15/15 before augmentation)',
        fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(
        3, 3, hspace=0.5, wspace=0.4, figure=fig)

    # Accuracy heatmap
    ax1    = fig.add_subplot(gs[0, :])
    hm_acc = np.full((len(ml), len(sl)), np.nan)
    for i, m in enumerate(ml):
        for j, s in enumerate(sl):
            r = df[(df['model_name']==m) &
                   (df['spec_type']==s)]
            if not r.empty:
                hm_acc[i, j] = r[
                    'test_accuracy'].values[0]

    mask    = np.isnan(hm_acc)
    hm_plot = np.where(mask, 0, hm_acc)
    im1 = ax1.imshow(
        hm_plot, cmap='RdYlGn',
        aspect='auto', vmin=80, vmax=100)
    plt.colorbar(im1, ax=ax1,
                 label='Test Accuracy (%)',
                 fraction=0.02, pad=0.02)
    ax1.set_xticks(range(len(sl)))
    ax1.set_yticks(range(len(ml)))
    ax1.set_xticklabels(sl, fontsize=11)
    ax1.set_yticklabels(ml, fontsize=10)
    ax1.set_title(
        'Test Accuracy Heatmap '
        '(Model × Spectrogram)',
        fontsize=12, fontweight='bold')

    for i in range(len(ml)):
        for j in range(len(sl)):
            if not mask[i, j]:
                v  = hm_acc[i, j]
                gr = df[(df['model_name']==ml[i]) &
                        (df['spec_type']==sl[j])
                        ]['grade'].values[0]
                c  = ('white' if v < 90
                      else 'black')
                ax1.text(
                    j, i,
                    f'{v:.2f}%\n({gr})',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color=c)
            else:
                ax1.text(
                    j, i, 'Pending',
                    ha='center', va='center',
                    fontsize=8, color='gray',
                    style='italic')

    # Pre-CHF F1 heatmap
    ax2   = fig.add_subplot(gs[1, :])
    hm_f1 = np.full((len(ml), len(sl)), np.nan)
    for i, m in enumerate(ml):
        for j, s in enumerate(sl):
            r = df[(df['model_name']==m) &
                   (df['spec_type']==s)]
            if not r.empty:
                hm_f1[i, j] = r.get(
                    'prechf_f1',
                    pd.Series([0])).values[0]

    hm_f1_plot = np.where(np.isnan(hm_f1), 0, hm_f1)
    im2 = ax2.imshow(
        hm_f1_plot, cmap='RdYlGn',
        aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2,
                 label='Pre-CHF F1-Score',
                 fraction=0.02, pad=0.02)
    ax2.set_xticks(range(len(sl)))
    ax2.set_yticks(range(len(ml)))
    ax2.set_xticklabels(sl, fontsize=11)
    ax2.set_yticklabels(ml, fontsize=10)
    ax2.set_title(
        'Pre-CHF F1 Heatmap '
        '(Prior work = 0.000)',
        fontsize=12, fontweight='bold')

    for i in range(len(ml)):
        for j in range(len(sl)):
            if not np.isnan(hm_f1[i, j]):
                v = hm_f1[i, j]
                c = ('white' if v < 0.5
                     else 'black')
                ax2.text(
                    j, i, f'{v:.4f}',
                    ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color=c)
            else:
                ax2.text(
                    j, i, 'Pending',
                    ha='center', va='center',
                    fontsize=8, color='gray',
                    style='italic')

    # Speed vs accuracy
    ax3  = fig.add_subplot(gs[2, 0])
    cm_l = ['#1f77b4','#ff7f0e','#2ca02c',
            '#d62728','#9467bd']
    mkrs = {'Original_CNN'   : 'o',
            'ResNet50V2_FT'  : 's',
            'EfficientNetB0' : '^',
            'MobileNetV3'    : 'D'}
    for i, m in enumerate(ml):
        sub = df[df['model_name'] == m]
        if 'inf_time_ms' not in sub.columns:
            continue
        ax3.scatter(
            sub['inf_time_ms'],
            sub['test_accuracy'],
            label=m, s=100, alpha=0.85,
            marker=mkrs.get(m, 'o'),
            color=cm_l[i % len(cm_l)],
            edgecolors='black', linewidth=0.5)
    ax3.axvline(x=100, color='red',
                linestyle='--', linewidth=1.5,
                alpha=0.7)
    ax3.set_xlabel('Inference Time (ms)',
                   fontsize=10)
    ax3.set_ylabel('Test Accuracy (%)',
                   fontsize=10)
    ax3.set_title('Speed vs Accuracy',
                  fontsize=10, fontweight='bold')
    ax3.legend(fontsize=7, loc='lower right')
    ax3.grid(True, alpha=0.3)

    # Pre-CHF F1 per model/spec bar
    ax4 = fig.add_subplot(gs[2, 1])
    x4  = np.arange(len(ml))
    w4  = 0.15
    for j, s in enumerate(sl):
        vals = []
        for m in ml:
            r = df[(df['model_name']==m) &
                   (df['spec_type']==s)]
            vals.append(
                r['prechf_f1'].values[0]
                if not r.empty else 0)
        ax4.bar(
            x4 + (j-2)*w4, vals, w4,
            label=s,
            color=SPEC_COLORS[s],
            alpha=0.85,
            edgecolor='black', linewidth=0.5)
    ax4.axhline(y=0, color='red',
                linestyle='--', linewidth=2)
    ax4.axhline(y=0.9, color='green',
                linestyle=':', linewidth=1.5,
                alpha=0.7)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(
        [m.replace('_', '\n') for m in ml],
        fontsize=8)
    ax4.set_ylabel('Pre-CHF F1-Score',
                   fontsize=10)
    ax4.set_title('Pre-CHF F1 by Model & Spec',
                  fontsize=10, fontweight='bold')
    ax4.legend(fontsize=7)
    ax4.set_ylim(0, 1.15)
    ax4.grid(True, alpha=0.3, axis='y')

    # Grade pie
    ax5          = fig.add_subplot(gs[2, 2])
    grade_counts = df['grade'].value_counts()
    gcolors      = {
        'A+': '#00aa00', 'A': '#55cc00',
        'B+': '#ffaa00', 'B': '#ff7700',
        'C' : '#ff4400', 'D': '#cc0000'}
    pie_c = [gcolors.get(g, '#999999')
             for g in grade_counts.index]
    wedges, texts, autotexts = ax5.pie(
        grade_counts.values,
        labels=grade_counts.index,
        colors=pie_c, autopct='%1.0f%%',
        startangle=90,
        textprops={'fontsize': 11})
    for at in autotexts:
        at.set_fontweight('bold')
    ax5.set_title(
        f'Grade Distribution\n({len(df)} exps)',
        fontsize=10, fontweight='bold')

    plt.savefig(
        os.path.join(FIGURES_DIR,
                     'complete_analysis.png'),
        dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: complete_analysis.png")


def _fig_cnn_comparison(df):
    """CNN spectrogram comparison figure."""
    cnn_data = df[
        df['model_name'] == 'Original_CNN'
    ].sort_values('test_accuracy', ascending=False)

    if cnn_data.empty:
        return

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(
        'Original CNN: Spectrogram Comparison\n'
        '(0.52M parameters | Real-time capable)',
        fontsize=13, fontweight='bold')

    # Accuracy
    ax21 = axes2[0]
    bars = ax21.bar(
        cnn_data['spec_type'],
        cnn_data['test_accuracy'],
        color=[SPEC_COLORS[s]
               for s in cnn_data['spec_type']],
        edgecolor='black', linewidth=1.2)
    ax21.set_title('Test Accuracy (%)',
                   fontsize=11, fontweight='bold')
    ax21.set_ylabel('Accuracy (%)')
    ymin = max(90, cnn_data['test_accuracy'].min()-2)
    ax21.set_ylim(ymin, 101)
    ax21.grid(True, alpha=0.3, axis='y')
    for bar, (_, row) in zip(
            bars, cnn_data.iterrows()):
        ax21.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"{row['test_accuracy']:.2f}%"
            f"\n({row['grade']})",
            ha='center', va='bottom',
            fontsize=9, fontweight='bold')

    # Pre-CHF F1
    ax22  = axes2[1]
    f1vals = cnn_data['prechf_f1'].fillna(0)
    bars2  = ax22.bar(
        cnn_data['spec_type'], f1vals,
        color=[SPEC_COLORS[s]
               for s in cnn_data['spec_type']],
        edgecolor='black', linewidth=1.2)
    ax22.axhline(y=0.0, color='red',
                 linestyle='--', linewidth=2,
                 label='Prior work: 0.000')
    ax22.axhline(y=0.9, color='green',
                 linestyle=':', linewidth=1.5,
                 label='Target: 0.90', alpha=0.8)
    ax22.set_title('Pre-CHF F1-Score\n'
                   '(Prior work = 0.000)',
                   fontsize=11, fontweight='bold')
    ax22.set_ylabel('F1-Score')
    ax22.set_ylim(0, 1.15)
    ax22.legend(fontsize=9)
    ax22.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, f1vals):
        ax22.text(
            bar.get_x() + bar.get_width()/2,
            val + 0.02,
            f'{val:.4f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold')

    # Training time (None/NaN for resumed rows -> plotted as 0, 'n/a')
    ax23  = axes2[2]
    tt    = pd.to_numeric(cnn_data['train_time_min'],
                          errors='coerce').fillna(0)
    ep    = pd.to_numeric(cnn_data['epochs_run'],
                          errors='coerce')
    bars3 = ax23.bar(
        cnn_data['spec_type'], tt,
        color=[SPEC_COLORS[s]
               for s in cnn_data['spec_type']],
        edgecolor='black', linewidth=1.2)
    ax23.set_title('Training Time (min)',
                   fontsize=11, fontweight='bold')
    ax23.set_ylabel('Time (min)')
    ax23.grid(True, alpha=0.3, axis='y')
    for bar, t, e in zip(bars3, tt, ep):
        lbl = ("n/a" if t == 0 else
               f"{t:.0f}min\n({int(e) if not np.isnan(e) else '?'}ep)")
        ax23.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1, lbl,
            ha='center', va='bottom',
            fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR,
                     'cnn_spectrogram_comparison.png'),
        dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: cnn_spectrogram_comparison.png")


def _fig_perclass_f1(df):
    """Per-class F1 analysis figure."""
    cnn_data = df[
        df['model_name'] == 'Original_CNN'
    ].sort_values('test_accuracy', ascending=False)

    if cnn_data.empty:
        return

    fig3, axes3 = plt.subplots(
        1, 2, figsize=(16, 6))
    fig3.suptitle(
        'Per-Class F1 Analysis',
        fontsize=13, fontweight='bold')

    # Per-class F1 grouped bar
    ax31     = axes3[0]
    cls_data = {c: [] for c in CLASS_NAMES}
    for _, row in cnn_data.iterrows():
        f1d = row.get('per_class_f1', {})
        for cls in CLASS_NAMES:
            cls_data[cls].append(f1d.get(cls, 0))

    x3 = np.arange(len(cnn_data))
    w3 = 0.2
    for i, cls in enumerate(CLASS_NAMES):
        ax31.bar(
            x3 + (i-1.5)*w3,
            cls_data[cls], w3,
            label=cls,
            color=REGIME_COLORS[cls],
            alpha=0.85,
            edgecolor='black', linewidth=0.5)
    ax31.set_xticks(x3)
    ax31.set_xticklabels(
        cnn_data['spec_type'], fontsize=10)
    ax31.set_ylabel('F1-Score', fontsize=11)
    ax31.set_title('Original CNN: Per-Class F1',
                   fontsize=11, fontweight='bold')
    ax31.set_ylim(0, 1.15)
    ax31.legend(fontsize=9)
    ax31.grid(True, alpha=0.3, axis='y')
    ax31.axhline(y=0.0, color='red',
                 linestyle='--', linewidth=1.5,
                 alpha=0.5)

    # CNN vs ResNet Pre-CHF F1
    ax32       = axes3[1]
    cnn_f1s    = []
    res_f1s    = []
    specs_done = []
    for s in SPEC_TYPES:
        rc = df[(df['model_name']=='Original_CNN') &
                (df['spec_type']==s)]
        rr = df[(df['model_name']=='ResNet50V2_FT') &
                (df['spec_type']==s)]
        if not rc.empty and not rr.empty:
            cnn_f1s.append(rc['prechf_f1'].values[0])
            res_f1s.append(rr['prechf_f1'].values[0])
            specs_done.append(s)

    if specs_done:
        x32 = np.arange(len(specs_done))
        w32 = 0.35
        ax32.bar(x32-w32/2, cnn_f1s, w32,
                 label='Original CNN (0.52M)',
                 color='#1f77b4',
                 edgecolor='black', linewidth=0.8)
        ax32.bar(x32+w32/2, res_f1s, w32,
                 label='ResNet50V2_FT (24.75M)',
                 color='#ff7f0e',
                 edgecolor='black', linewidth=0.8)
        ax32.set_xticks(x32)
        ax32.set_xticklabels(specs_done, fontsize=10)
        ax32.set_ylabel('Pre-CHF F1-Score',
                        fontsize=11)
        ax32.set_title(
            'CNN vs ResNet: Pre-CHF F1\n'
            '(Lighter model wins!)',
            fontsize=11, fontweight='bold')
        ax32.set_ylim(0, 1.15)
        ax32.legend(fontsize=9)
        ax32.grid(True, alpha=0.3, axis='y')
        ax32.axhline(y=0.0, color='red',
                     linestyle='--', linewidth=1.5,
                     alpha=0.6)
        for i, (cv, rv) in enumerate(
                zip(cnn_f1s, res_f1s)):
            ax32.text(i-w32/2, cv+0.01, f'{cv:.3f}',
                      ha='center', fontsize=8,
                      fontweight='bold',
                      color='#1f77b4')
            ax32.text(i+w32/2, rv+0.01, f'{rv:.3f}',
                      ha='center', fontsize=8,
                      fontweight='bold',
                      color='#ff7f0e')

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR,
                     'perclass_f1_analysis.png'),
        dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: perclass_f1_analysis.png")


# ============================================================
# STEP 5: MISSING FIGURES (dataset / pipeline / examples)
# ============================================================
def step5_missing_figures():
    """Generate the 3 figures for paper sections."""
    print("\n" + "="*60)
    print("STEP 5: PAPER FIGURES")
    print("="*60)

    _fig_dataset()
    _fig_pipeline()
    _fig_spectrogram_examples()


def _fig_dataset():
    """Dataset distribution figure."""
    print("  Making fig_dataset.png...")

    # Prefer the ACTUAL counts written by step1b (split_index.json)
    # over the hardcoded paper numbers, so figure and text agree.
    stats     = load_split_counts()
    ref_spec  = 'STFT'
    counts    = None
    if stats and ref_spec in stats and all(
            c in stats[ref_spec] for c in CLASS_NAMES):
        counts = stats[ref_spec]
        orig   = [counts[c]['total'] for c in CLASS_NAMES]
        tr     = [counts[c]['train'] for c in CLASS_NAMES]
        va     = [counts[c]['val']   for c in CLASS_NAMES]
        te     = [counts[c]['test']  for c in CLASS_NAMES]
        print(f"  Using actual split counts from split_index.json")
    else:
        orig = [ORIG_COUNTS[c] for c in CLASS_NAMES]
        tr   = [1439, 9357, 279, 322]
        va   = [308,  2005,  60,  69]
        te   = [309,  2005,  60,  69]
        print(f"  WARNING: split_index.json not found — "
              f"using hardcoded paper counts")

    # Actual augmented train counts if the final split exists
    aug_tr = []
    for c in CLASS_NAMES:
        p = os.path.join(SPLIT_FINAL, ref_spec, 'train', c)
        aug_tr.append(
            len([f for f in os.listdir(p) if f.endswith('.png')])
            if os.path.exists(p) else TARGET_N)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Dataset Composition and Splitting Strategy\n'
        'Source: Sinha et al. (2023) '
        '[DOI: 10.17632/vnnzc7n97m.1]',
        fontsize=13, fontweight='bold')
    bars1 = axes[0].bar(
        CLASS_NAMES, orig,
        color=[REGIME_COLORS[c] for c in CLASS_NAMES],
        edgecolor='black', linewidth=1.2, width=0.6)
    axes[0].set_title(
        '(a) Original Dataset\n'
        '(Severely Imbalanced)',
        fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Number of Files',
                       fontsize=11)
    axes[0].set_yscale('log')
    for bar, val in zip(bars1, orig):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            val*1.15, f'{val:,}',
            ha='center', fontsize=10,
            fontweight='bold')
    axes[0].set_ylim(100, 50000)
    axes[0].grid(True, alpha=0.3, axis='y')
    pc_i  = CLASS_NAMES.index('Pre-CHF')
    ratio = max(orig) / max(orig[pc_i], 1)
    axes[0].annotate(
        f'{ratio:.0f}× minority!',
        xy=(pc_i, orig[pc_i]), xytext=(pc_i + 0.8, 3000),
        arrowprops=dict(
            arrowstyle='->', color='red', lw=2),
        fontsize=10, color='red',
        fontweight='bold')

    # After split
    split_data = {
        'Train\n(70%)': tr,
        'Val\n(15%)'  : va,
        'Test\n(15%)' : te,
    }
    x  = np.arange(len(CLASS_NAMES))
    w  = 0.25
    sc = ['#2196F3', '#4CAF50', '#FF9800']
    for i, (split, vals) in enumerate(
            split_data.items()):
        axes[1].bar(
            x+(i-1)*w, vals, w,
            label=split, color=sc[i],
            edgecolor='black',
            linewidth=0.8, alpha=0.85)
    axes[1].set_title(
        '(b) After Split\n'
        '(Original Only, No Augment)',
        fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Number of Files')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')
    axes[1].set_ylim(10, 20000)

    # After augmentation
    aug_data = {
        'Train\n(augmented)': aug_tr,
        'Val\n(original)'   : va,
        'Test\n(original)'  : te,
    }
    ac = ['#673AB7', '#4CAF50', '#FF9800']
    for i, (split, vals) in enumerate(
            aug_data.items()):
        axes[2].bar(
            x+(i-1)*w, vals, w,
            label=split, color=ac[i],
            edgecolor='black',
            linewidth=0.8, alpha=0.85)
    axes[2].set_title(
        '(c) Final Dataset\n'
        'Train: Balanced | Val+Test: Original',
        fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Number of Files')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASS_NAMES)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_yscale('log')
    axes[2].set_ylim(10, 30000)
    axes[2].axhline(y=TARGET_N, color='purple',
                    linestyle='--',
                    linewidth=2, alpha=0.7)
    axes[2].text(
        3.4, TARGET_N * 1.08, f'{TARGET_N:,}\n(balanced)',
        fontsize=8, color='purple',
        fontweight='bold')

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(d, 'fig_dataset.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print("  Done: fig_dataset.png")


def _fig_pipeline():
    """Pipeline flowchart figure."""
    print("  Making fig_pipeline.png...")

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle(
        'Proposed Methodology: Acoustic Boiling '
        'Regime Classification Pipeline',
        fontsize=14, fontweight='bold', y=0.98)

    def draw_box(ax, x, y, w, h, text,
                 fc, ec='black', fs=8.5):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text,
                ha='center', va='center',
                fontsize=fs, fontweight='bold',
                multialignment='center')

    def arrow(ax, x1, y1, x2, y2, lbl=''):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->>',
                color='#333333', lw=2.5))
        if lbl:
            ax.text(
                (x1+x2)/2, (y1+y2)/2+0.15, lbl,
                ha='center', fontsize=8,
                color='#555555', style='italic')

    draw_box(ax, 0.3, 3.2, 2.5, 2.0,
             'DATASET\n(Sinha 2023)\n'
             '16,282 WAV\n4 Classes\nSR=10kHz',
             '#BBDEFB')
    draw_box(ax, 3.5, 3.2, 2.8, 2.0,
             'SPECTROGRAM\nGENERATION\n'
             'N_FFT=512|Hop=256\n'
             '5 Types:\nSTFT/MEL/MFCC\n'
             'CQT/CHROMA',
             '#C8E6C9')
    draw_box(ax, 7.1, 3.2, 2.8, 2.0,
             'SPLIT FIRST\n'
             '(Before Augment!)\n'
             'Train: 70%\n'
             'Val  : 15%\n'
             'Test : 15%',
             '#FFE0B2')
    draw_box(ax, 10.7, 4.5, 2.8, 1.5,
             'AUGMENT\nTRAIN ONLY\n'
             '13,000/class\n(6 techniques)',
             '#F8BBD9')
    draw_box(ax, 10.7, 2.0, 2.8, 1.5,
             'VAL + TEST\nNO AUGMENT\n'
             'Original images\nonly',
             '#E1BEE7')
    draw_box(ax, 14.3, 3.2, 2.8, 2.0,
             'CNN TRAINING\n'
             'Original CNN\n(0.52M params)\n'
             'ResNet50V2\n(24.75M params)',
             '#B2EBF2')
    draw_box(ax, 17.5, 3.2, 2.2, 2.0,
             'EVALUATION\n'
             '✓ Accuracy\n'
             '✓ Pre-CHF F1\n'
             '✓ Noise Test\n'
             '✓ Grad-CAM\n'
             '✓ Ablation',
             '#DCEDC8')

    arrow(ax, 2.8, 4.2, 3.5, 4.2)
    arrow(ax, 6.3, 4.2, 7.1, 4.2)
    arrow(ax, 9.9, 4.8, 10.7, 5.2, 'Train set')
    arrow(ax, 9.9, 3.6, 10.7, 2.8, 'Val+Test')
    arrow(ax, 13.5, 5.2, 14.3, 4.8)
    arrow(ax, 13.5, 2.8, 14.3, 3.6)
    arrow(ax, 17.1, 4.2, 17.5, 4.2)

    warn = mpatches.FancyBboxPatch(
        (6.8, 1.0), 3.4, 1.0,
        boxstyle="round,pad=0.1",
        facecolor='#FFFF00',
        edgecolor='red', linewidth=2.5)
    ax.add_patch(warn)
    ax.text(8.5, 1.5,
            '⚠ CRITICAL: Split BEFORE augment\n'
            '→ prevents data leakage!',
            ha='center', va='center',
            fontsize=9, color='red',
            fontweight='bold')

    for xi, lbl in [
        (1.55, 'Step 1'), (4.9, 'Step 2'),
        (8.5, 'Step 3'), (12.1, 'Step 4'),
        (15.7, 'Step 5'), (18.6, 'Step 6')]:
        ax.text(xi, 5.5, lbl,
                ha='center', fontsize=10,
                fontweight='bold',
                color='#1565C0', style='italic')

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(d, 'fig_pipeline.png'),
            dpi=150, bbox_inches='tight',
            facecolor='#FAFAFA')
    plt.close()
    print("  Done: fig_pipeline.png")


def _fig_spectrogram_examples():
    """Spectrogram examples grid figure."""
    print("  Making fig_spectrogram_examples.png...")

    fig, axes = plt.subplots(4, 5, figsize=(20, 14))
    fig.suptitle(
        'Spectrogram Representations for '
        'All Four Boiling Regimes\n'
        'SR=10kHz | N_FFT=512 | '
        'Hop=256 | Image=224×224px',
        fontsize=13, fontweight='bold')

    for row, cls in enumerate(CLASS_NAMES):
        for col, spec in enumerate(SPEC_TYPES):
            ax       = axes[row][col]
            spec_dir = os.path.join(
                SPEC_DIR, spec, cls)
            loaded   = False

            if os.path.exists(spec_dir):
                pngs = sorted([
                    f for f in os.listdir(spec_dir)
                    if f.endswith('.png')
                    and not f.startswith('aug')])
                if pngs:
                    try:
                        img = Image.open(
                            os.path.join(
                                spec_dir,
                                pngs[0])
                        ).convert('RGB')
                        ax.imshow(img)
                        loaded = True
                    except Exception:
                        pass

            if not loaded:
                ax.set_facecolor('#EEEEEE')
                ax.text(0.5, 0.5, 'Image\nPending',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=9, color='gray')
            ax.axis('off')

            if row == 0:
                ax.set_title(
                    SPEC_LABELS[spec],
                    fontsize=10,
                    fontweight='bold', pad=8)
            if col == 0:
                ax.text(
                    -0.15, 0.5, cls,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight='bold',
                    color=REGIME_COLORS[cls],
                    va='center', ha='right')
            if col == 4:
                ax.text(
                    1.05, 0.5,
                    PHYSICS_SHORT[cls],
                    transform=ax.transAxes,
                    fontsize=8, va='center',
                    color='darkgreen',
                    fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(
        left=0.08, right=0.92,
        hspace=0.15, wspace=0.05)
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(
                d, 'fig_spectrogram_examples.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print("  Done: fig_spectrogram_examples.png")


# ============================================================
# REVISION STEP A: ABLATION STUDY
# ============================================================
def step_ablation_study(
        model_name='Original_CNN',
        spec_type='STFT',
        epochs=40,
        n_seeds=3):
    """
    5-experiment ablation:
    A=None, B=Mild, C=Full,
    D=No_FreqMask, E=No_Noise
    Each repeated n_seeds times → mean ± std.
    """
    print("\n" + "="*60)
    print("REVISION A: ABLATION STUDY")
    print(f"  Model: {model_name} | Spec: {spec_type}")
    print(f"  Seeds: {n_seeds}")
    print("="*60)

    experiments = {
        'A_None'       : ('none',       TARGET_N),
        'B_Mild'       : ('mild',       TARGET_N),
        'C_Full'       : ('full',       TARGET_N),
        'D_No_FreqMask': ('no_freqmask',TARGET_N),
        'E_No_Noise'   : ('no_noise',   TARGET_N),
    }

    seeds        = [42, 123, 456][:n_seeds]
    all_ablation = {}

    for exp_name, (aug_mode, target_n) in \
            experiments.items():
        print(f"\n  Experiment {exp_name} "
              f"(mode={aug_mode})")

        split_dir = build_augmented_split(
            spec_type = spec_type,
            target_n  = target_n,
            aug_mode  = aug_mode,
            out_dir   = os.path.join(
                ABLATION_DIR,
                f"split_{exp_name}_{spec_type}"))

        exp_accs = []
        exp_f1s  = []

        for seed in seeds:
            set_all_seeds(seed)

            model = BUILDERS[model_name](
                input_shape=IMG_SIZE+(3,),
                num_classes=NUM_CLASSES)

            result = _train_model_on_split(
                model      = model,
                split_path = split_dir,
                spec_type  = spec_type,
                epochs     = epochs,
                verbose    = 0)

            if result is None:
                tf.keras.backend.clear_session()
                continue

            acc, y_true, y_pred, _, _ = result
            report = classification_report(
                y_true, y_pred,
                target_names=CLASS_NAMES,
                output_dict=True, zero_division=0)
            f1 = report.get(
                'Pre-CHF', {}).get('f1-score', 0)

            exp_accs.append(acc*100)
            exp_f1s.append(f1)
            print(f"    seed={seed}: "
                  f"acc={acc*100:.2f}%  "
                  f"Pre-CHF F1={f1:.4f}")
            del model
            tf.keras.backend.clear_session()

        if not exp_accs:
            print(f"  WARNING: no successful runs for {exp_name}")
            continue

        # Record what the 'none' baseline actually contains: the raw
        # imbalanced originals (no class reaches TARGET_N), so that the
        # JSON does not misleadingly claim 13000/class.
        actual_counts = {}
        for folder in FOLDERS:
            p = os.path.join(split_dir, spec_type, 'train', folder)
            actual_counts[folder] = (
                len([f for f in os.listdir(p) if f.endswith('.png')])
                if os.path.exists(p) else 0)

        all_ablation[exp_name] = {
            'aug_mode'      : aug_mode,
            'target_n'      : target_n,
            'actual_train_n': actual_counts,
            'mean_acc' : safe_mean(exp_accs),
            'std_acc'  : safe_std(exp_accs),
            'mean_f1'  : safe_mean(exp_f1s),
            'std_f1'   : safe_std(exp_f1s),
            'accs'     : exp_accs,
            'f1s'      : exp_f1s,
            'n_runs'   : len(exp_accs),
        }
        print(f"  → Acc: {safe_mean(exp_accs):.2f}%"
              f"±{safe_std(exp_accs):.2f}%  "
              f"Pre-CHF F1: {safe_mean(exp_f1s):.4f}"
              f"±{safe_std(exp_f1s):.4f}")

    if not all_ablation:
        print("  ERROR: ablation produced no results")
        return None

    # Print table
    print("\n" + "="*70)
    print(f"  {'Exp':<18} {'Mode':<14}"
          f"{'Acc (mean±std)':<22}"
          f"{'Pre-CHF F1 (mean±std)'}")
    print("  " + "-"*70)
    for k, d in all_ablation.items():
        print(f"  {k:<18} {d['aug_mode']:<14}"
              f"{d['mean_acc']:.2f}±"
              f"{d['std_acc']:.2f}%"
              f"{'':>8}"
              f"{d['mean_f1']:.4f}±"
              f"{d['std_f1']:.4f}")

    # Save
    afile = os.path.join(
        RESULTS_DIR, 'ablation_study.json')
    with open(afile, 'w') as f:
        json.dump(all_ablation, f, indent=2, cls=NumpyEncoder)

    _plot_ablation(all_ablation)
    print(f"\n  Saved: {afile}")
    return all_ablation


def _plot_ablation(all_ablation):
    exp_names = list(all_ablation.keys())
    mean_accs = [all_ablation[e]['mean_acc']
                 for e in exp_names]
    std_accs  = [all_ablation[e]['std_acc']
                 for e in exp_names]
    mean_f1s  = [all_ablation[e]['mean_f1']
                 for e in exp_names]
    std_f1s   = [all_ablation[e]['std_f1']
                 for e in exp_names]
    colors    = ['#d62728','#ff7f0e','#2ca02c',
                 '#1f77b4','#9467bd']
    x         = np.arange(len(exp_names))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6))
    fig.suptitle(
        'Ablation Study\n'
        'Effect of Augmentation Strategy '
        'on Classification Performance',
        fontsize=12, fontweight='bold')

    bars1 = ax1.bar(x, mean_accs, color=colors,
                    alpha=0.85, edgecolor='black',
                    linewidth=0.8)
    ax1.errorbar(x, mean_accs, yerr=std_accs,
                 fmt='none', color='black',
                 capsize=5, linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [e.replace('_', '\n') for e in exp_names],
        fontsize=9)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_title('Overall Test Accuracy',
                  fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, m, s in zip(bars1, mean_accs, std_accs):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + s + 1,
            f'{m:.1f}%',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold')

    bars2 = ax2.bar(x, mean_f1s, color=colors,
                    alpha=0.85, edgecolor='black',
                    linewidth=0.8)
    ax2.errorbar(x, mean_f1s, yerr=std_f1s,
                 fmt='none', color='black',
                 capsize=5, linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [e.replace('_', '\n') for e in exp_names],
        fontsize=9)
    ax2.set_ylabel('Pre-CHF F1-Score', fontsize=11)
    ax2.set_title('Pre-CHF F1-Score', fontsize=11)
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, m, s in zip(bars2, mean_f1s, std_f1s):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + s + 0.01,
            f'{m:.3f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold')

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(d, 'ablation_study.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ablation_study.png")


# ============================================================
# REVISION STEP B: AUGMENTATION SENSITIVITY
# ============================================================
def step_augmentation_sensitivity(
        model_name='Original_CNN',
        spec_type='STFT',
        epochs=40,
        n_seeds=3):
    """
    Train with 5 expansion factors applied to EVERY class:
    1× 5× 10× 25× 46× (46× ≈ Pre-CHF 279 → 13000)

    Each class is augmented to  min(n_orig_class * factor, TARGET_N)
    but never BELOW its own number of originals.  Previously a single
    global target_n was used, which truncated NB/BKG/TB down to 279
    images at 1× and confounded 'augmentation factor' with 'total
    dataset size'.  Shows saturation → anti-artifact evidence.
    """
    print("\n" + "="*60)
    print("REVISION B: AUGMENTATION SENSITIVITY")
    print(f"  Model: {model_name} | Spec: {spec_type}")
    print("="*60)

    # Per-class original train counts
    n_orig = {}
    for folder in FOLDERS:
        p = os.path.join(SPLIT_ORIG, spec_type, 'train', folder)
        n_orig[folder] = (
            len([f for f in os.listdir(p) if f.endswith('.png')])
            if os.path.exists(p) else 0)
    if not any(n_orig.values()):
        print(f"  ERROR: no originals under {SPLIT_ORIG}/{spec_type}")
        return None
    print(f"  Original train counts: {n_orig}")

    factors     = [1, 5, 10, 25, 46]
    seeds       = [42, 123, 456][:n_seeds]
    sensitivity = {}

    for factor in factors:
        lbl = f"{factor}x"
        per_class = {
            c: (TARGET_N if factor == factors[-1]
                else max(n, min(int(n * factor), TARGET_N)))
            for c, n in n_orig.items()}
        print(f"\n  Factor {lbl} → per-class targets={per_class}")

        split_dir = build_augmented_split(
            spec_type        = spec_type,
            target_n         = TARGET_N,
            aug_mode         = 'full',
            per_class_target = per_class,
            out_dir          = os.path.join(
                ABLATION_DIR,
                f"sens_{lbl}_{spec_type}"))

        run_accs = []
        run_f1s  = []

        for seed in seeds:
            set_all_seeds(seed)

            model  = BUILDERS[model_name](
                input_shape=IMG_SIZE+(3,),
                num_classes=NUM_CLASSES)
            result = _train_model_on_split(
                model      = model,
                split_path = split_dir,
                spec_type  = spec_type,
                epochs     = epochs,
                verbose    = 0)

            if result is None:
                tf.keras.backend.clear_session()
                continue

            acc, y_true, y_pred, _, _ = result
            report = classification_report(
                y_true, y_pred,
                target_names=CLASS_NAMES,
                output_dict=True, zero_division=0)
            f1 = report.get(
                'Pre-CHF', {}).get('f1-score', 0)

            run_accs.append(acc*100)
            run_f1s.append(f1)
            print(f"    seed={seed}: "
                  f"acc={acc*100:.2f}%  "
                  f"Pre-CHF F1={f1:.4f}")
            del model
            tf.keras.backend.clear_session()

        if not run_accs:
            print(f"  WARNING: no successful runs for {lbl}")
            continue

        sensitivity[lbl] = {
            'factor'          : factor,
            'per_class_target': per_class,
            'prechf_target_n' : per_class.get('Pre-CHF'),
            'mean_acc' : safe_mean(run_accs),
            'std_acc'  : safe_std(run_accs),
            'mean_f1'  : safe_mean(run_f1s),
            'std_f1'   : safe_std(run_f1s),
            'accs'     : run_accs,
            'f1s'      : run_f1s,
            'n_runs'   : len(run_accs),
        }
        print(f"  → Acc: {safe_mean(run_accs):.2f}%"
              f"±{safe_std(run_accs):.2f}%  "
              f"Pre-CHF F1: {safe_mean(run_f1s):.4f}"
              f"±{safe_std(run_f1s):.4f}")

    if not sensitivity:
        print("  ERROR: sensitivity produced no results")
        return None

    sf = os.path.join(
        RESULTS_DIR, 'augmentation_sensitivity.json')
    with open(sf, 'w') as f:
        json.dump(sensitivity, f, indent=2, cls=NumpyEncoder)
    _plot_sensitivity(sensitivity)
    print(f"\n  Saved: {sf}")
    return sensitivity


def _plot_sensitivity(sensitivity):
    labels    = list(sensitivity.keys())
    factors   = [sensitivity[l]['factor']
                 for l in labels]
    mean_accs = [sensitivity[l]['mean_acc']
                 for l in labels]
    std_accs  = [sensitivity[l]['std_acc']
                 for l in labels]
    mean_f1s  = [sensitivity[l]['mean_f1']
                 for l in labels]
    std_f1s   = [sensitivity[l]['std_f1']
                 for l in labels]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6))
    fig.suptitle(
        'Augmentation Sensitivity Analysis\n'
        'Performance vs Expansion Factor\n'
        '(Saturation confirms no artifact '
        'memorization)',
        fontsize=11, fontweight='bold')

    for ax, means, stds, ylabel, color, lbl in [
        (ax1, mean_accs, std_accs,
         'Test Accuracy (%)', '#1f77b4',
         'Mean accuracy'),
        (ax2, mean_f1s, std_f1s,
         'Pre-CHF F1-Score', '#2ca02c',
         'Mean Pre-CHF F1')
    ]:
        ax.plot(factors, means,
                '-o', color=color,
                linewidth=2.5, markersize=10,
                markerfacecolor='white',
                markeredgewidth=2.5,
                label=lbl)
        ax.fill_between(
            factors,
            [m-s for m, s in zip(means, stds)],
            [m+s for m, s in zip(means, stds)],
            alpha=0.2, color=color,
            label='±1 Std')
        for f, m, s in zip(factors, means, stds):
            ax.annotate(
                f'{m:.2f}±{s:.2f}',
                (f, m+s+0.5),
                ha='center', fontsize=8,
                fontweight='bold')
        ax.set_xlabel(
            'Augmentation Factor (×)',
            fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(factors)
        ax.set_xticklabels(
            [f'{f}×' for f in factors])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax1.set_title('Overall Accuracy vs Factor',
                  fontsize=11)
    ax1.set_ylim(min(mean_accs)-5, 105)
    ax2.set_title('Pre-CHF F1 vs Factor',
                  fontsize=11)
    ax2.set_ylim(0, 1.15)

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(
                d, 'augmentation_sensitivity.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: augmentation_sensitivity.png")


# ============================================================
# REVISION STEP C: STATISTICAL SIGNIFICANCE
# ============================================================
def step_statistical_significance(
        results_json=None, n_seeds=5):
    """
    Paired t-test + Wilcoxon for:
    1. STFT vs MEL (best model)
    2. Original_CNN vs ResNet50V2_FT (STFT)
    Reports p-values for publication.
    """
    print("\n" + "="*60)
    print("REVISION C: STATISTICAL SIGNIFICANCE")
    print(f"  Seeds: {n_seeds}")
    print("="*60)

    df = load_results_df(results_json)
    if df is None:
        return None
    best_mn = df.loc[
        df['test_accuracy'].idxmax(),
        'model_name']
    print(f"  Best model: {best_mn}")

    # NOTE: a two-sided Wilcoxon signed-rank test cannot reach p<0.05
    # with fewer than 6 paired samples (min p at n=5 is 0.0625).
    if n_seeds < 6:
        print(f"  WARNING: n_seeds={n_seeds} < 6 — Wilcoxon can never "
              f"be significant at alpha=0.05; rely on the paired "
              f"t-test or increase --n_seeds to >= 6.")

    seeds = [42, 123, 456, 789, 1024,
             2024, 999, 314, 271, 100][:n_seeds]

    def collect(mn, st):
        accs = []
        for seed in seeds:
            set_all_seeds(seed)
            model = BUILDERS[mn](
                input_shape=IMG_SIZE+(3,),
                num_classes=NUM_CLASSES)
            result = _train_model_on_split(
                model      = model,
                split_path = SPLIT_FINAL,
                spec_type  = st,
                epochs     = 30,
                verbose    = 0)
            if result:
                accs.append(result[0]*100)
            else:
                # keep pairing aligned: record NaN and drop later
                accs.append(np.nan)
            del model
            tf.keras.backend.clear_session()
        return np.array(accs, dtype=float)

    def paired_clean(a, b):
        """Drop pairs where either run failed (NaN)."""
        m = ~(np.isnan(a) | np.isnan(b))
        return a[m], b[m]

    sig_results = {}

    # Comparison 1: STFT vs MEL
    print(f"\n  Comparison 1: STFT vs MEL "
          f"({best_mn})")
    stft_a, mel_a = paired_clean(
        collect(best_mn, 'STFT'),
        collect(best_mn, 'MEL'))

    if len(stft_a) >= 3 and len(mel_a) >= 3:
        t_s, p_t   = scipy_stats.ttest_rel(
            stft_a, mel_a)
        try:
            w_s, p_w = scipy_stats.wilcoxon(
                stft_a, mel_a,
                alternative='two-sided')
        except ValueError:
            w_s, p_w = 0.0, 1.0

        sig = (p_t < 0.05 or p_w < 0.05)
        print(f"    STFT: {stft_a.mean():.2f}%"
              f"±{stft_a.std():.2f}%")
        print(f"    MEL : {mel_a.mean():.2f}%"
              f"±{mel_a.std():.2f}%")
        print(f"    t-test : t={t_s:.3f}, "
              f"p={p_t:.4f} "
              f"{'*' if p_t<0.05 else 'ns'}")
        print(f"    Wilcoxon: W={w_s:.3f}, "
              f"p={p_w:.4f} "
              f"{'*' if p_w<0.05 else 'ns'}")
        print(f"    Significant: {sig}")

        sig_results['STFT_vs_MEL'] = {
            'model'      : best_mn,
            'stft_mean'  : float(stft_a.mean()),
            'stft_std'   : float(stft_a.std()),
            'mel_mean'   : float(mel_a.mean()),
            'mel_std'    : float(mel_a.std()),
            'p_ttest'    : float(p_t),
            'p_wilcoxon' : float(p_w),
            'significant': bool(sig),
            'n_pairs'    : int(len(stft_a)),
        }

    # Comparison 2: CNN vs ResNet
    print(f"\n  Comparison 2: CNN vs ResNet (STFT)")
    # Reuse STFT runs of the best model if it is the CNN (saves ~n_seeds
    # trainings); otherwise train fresh.
    if best_mn == 'Original_CNN' and len(stft_a) == n_seeds:
        cnn_raw = stft_a
    else:
        cnn_raw = collect('Original_CNN', 'STFT')
    cnn_a, resnet_a = paired_clean(
        cnn_raw, collect('ResNet50V2_FT', 'STFT'))

    if (len(cnn_a) >= 3 and
            len(resnet_a) >= 3):
        n   = min(len(cnn_a), len(resnet_a))
        t_s2, p_t2 = scipy_stats.ttest_rel(
            cnn_a[:n], resnet_a[:n])
        try:
            w_s2, p_w2 = scipy_stats.wilcoxon(
                cnn_a[:n], resnet_a[:n],
                alternative='two-sided')
        except ValueError:
            w_s2, p_w2 = 0.0, 1.0

        sig2 = (p_t2 < 0.05 or p_w2 < 0.05)
        print(f"    CNN   : {cnn_a.mean():.2f}%"
              f"±{cnn_a.std():.2f}%")
        print(f"    ResNet: {resnet_a.mean():.2f}%"
              f"±{resnet_a.std():.2f}%")
        print(f"    t-test : t={t_s2:.3f}, "
              f"p={p_t2:.4f} "
              f"{'*' if p_t2<0.05 else 'ns'}")
        print(f"    Wilcoxon: W={w_s2:.3f}, "
              f"p={p_w2:.4f} "
              f"{'*' if p_w2<0.05 else 'ns'}")
        print(f"    Significant: {sig2}")

        sig_results['CNN_vs_ResNet'] = {
            'cnn_mean'   : float(cnn_a.mean()),
            'cnn_std'    : float(cnn_a.std()),
            'resnet_mean': float(resnet_a.mean()),
            'resnet_std' : float(resnet_a.std()),
            'p_ttest'    : float(p_t2),
            'p_wilcoxon' : float(p_w2),
            'significant': bool(sig2),
            'n_pairs'    : int(n),
        }

    sf = os.path.join(
        RESULTS_DIR,
        'statistical_significance.json')
    with open(sf, 'w') as f:
        json.dump(sig_results, f, indent=2, cls=NumpyEncoder)
    _plot_significance(sig_results)
    print(f"\n  Saved: {sf}")
    return sig_results


def _plot_significance(sig_results):
    n = len(sig_results)
    if n == 0:
        return

    fig, axes = plt.subplots(
        1, n, figsize=(7*n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        'Statistical Significance Testing\n'
        'Paired t-test & Wilcoxon signed-rank',
        fontsize=12, fontweight='bold')

    for ax, (comp, data) in zip(
            axes, sig_results.items()):
        if comp == 'STFT_vs_MEL':
            lbs  = ['STFT', 'MEL']
            mns  = [data['stft_mean'],
                    data['mel_mean']]
            sts  = [data['stft_std'],
                    data['mel_std']]
            cols = ['#1f77b4', '#ff7f0e']
        else:
            lbs  = ['CNN', 'ResNet50V2']
            mns  = [data['cnn_mean'],
                    data['resnet_mean']]
            sts  = [data['cnn_std'],
                    data['resnet_std']]
            cols = ['#2ca02c', '#d62728']

        p   = min(data['p_ttest'],
                  data['p_wilcoxon'])
        x   = np.arange(len(lbs))
        bars = ax.bar(x, mns, color=cols,
                      alpha=0.85,
                      edgecolor='black',
                      linewidth=0.8,
                      width=0.5)
        ax.errorbar(x, mns, yerr=sts,
                    fmt='none', color='black',
                    capsize=8, linewidth=2)
        for bar, m, s in zip(bars, mns, sts):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + 0.3,
                f'{m:.2f}±{s:.2f}%',
                ha='center', fontsize=10,
                fontweight='bold')

        y_top = max(m+s for m, s in zip(mns, sts))
        y_br  = y_top + 1.5
        ax.plot([0, 0, 1, 1],
                [y_br-0.5, y_br, y_br, y_br-0.5],
                lw=1.5, color='black')
        ax.text(0.5, y_br+0.3,
                (f'p={p:.4f} *' if p < 0.05
                 else f'p={p:.4f} ns'),
                ha='center', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(lbs, fontsize=11)
        ax.set_ylabel('Test Accuracy (%)',
                      fontsize=11)
        ax.set_title(
            comp.replace('_', ' '),
            fontsize=11)
        ax.set_ylim(min(mns)-10, y_br+3)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(
                d, 'statistical_significance.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: statistical_significance.png")


# ============================================================
# REVISION STEP D: CALIBRATION ANALYSIS
# ============================================================
def step_calibration_analysis(results_json=None):
    """
    Reliability diagram + ECE + confidence hist.
    Safety-critical: Pre-CHF prediction confidence.
    """
    print("\n" + "="*60)
    print("REVISION D: CALIBRATION ANALYSIS")
    print("="*60)

    ctx = load_best_model_and_test(results_json)
    if ctx is None:
        return None
    best         = ctx['best']
    spec_type    = ctx['spec_type']
    y_pred_probs = ctx['y_probs']
    y_true       = ctx['y_true']
    y_pred       = ctx['y_pred']
    y_conf       = np.max(y_pred_probs, axis=1)

    n_bins    = 15
    bin_edges = np.linspace(0, 1, n_bins+1)
    bin_acc   = []
    bin_conf  = []
    bin_cnts  = []
    ece       = 0.0
    n_total   = len(y_true)

    for i in range(n_bins):
        lo   = bin_edges[i]
        hi   = bin_edges[i+1]
        mask = (y_conf > lo) & (y_conf <= hi)
        n    = mask.sum()
        if n > 0:
            acc  = (y_pred[mask] == y_true[mask]
                    ).mean()
            conf = y_conf[mask].mean()
            bin_acc.append(acc)
            bin_conf.append(conf)
            bin_cnts.append(n)
            ece += (n/n_total)*abs(acc-conf)
        else:
            bin_acc.append(0)
            bin_conf.append((lo+hi)/2)
            bin_cnts.append(0)

    # Per-class ECE
    y_bin        = label_binarize(
        y_true, classes=list(range(NUM_CLASSES)))
    per_cls_ece  = {}
    print(f"\n  Model   : {best['model_name']}")
    print(f"  Spec    : {spec_type}")
    print(f"  ECE     : {ece:.4f}")
    print(f"\n  Per-class ECE:")
    for ci, cls in enumerate(CLASS_NAMES):
        cls_probs = y_pred_probs[:, ci]
        cls_true  = y_bin[:, ci]
        cls_ece   = 0.0
        for i in range(n_bins):
            lo   = bin_edges[i]
            hi   = bin_edges[i+1]
            mask = (cls_probs>lo) & (cls_probs<=hi)
            n    = mask.sum()
            if n > 0:
                a = cls_true[mask].mean()
                c = cls_probs[mask].mean()
                cls_ece += (n/n_total)*abs(a-c)
        per_cls_ece[cls] = round(cls_ece, 4)
        st = ('Good' if cls_ece < 0.05
              else 'Moderate' if cls_ece < 0.10
              else 'Poor')
        print(f"    {cls:<10}: {cls_ece:.4f} [{st}]")

    # Plot
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(
        f'Calibration Analysis\n'
        f'{best["model_name"]} + {spec_type} | '
        f'ECE={ece:.4f}',
        fontsize=12, fontweight='bold')

    # Reliability diagram
    ax1 = fig.add_subplot(131)
    ax1.plot([0,1],[0,1],'k--', linewidth=1.5,
             label='Perfect calibration')
    bin_centers = [(bin_edges[i]+bin_edges[i+1])/2
                   for i in range(n_bins)]
    valid = [i for i, c in enumerate(bin_cnts)
             if c > 0]
    ax1.bar(
        [bin_centers[i] for i in valid],
        [bin_acc[i] for i in valid],
        width=1.0/n_bins,
        alpha=0.7, color='#1f77b4',
        edgecolor='black', linewidth=0.5,
        label=f'Model (ECE={ece:.3f})')
    for i in valid:
        gap = bin_conf[i] - bin_acc[i]
        if gap > 0:
            ax1.bar(
                bin_centers[i], gap,
                bottom=bin_acc[i],
                width=1.0/n_bins,
                alpha=0.3, color='red')
    ax1.set_xlabel('Confidence', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Reliability Diagram',
                  fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Confidence histogram
    ax2 = fig.add_subplot(132)
    ax2.hist(y_conf, bins=25, color='#2ca02c',
             alpha=0.8, edgecolor='black',
             linewidth=0.5)
    ax2.axvline(x=y_conf.mean(), color='red',
                linestyle='--', linewidth=2,
                label=f'Mean: {y_conf.mean():.3f}')
    ax2.set_xlabel('Max Softmax Confidence',
                   fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Confidence Distribution',
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Per-class ECE bar
    ax3  = fig.add_subplot(133)
    cls_n = list(per_cls_ece.keys())
    cls_e = list(per_cls_ece.values())
    cols  = [REGIME_COLORS[c] for c in cls_n]
    bars  = ax3.bar(cls_n, cls_e, color=cols,
                    alpha=0.85,
                    edgecolor='black',
                    linewidth=0.8)
    ax3.axhline(y=0.05, color='green',
                linestyle='--', linewidth=1.5,
                label='Good (0.05)')
    ax3.axhline(y=0.10, color='orange',
                linestyle='--', linewidth=1.5,
                label='Moderate (0.10)')
    for bar, v in zip(bars, cls_e):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002,
                 f'{v:.4f}',
                 ha='center', fontsize=9,
                 fontweight='bold')
    ax3.set_ylabel('ECE', fontsize=11)
    ax3.set_title('Per-Class ECE', fontsize=11)
    ax3.set_ylim(0, max(cls_e)*1.5 + 0.02)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(
                d, 'calibration_analysis.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: calibration_analysis.png")

    out = {
        'model'      : best['model_name'],
        'spec_type'  : spec_type,
        'overall_ece': round(ece, 4),
        'mean_conf'  : round(float(y_conf.mean()),
                             4),
        'per_class'  : per_cls_ece
    }
    cf = os.path.join(RESULTS_DIR,
                      'calibration.json')
    with open(cf, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {cf}")
    return out


# ============================================================
# REVISION STEP E: ROC CURVES
# ============================================================
def step_roc_curves(results_json=None):
    """
    ROC curves: all classes OVR +
    binary Pre-CHF vs NB.
    """
    print("\n" + "="*60)
    print("REVISION E: ROC CURVES")
    print("  Focus: Pre-CHF vs NB discrimination")
    print("="*60)

    ctx = load_best_model_and_test(results_json)
    if ctx is None:
        return None
    best         = ctx['best']
    spec_type    = ctx['spec_type']
    y_pred_probs = ctx['y_probs']
    y_true       = ctx['y_true']
    y_bin        = label_binarize(
        y_true, classes=list(range(NUM_CLASSES)))

    roc_data = {}
    print(f"\n  AUC scores:")
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(
            y_bin[:, i], y_pred_probs[:, i])
        ra = auc(fpr, tpr)
        roc_data[cls] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': round(float(ra), 4)
        }
        print(f"    {cls:<10}: AUC={ra:.4f}")

    # Binary Pre-CHF vs NB
    nb_idx     = CLASS_NAMES.index('NB')
    pc_idx     = CLASS_NAMES.index('Pre-CHF')
    mask       = ((y_true==nb_idx) |
                  (y_true==pc_idx))
    y_bin_nb   = (y_true[mask]==pc_idx).astype(int)
    p_bin      = y_pred_probs[mask, pc_idx]
    fpr_b, tpr_b, _ = roc_curve(y_bin_nb, p_bin)
    auc_b            = auc(fpr_b, tpr_b)
    print(f"\n  Pre-CHF vs NB binary AUC: {auc_b:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'ROC Curve Analysis\n'
        f'{best["model_name"]} + {spec_type}',
        fontsize=12, fontweight='bold')

    ax1 = axes[0]
    ax1.plot([0,1],[0,1],'k--',
             linewidth=1, label='Random')
    lws = ['-', '--', '-.', ':']
    for i, cls in enumerate(CLASS_NAMES):
        d = roc_data[cls]
        ax1.plot(d['fpr'], d['tpr'],
                 lw=2.5, linestyle=lws[i%4],
                 color=REGIME_COLORS[cls],
                 label=f'{cls} '
                       f'(AUC={d["auc"]:.4f})')
    ax1.set_xlabel('False Positive Rate',
                   fontsize=11)
    ax1.set_ylabel('True Positive Rate',
                   fontsize=11)
    ax1.set_title('ROC (One-vs-Rest)', fontsize=11)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])

    ax2 = axes[1]
    ax2.plot([0,1],[0,1],'k--',
             linewidth=1, label='Random')
    ax2.plot(fpr_b, tpr_b,
             color='#9467bd', linewidth=3,
             label=f'Pre-CHF vs NB '
                   f'(AUC={auc_b:.4f})')
    ax2.fill_between(fpr_b, tpr_b,
                     alpha=0.15, color='#9467bd')
    ax2.set_xlabel('False Positive Rate',
                   fontsize=11)
    ax2.set_ylabel('True Positive Rate',
                   fontsize=11)
    ax2.set_title(
        'Binary ROC: Pre-CHF vs NB\n'
        '(Most Acoustically Similar)',
        fontsize=11)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(d, 'roc_curves.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: roc_curves.png")

    out = {
        'model'          : best['model_name'],
        'spec_type'      : spec_type,
        'per_class_auc'  : {
            cls: roc_data[cls]['auc']
            for cls in CLASS_NAMES},
        'prechf_vs_nb_auc': round(float(auc_b), 4)
    }
    rf = os.path.join(RESULTS_DIR, 'roc.json')
    with open(rf, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {rf}")
    return out


# ============================================================
# REVISION STEP F: CONFIDENCE INTERVALS
# ============================================================
def step_confidence_intervals(
        results_json=None, n_bootstrap=1000):
    """
    Bootstrap 95% CIs for precision/recall/F1
    per class.
    """
    print("\n" + "="*60)
    print("REVISION F: CONFIDENCE INTERVALS")
    print(f"  Bootstrap: {n_bootstrap} samples")
    print("="*60)

    ctx = load_best_model_and_test(results_json)
    if ctx is None:
        return None
    best      = ctx['best']
    spec_type = ctx['spec_type']
    y_pred    = ctx['y_pred']
    y_true    = ctx['y_true']
    n_test    = len(y_true)

    rng = np.random.RandomState(42)
    boot = {
        cls: {'p': [], 'r': [], 'f1': []}
        for cls in CLASS_NAMES}

    print(f"  Running bootstrap...")
    all_labels = list(range(NUM_CLASSES))
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_test, size=n_test)
        # Pass labels= explicitly: a resample may miss a rare class
        # entirely, and classification_report(target_names=...)
        # would then raise a length-mismatch ValueError.
        p_ = precision_score(y_true[idx], y_pred[idx],
                             labels=all_labels, average=None,
                             zero_division=0)
        r_ = recall_score(y_true[idx], y_pred[idx],
                          labels=all_labels, average=None,
                          zero_division=0)
        f_ = f1_score(y_true[idx], y_pred[idx],
                      labels=all_labels, average=None,
                      zero_division=0)
        for ci, cls in enumerate(CLASS_NAMES):
            boot[cls]['p'].append(float(p_[ci]))
            boot[cls]['r'].append(float(r_[ci]))
            boot[cls]['f1'].append(float(f_[ci]))

    base_rep = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)

    print(f"\n  95% Bootstrap CIs:")
    print(f"  {'Class':<10} "
          f"{'Precision':<28} "
          f"{'Recall':<28} "
          f"{'F1-Score'}")
    print("  " + "-"*95)

    ci_out = {}
    for cls in CLASS_NAMES:
        b  = boot[cls]
        r  = base_rep.get(cls, {})
        pp = r.get('precision', 0)
        pr = r.get('recall', 0)
        pf = r.get('f1-score', 0)

        ci_p  = np.percentile(b['p'],  [2.5, 97.5])
        ci_r  = np.percentile(b['r'],  [2.5, 97.5])
        ci_f1 = np.percentile(b['f1'], [2.5, 97.5])

        ci_out[cls] = {
            'precision': {
                'point': round(pp, 4),
                'ci_lo': round(float(ci_p[0]), 4),
                'ci_hi': round(float(ci_p[1]), 4)},
            'recall': {
                'point': round(pr, 4),
                'ci_lo': round(float(ci_r[0]), 4),
                'ci_hi': round(float(ci_r[1]), 4)},
            'f1': {
                'point': round(pf, 4),
                'ci_lo': round(float(ci_f1[0]), 4),
                'ci_hi': round(float(ci_f1[1]), 4)},
        }
        print(
            f"  {cls:<10} "
            f"{pp:.4f} [{ci_p[0]:.4f},"
            f"{ci_p[1]:.4f}]{'':>4}"
            f"{pr:.4f} [{ci_r[0]:.4f},"
            f"{ci_r[1]:.4f}]{'':>4}"
            f"{pf:.4f} [{ci_f1[0]:.4f},"
            f"{ci_f1[1]:.4f}]")

    _plot_ci(ci_out)

    cf = os.path.join(
        RESULTS_DIR, 'confidence_intervals.json')
    with open(cf, 'w') as f:
        json.dump({
            'model'      : best['model_name'],
            'spec_type'  : spec_type,
            'n_bootstrap': n_bootstrap,
            'results'    : ci_out}, f, indent=2)
    print(f"\n  Saved: {cf}")
    return ci_out


def _plot_ci(ci_results):
    classes = list(ci_results.keys())
    metrics = ['precision', 'recall', 'f1']
    labels  = ['Precision', 'Recall', 'F1-Score']
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        'Per-Class Metrics with 95% Bootstrap CIs',
        fontsize=12, fontweight='bold')

    for i, (metric, label, color) in enumerate(
            zip(metrics, labels, colors)):
        pts   = [ci_results[c][metric]['point']
                 for c in classes]
        lo_e  = [ci_results[c][metric]['point']
                 - ci_results[c][metric]['ci_lo']
                 for c in classes]
        hi_e  = [ci_results[c][metric]['ci_hi']
                 - ci_results[c][metric]['point']
                 for c in classes]
        ax.bar(x+(i-1)*w, pts, w,
               label=label, color=color,
               alpha=0.85,
               edgecolor='black', linewidth=0.5)
        ax.errorbar(x+(i-1)*w, pts,
                    yerr=[lo_e, hi_e],
                    fmt='none', color='black',
                    capsize=4, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(
        'Metrics ± 95% CI (Bootstrap n=1000)',
        fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(
                d, 'confidence_intervals.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: confidence_intervals.png")


# ============================================================
# REVISION STEP G: HARD-NEGATIVE ANALYSIS
# ============================================================
def step_hard_negative_analysis(results_json=None):
    """
    Deep dive: NB ↔ Pre-CHF confusion.
    Most acoustically similar pair.
    """
    print("\n" + "="*60)
    print("REVISION G: HARD-NEGATIVE ANALYSIS")
    print("  Focus: NB vs Pre-CHF confusion")
    print("="*60)

    ctx = load_best_model_and_test(results_json)
    if ctx is None:
        return None
    best      = ctx['best']
    spec_type = ctx['spec_type']
    ypp       = ctx['y_probs']
    y_pred    = ctx['y_pred']
    y_true    = ctx['y_true']
    y_conf    = np.max(ypp, axis=1)

    nb_idx = CLASS_NAMES.index('NB')
    pc_idx = CLASS_NAMES.index('Pre-CHF')

    nb_true    = y_true == nb_idx
    pc_true    = y_true == pc_idx
    nb_as_pc   = nb_true & (y_pred == pc_idx)
    pc_as_nb   = pc_true & (y_pred == nb_idx)
    nb_correct = nb_true & (y_pred == nb_idx)
    pc_correct = pc_true & (y_pred == pc_idx)

    nb_tot = int(nb_true.sum())
    pc_tot = int(pc_true.sum())
    if nb_tot + pc_tot == 0:
        print("  ERROR: no NB / Pre-CHF samples in test set")
        return None
    cr     = ((nb_as_pc.sum() + pc_as_nb.sum()) /
              (nb_tot + pc_tot) * 100)

    print(f"\n  NB → Pre-CHF: "
          f"{nb_as_pc.sum()}/{nb_tot} "
          f"({nb_as_pc.sum()/max(nb_tot,1)*100:.1f}%)")
    print(f"  Pre-CHF → NB: "
          f"{pc_as_nb.sum()}/{pc_tot} "
          f"({pc_as_nb.sum()/max(pc_tot,1)*100:.1f}%)")
    print(f"  Overall confusion rate: {cr:.2f}%")

    if   cr < 5:
        interp = ("Excellent: model clearly "
                  "separates NB and Pre-CHF "
                  "frequency bands")
    elif cr < 15:
        interp = ("Good: some spectral overlap, "
                  "consistent with physics")
    else:
        interp = ("High confusion: significant "
                  "spectral overlap between regimes")
    print(f"  Physics: {interp}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Hard-Negative Analysis: NB vs Pre-CHF\n'
        f'{best["model_name"]} + {spec_type}',
        fontsize=12, fontweight='bold')

    # 3-column sub-confusion: rows = true NB / Pre-CHF,
    # cols = predicted NB / Pre-CHF / Other (BKG or TB).
    # (Previously any non-NB prediction was counted as Pre-CHF,
    #  inflating the apparent NB→Pre-CHF confusion.)
    sub_mask = (nb_true | pc_true)
    st       = np.where(y_true[sub_mask] == nb_idx, 0, 1)
    sp_raw   = y_pred[sub_mask]
    sp       = np.where(sp_raw == nb_idx, 0,
               np.where(sp_raw == pc_idx, 1, 2))
    scm      = confusion_matrix(st, sp, labels=[0, 1, 2])[:2, :]
    row_sum  = scm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    scm_n    = scm.astype(float) / row_sum

    sns.heatmap(
        scm_n, annot=True, fmt='.2%',
        cmap='Reds', ax=axes[0],
        xticklabels=['NB', 'Pre-CHF', 'Other'],
        yticklabels=['NB', 'Pre-CHF'],
        linewidths=1, linecolor='black',
        vmin=0, vmax=1)
    axes[0].set_title(
        'NB vs Pre-CHF Confusion '
        '(Normalized)', fontsize=11)
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('True', fontsize=11)

    # Confidence distributions
    groups = [
        (y_conf[nb_correct],
         f'NB→NB (n={nb_correct.sum()})',
         '#1f77b4'),
        (y_conf[nb_as_pc],
         f'NB→Pre-CHF (n={nb_as_pc.sum()})',
         '#d62728'),
        (y_conf[pc_correct],
         f'Pre-CHF→Pre-CHF '
         f'(n={pc_correct.sum()})',
         '#2ca02c'),
        (y_conf[pc_as_nb],
         f'Pre-CHF→NB (n={pc_as_nb.sum()})',
         '#ff7f0e'),
    ]
    for data, lbl, col in groups:
        if len(data) > 0:
            axes[1].hist(
                data, bins=15, alpha=0.7,
                label=lbl, color=col,
                edgecolor='black',
                linewidth=0.5)
    axes[1].set_xlabel('Model Confidence',
                       fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title(
        'Confidence: Correct vs Confused',
        fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(
                d, 'hard_negative_analysis.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: hard_negative_analysis.png")

    out = {
        'model'             : best['model_name'],
        'spec_type'         : spec_type,
        'nb_total'          : int(nb_tot),
        'prechf_total'      : int(pc_tot),
        'nb_as_prechf'      : int(nb_as_pc.sum()),
        'prechf_as_nb'      : int(pc_as_nb.sum()),
        'confusion_rate_pct': round(cr, 2),
        'interpretation'    : interp,
    }
    hf = os.path.join(
        RESULTS_DIR, 'hard_negative.json')
    with open(hf, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {hf}")
    return out


# ============================================================
# REVISION STEP H: GRAD-CAM
# ============================================================
def _layer_out_rank(layer):
    """Rank of a layer's output tensor, robust to Keras 2/3."""
    try:
        shp = layer.output.shape          # Keras 3 & 2
    except Exception:
        try:
            shp = layer.output_shape      # legacy Keras 2
        except Exception:
            return 0
    if isinstance(shp, (list, tuple)) and shp and isinstance(
            shp[0], (list, tuple)):
        shp = shp[0]
    return len(shp) if shp is not None else 0


def _find_last_4d_layer(model):
    """
    Locate the last layer with a 4-D (B,H,W,C) output.
    Handles (a) flat Sequential CNNs and (b) models whose first
    functional block is a nested pretrained base (ResNet/EffNet).
    Returns (sub_model_or_None, layer_name).
    """
    # (b) nested base model inside a wrapper
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer is not model:
            for inner in reversed(layer.layers):
                if _layer_out_rank(inner) == 4:
                    return layer, inner.name
    # (a) flat model
    for layer in reversed(model.layers):
        if _layer_out_rank(layer) == 4 and not isinstance(
                layer, tf.keras.layers.InputLayer):
            return None, layer.name
    return None, None


def compute_gradcam(model, img_array,
                    class_idx,
                    last_conv_layer=None):
    """
    Grad-CAM heatmap in [0,1] for class_idx, or None if no 4-D
    feature map can be found.  Works for:
      - Original_CNN (Sequential)
      - ResNet50V2_FT / EfficientNet wrappers (nested base model)
    on both Keras 2 (TF<=2.15) and Keras 3 (TF>=2.16).
    """
    it = tf.cast(np.expand_dims(img_array, 0), tf.float32)

    try:
        base = None
        if last_conv_layer is None:
            base, last_conv_layer = _find_last_4d_layer(model)
        if last_conv_layer is None:
            return None

        if base is None:
            # Flat model: build a 2-output model
            gm = tf.keras.Model(
                inputs  = model.inputs,
                outputs = [model.get_layer(last_conv_layer).output,
                           model.output])
            with tf.GradientTape() as tape:
                co, preds = gm(it, training=False)
                loss = preds[:, class_idx]
        else:
            # Nested base: run base -> conv map, then feed the
            # remaining head layers manually.
            base_feat = tf.keras.Model(
                inputs  = base.inputs,
                outputs = [base.get_layer(last_conv_layer).output,
                           base.output])
            # head = every layer after the base in the outer model
            outer   = model.layers
            b_index = outer.index(base)
            head    = outer[b_index + 1:]
            with tf.GradientTape() as tape:
                co, x = base_feat(it, training=False)
                for lyr in head:
                    x = lyr(x, training=False)
                preds = x
                loss  = preds[:, class_idx]

        grads = tape.gradient(loss, co)
        if grads is None:
            return None
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(co[0] * pooled, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"  WARNING: Grad-CAM failed: {e}")
        return None


def step_gradcam_analysis(results_json=None):
    """
    Grad-CAM for all classes.
    Validates model learns physics bands.
    """
    print("\n" + "="*60)
    print("REVISION H: GRAD-CAM ANALYSIS")
    print("  Qualitative check of model attention vs physics bands")
    print("="*60)

    df = load_results_df(results_json)
    if df is None:
        return None
    best      = df.loc[df['test_accuracy'].idxmax()]
    model     = safe_load_model(best['model_path'])
    if model is None:
        return None
    spec_type = best['spec_type']
    test_dir  = os.path.join(
        SPLIT_FINAL, spec_type, 'test')

    print(f"  Model    : {best['model_name']}")
    print(f"  Spec     : {spec_type}")
    print(f"  Accuracy : {best['test_accuracy']}%")

    fig, axes = plt.subplots(
        len(CLASS_NAMES), 3,
        figsize=(15, 5*len(CLASS_NAMES)))
    fig.suptitle(
        f'Grad-CAM: {best["model_name"]} + '
        f'{spec_type}\n'
        'Red=Focus | Verify vs physics bands',
        fontsize=12, fontweight='bold')

    summary = {}
    for ci, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue

        pngs = [f for f in os.listdir(cls_dir)
                if f.endswith('.png')
                and not f.startswith('aug')]
        if not pngs:
            pngs = [f for f in os.listdir(cls_dir)
                    if f.endswith('.png')]
        if not pngs:
            continue

        png_p = os.path.join(
            cls_dir, sorted(pngs)[0])
        img   = np.array(
            Image.open(png_p).convert('RGB')
            .resize(IMG_SIZE)
        ).astype(np.float32) / 255.0

        pred     = model.predict(
            np.expand_dims(img, 0), verbose=0)
        pred_cls = np.argmax(pred[0])
        pred_pr  = pred[0][pred_cls]
        correct  = (pred_cls == ci)
        status   = "CORRECT" if correct else "WRONG"

        heatmap  = compute_gradcam(model, img, ci)
        summary[cls] = {
            'predicted'   : CLASS_NAMES[pred_cls],
            'probability' : round(float(pred_pr), 4),
            'correct'     : bool(correct),
            'physics_band': PHYSICS_BANDS[cls]
        }
        print(f"\n  {cls}: pred="
              f"{CLASS_NAMES[pred_cls]} "
              f"({pred_pr:.2%}) [{status}]")

        axes[ci][0].imshow(img)
        axes[ci][0].set_title(
            f'True: {cls}\n'
            f'Pred: {CLASS_NAMES[pred_cls]} '
            f'({pred_pr:.2%}) [{status}]',
            fontsize=9)
        axes[ci][0].set_ylabel(
            f'{cls}\n{PHYSICS_BANDS[cls]}',
            fontsize=9, fontweight='bold')
        axes[ci][0].axis('off')

        if heatmap is None:
            axes[ci][1].text(0.5, 0.5, 'Grad-CAM\nunavailable',
                             ha='center', va='center',
                             transform=axes[ci][1].transAxes)
            axes[ci][1].axis('off')
            axes[ci][2].axis('off')
            summary[cls]['gradcam'] = 'unavailable'
        else:
            hm_big = np.array(
                Image.fromarray(
                    (heatmap * 255).astype(np.uint8)
                ).resize(IMG_SIZE, resample=Image.BILINEAR)
            ).astype(np.float32) / 255.0
            # Row-wise (frequency-axis) energy of the heatmap: which
            # vertical fraction of the spectrogram the model attends to.
            row_energy = hm_big.mean(axis=1)
            peak_row   = int(np.argmax(row_energy))
            summary[cls]['attention_peak_row_frac'] = round(
                peak_row / IMG_SIZE[0], 3)
            im = axes[ci][1].imshow(
                hm_big, cmap='jet',
                vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[ci][1],
                         fraction=0.046, pad=0.04)
            axes[ci][1].set_title(
                'Grad-CAM Heatmap', fontsize=9)
            axes[ci][1].axis('off')

            hmc     = plt.cm.jet(hm_big)[:,:,:3]
            overlay = np.clip(
                0.55*img + 0.45*hmc, 0, 1)
            axes[ci][2].imshow(overlay)
            axes[ci][2].set_title(
                'Overlay', fontsize=9)
            axes[ci][2].set_xlabel(
                f"Expected: {PHYSICS_BANDS[cls]}",
                fontsize=8, color='darkgreen',
                fontweight='bold')
            axes[ci][2].axis('off')

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(
                d, 'gradcam_all_classes.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: gradcam_all_classes.png")

    gf = os.path.join(
        RESULTS_DIR, 'gradcam_summary.json')
    with open(gf, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {gf}")
    return summary


# ============================================================
# REVISION STEP I: NOISE ROBUSTNESS
# ============================================================
def add_noise_paper_method(img_array, noise_db=50, rng=None):
    """
    Additive Gaussian noise applied in *spectrogram-image* space.

    noise_db is mapped to a relative noise RMS via
        noise_rms = signal_rms * 10**((noise_db - 94)/20)
    i.e. 50/60/70 dB -> 0.6% / 2.0% / 6.3% of the image RMS.

    NOTE FOR THE PAPER: this is an image-domain perturbation that
    emulates a given acoustic SNR; it is NOT noise injected into the
    raw waveform before spectrogram generation. State this explicitly
    in the methods section (reviewers will ask).
    """
    rng         = rng or np.random
    signal_norm = img_array.astype(np.float32) / 255.0
    signal_rms  = np.sqrt(np.mean(signal_norm**2))
    noise_rms   = signal_rms * (
        10**((noise_db - 94.0) / 20.0))
    noise_rms   = max(noise_rms, 0.001)
    noise       = rng.normal(0, noise_rms, signal_norm.shape)
    noisy       = np.clip(signal_norm + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)


def step_noise_robustness(
        results_json=None,
        noise_levels=None):
    """Test noise robustness on original test images."""
    if noise_levels is None:
        noise_levels = [50, 60, 70]

    print("\n" + "="*60)
    print("REVISION I: NOISE ROBUSTNESS")
    print(f"  Levels: {noise_levels} dB")
    print("  (image-domain Gaussian noise emulating acoustic SNR)")
    print("="*60)

    df = load_results_df(results_json)
    if df is None:
        return None
    top3 = df.nlargest(3, 'test_accuracy')

    all_noise = {}

    for _, row in top3.iterrows():
        mn  = row['model_name']
        st  = row['spec_type']
        key = f"{mn}_{st}"

        print(f"\n  {mn}+{st} "
              f"(clean={row['test_accuracy']}%)")

        model = safe_load_model(row['model_path'])
        if model is None:
            continue
        test_dir = os.path.join(
            SPLIT_FINAL, st, 'test')

        # Load all original test images ONCE (as uint8 to save RAM)
        imgs, labels = [], []
        for ci, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(test_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            pngs = [f for f in os.listdir(cls_dir)
                    if f.endswith('.png')
                    and not f.startswith('aug')]
            if not pngs:
                pngs = [f for f in os.listdir(cls_dir)
                        if f.endswith('.png')]
            for png in pngs:
                imgs.append(np.array(
                    Image.open(os.path.join(cls_dir, png))
                    .convert('RGB').resize(IMG_SIZE),
                    dtype=np.uint8))
                labels.append(ci)
        if not imgs:
            print("  ERROR: no test images found")
            continue
        imgs = np.stack(imgs)
        at   = np.array(labels)

        nr = {
            'model'    : mn, 'spec': st,
            'clean_acc': row['test_accuracy'],
            'levels'   : {}
        }

        for db in noise_levels:
            rng = np.random.RandomState(1000 + db)  # reproducible
            all_preds = []
            # Batched prediction (was: one predict() call per image)
            for b0 in range(0, len(imgs), BATCH_SIZE):
                batch = imgs[b0:b0 + BATCH_SIZE]
                noisy = np.stack([
                    add_noise_paper_method(im, db, rng)
                    for im in batch]).astype(np.float32) / 255.0
                pred  = model.predict(noisy, verbose=0)
                all_preds.extend(np.argmax(pred, axis=1).tolist())

            ap  = np.array(all_preds)
            ov  = np.mean(ap == at) * 100
            pc  = {}
            for i, cls in enumerate(CLASS_NAMES):
                mask = at == i
                if mask.sum() > 0:
                    pc[cls] = round(
                        np.mean(ap[mask]==i)*100,
                        2)
            nr['levels'][f'{db}dB'] = {
                'overall'  : round(ov, 2),
                'per_class': pc
            }
            print(f"    {db}dB → {ov:.2f}%  "
                  + "  ".join(
                    f"{c}: {pc.get(c,0):.1f}%"
                    for c in CLASS_NAMES))

        all_noise[key] = nr
        del model
        tf.keras.backend.clear_session()

    if not all_noise:
        print("  ERROR: no models evaluated")
        return None

    nf = os.path.join(
        RESULTS_DIR, 'noise_robustness.json')
    with open(nf, 'w') as f:
        json.dump(all_noise, f, indent=2, cls=NumpyEncoder)

    _plot_noise(all_noise, noise_levels)
    print(f"\n  Saved: {nf}")
    return all_noise


def _plot_noise(noise_results, noise_levels):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        'Noise Robustness Analysis\n'
        'Gaussian White Noise '
        '(Original Test Images Only)',
        fontsize=13, fontweight='bold')

    styles = ['-o','-s','-^','-D','-*']
    colors = ['#1f77b4','#ff7f0e','#2ca02c',
              '#d62728','#9467bd']
    x_lbs  = ['Clean'] + [
        f'{n}dB' for n in noise_levels]
    ax1    = axes[0]

    for i, (key, data) in enumerate(
            noise_results.items()):
        y = ([data['clean_acc']] +
             [data['levels'][f'{n}dB']['overall']
              for n in noise_levels])
        lbl = (f"{data['model']}\n"
               f"+{data['spec']}")
        ax1.plot(x_lbs, y,
                 styles[i % len(styles)],
                 color=colors[i % len(colors)],
                 linewidth=2.5, markersize=9,
                 label=lbl)
        for xi, yi in enumerate(y):
            ax1.annotate(
                f'{yi:.1f}%', (xi, yi),
                textcoords='offset points',
                xytext=(0, 8),
                ha='center', fontsize=8)

    ax1.axhline(y=70, color='red',
                linestyle='--', linewidth=2,
                label='70% benchmark')
    ax1.set_title('Overall Accuracy vs Noise',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Noise Level', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=8, loc='lower left')
    ax1.grid(True, alpha=0.3)

    # Per-class for best
    ax2      = axes[1]
    best_key = max(noise_results,
                   key=lambda k:
                   noise_results[k]['clean_acc'])
    bd       = noise_results[best_key]
    x  = np.arange(len(x_lbs))
    w  = 0.18
    for i, cls in enumerate(CLASS_NAMES):
        y = ([bd['clean_acc']] +
             [bd['levels'][f'{n}dB']
              ['per_class'].get(cls, 0)
              for n in noise_levels])
        ax2.bar(x + (i-1.5)*w, y, w,
                label=cls,
                color=REGIME_COLORS[cls],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5)
    ax2.axhline(y=70, color='red',
                linestyle='--', linewidth=1.5,
                alpha=0.7)
    ax2.set_title(
        f'Per-Class Accuracy vs Noise\n'
        f'({bd["model"]}+{bd["spec"]})',
        fontsize=12, fontweight='bold')
    ax2.set_xlabel('Noise Level', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_lbs, fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(
            os.path.join(
                d, 'noise_robustness.png'),
            dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: noise_robustness.png")


# ============================================================
# REVISION STEP J: CONTRIBUTION SECTION
# ============================================================
def step_contribution_section(
        results_json=None,
        stats_json=None,
        ablation_json=None):
    """
    Auto-generate paper-ready contribution section
    using actual experimental numbers.
    """
    print("\n" + "="*60)
    print("REVISION J: CONTRIBUTION SECTION")
    print("="*60)

    df = load_results_df(results_json)
    if df is None:
        return None
    best = df.loc[df['test_accuracy'].idxmax()]
    mn   = best['model_name']
    st   = best['spec_type']
    acc  = best['test_accuracy']
    f1   = best.get('prechf_f1', 0)

    def _load(p):
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return {}

    stats = _load(stats_json)
    abl   = _load(ablation_json)
    sig   = _load(os.path.join(RESULTS_DIR,
                               'statistical_significance.json'))
    noise = _load(os.path.join(RESULTS_DIR, 'noise_robustness.json'))
    sens  = _load(os.path.join(RESULTS_DIR,
                               'augmentation_sensitivity.json'))

    # ---- Params ----
    cnn_pm = df[df['model_name']=='Original_CNN']['params_M'].mean()
    res_pm = df[df['model_name']=='ResNet50V2_FT']['params_M'].mean()
    if np.isnan(cnn_pm): cnn_pm = 0.52
    if np.isnan(res_pm): res_pm = 24.75

    # ---- Multi-seed stats ----
    mean_f1 = stats.get('prechf_f1', {}).get('mean', f1)
    std_f1  = stats.get('prechf_f1', {}).get('std', 0)
    n_runs  = stats.get('n_runs', 1)

    # ---- Spectrogram ranking (best model) ----
    sub = df[df['model_name'] == mn].sort_values(
        'test_accuracy', ascending=False)
    spec_rank = ", ".join(
        f"{r.spec_type} {r.test_accuracy:.2f}%"
        for r in sub.itertuples())
    top_spec = sub.iloc[0]['spec_type'] if len(sub) else st

    # ---- CNN vs ResNet: computed, not asserted ----
    wins, total = 0, 0
    for s in SPEC_TYPES:
        c = df[(df.model_name=='Original_CNN') & (df.spec_type==s)]
        r = df[(df.model_name=='ResNet50V2_FT') & (df.spec_type==s)]
        if not c.empty and not r.empty:
            total += 1
            if c.iloc[0]['test_accuracy'] >= r.iloc[0]['test_accuracy']:
                wins += 1
    if total == 0:
        arch_claim = ("Only one architecture was evaluated; no "
                      "architecture comparison is claimed.")
    elif wins == total:
        arch_claim = (f"The lightweight CNN ({cnn_pm:.2f}M params) "
                      f"matches or outperforms ResNet50V2 "
                      f"({res_pm:.2f}M params) on all {total} "
                      f"spectrogram types.")
    else:
        arch_claim = (f"The lightweight CNN ({cnn_pm:.2f}M params) "
                      f"matches or outperforms ResNet50V2 "
                      f"({res_pm:.2f}M params) on {wins}/{total} "
                      f"spectrogram types.")

    # ---- CHROMA ----
    chroma = df[df.spec_type == 'CHROMA']
    chroma_claim = (
        f"CHROMA yields the lowest accuracy "
        f"({chroma['test_accuracy'].max():.2f}%), consistent with "
        f"pitch-class encoding discarding absolute frequency."
        if not chroma.empty and
           chroma['test_accuracy'].max() == df.groupby('spec_type')[
               'test_accuracy'].max().min()
        else "CHROMA was evaluated; see Table X for its ranking.")

    # ---- Ablation ----
    if abl:
        a_none = abl.get('A_None', {}).get('mean_f1')
        a_full = abl.get('C_Full', {}).get('mean_f1')
        abl_claim = (
            f"Pre-CHF F1 rises from {a_none:.3f} (no augmentation) "
            f"to {a_full:.3f} (full augmentation) in the ablation."
            if a_none is not None and a_full is not None
            else "See ablation table.")
    else:
        abl_claim = "[ablation not yet run]"

    # ---- Sensitivity / saturation ----
    if sens:
        ks = sorted(sens, key=lambda k: sens[k]['factor'])
        sens_claim = " -> ".join(
            f"{sens[k]['factor']}x: {sens[k]['mean_f1']:.3f}" for k in ks)
    else:
        sens_claim = "[sensitivity not yet run]"

    # ---- Noise ----
    if noise:
        bk = max(noise, key=lambda k: noise[k]['clean_acc'])
        lv = noise[bk]['levels']
        worst = min(v['overall'] for v in lv.values()) if lv else None
        noise_claim = (f"Accuracy stays >= {worst:.1f}% across "
                       f"{', '.join(lv.keys())} image-domain noise."
                       if worst is not None else "[noise not run]")
    else:
        noise_claim = "[noise robustness not yet run]"

    # ---- Significance ----
    def _sig_line(key, a, b):
        d = sig.get(key)
        if not d:
            return f"{a} vs {b}: [not yet run]"
        return (f"{a} vs {b}: p_t={d['p_ttest']:.4f}, "
                f"p_W={d['p_wilcoxon']:.4f}, n={d.get('n_pairs','?')} "
                f"-> {'significant' if d['significant'] else 'n.s.'}")
    sig_claim = (_sig_line('STFT_vs_MEL', 'STFT', 'MEL') + "\n   " +
                 _sig_line('CNN_vs_ResNet', 'CNN', 'ResNet50V2'))

    text = f"""
1.X MAIN CONTRIBUTIONS  (auto-generated from actual results)
=============================================================

1. Systematic spectrogram comparison:
   Five time-frequency representations (STFT, MEL, MFCC, CQT,
   CHROMA) are evaluated. Best: {top_spec} with {mn}
   ({acc:.2f}% test accuracy). Ranking ({mn}): {spec_rank}.

2. Pre-CHF acoustic detection:
   Pre-CHF F1 = {f1:.4f} (single best run); mean {mean_f1:.4f}
   +/- {std_f1:.4f} over {n_runs} seed(s). Prior acoustic-only
   work reports F1 = 0.000 for Pre-CHF.

3. Architecture:
   {arch_claim}

4. CHROMA:
   {chroma_claim}

5. Augmentation ablation:
   {abl_claim}
   Sensitivity (Pre-CHF F1 vs expansion factor): {sens_claim}

6. Noise robustness:
   {noise_claim}

7. Statistical validation:
   {sig_claim}
   95% bootstrap CIs on P/R/F1 are reported in Table X.

LIMITATION (must include):
   The Pre-CHF training set contains ~279 original recordings
   expanded to {TARGET_N:,} by augmentation; the test set holds
   ~60 original Pre-CHF recordings. Noise robustness is assessed
   with image-domain perturbations, not waveform noise.
"""

    print(text)

    cf = os.path.join(
        RESULTS_DIR, 'contribution_section.txt')
    with open(cf, 'w') as f:
        f.write(text)
    print(f"  Saved: {cf}")
    return text


# ============================================================
# FINAL SUMMARY PRINTER
# ============================================================
def print_final_summary():
    """Print checklist of all output files."""
    print("\n" + "="*65)
    print("COMPLETE OUTPUT SUMMARY")
    print("="*65)

    file_map = [
        # Data files
        (BASE_DIR, 'all_results.json',
         'Training results (all experiments)'),
        (RESULTS_DIR, 'results_graded.csv',
         'Graded results CSV'),
        # Ablation
        (RESULTS_DIR, 'ablation_study.json',
         'Ablation Study (Exp A–E)'),
        (RESULTS_DIR, 'augmentation_sensitivity.json',
         'Augmentation Sensitivity (1x–46x)'),
        # Stats
        (RESULTS_DIR, 'statistical_significance.json',
         'Significance Tests (t-test + Wilcoxon)'),
        (RESULTS_DIR, 'confidence_intervals.json',
         '95% Bootstrap CIs (P/R/F1)'),
        (RESULTS_DIR, 'calibration.json',
         'Calibration + ECE per class'),
        (RESULTS_DIR, 'roc.json',
         'ROC + AUC (all + Pre-CHF vs NB)'),
        (RESULTS_DIR, 'hard_negative.json',
         'NB↔Pre-CHF confusion analysis'),
        (RESULTS_DIR, 'gradcam_summary.json',
         'Grad-CAM physics validation'),
        (RESULTS_DIR, 'noise_robustness.json',
         'Noise robustness (50/60/70 dB)'),
        (RESULTS_DIR, 'contribution_section.txt',
         'Paper contribution section'),
        (RESULTS_DIR, 'prechf_analysis.json',
         'Pre-CHF detailed analysis'),
        (RESULTS_DIR, 'statistical_validation.json',
         '5-seed validation (Table 5)'),
        (RESULTS_DIR, 'kfold_cv.json',
         'Stratified K-fold CV (Table 6)'),
        (RESULTS_DIR, 'pr_curve.json',
         'PR-AUC (Pre-CHF)'),
        (RESULTS_DIR, 'threshold_sensitivity.json',
         'Threshold sensitivity (Table 7)'),
        # Figures
        (FIGURES_DIR, 'fig_dataset.png',
         'Fig: Dataset distribution'),
        (FIGURES_DIR, 'fig_pipeline.png',
         'Fig: Pipeline flowchart'),
        (FIGURES_DIR, 'fig_spectrogram_examples.png',
         'Fig: Spectrogram examples'),
        (FIGURES_DIR, 'complete_analysis.png',
         'Fig: Complete heatmaps + grades'),
        (FIGURES_DIR, 'cnn_spectrogram_comparison.png',
         'Fig: CNN spectrogram ranking'),
        (FIGURES_DIR, 'perclass_f1_analysis.png',
         'Fig: Per-class F1 + CNN vs ResNet'),
        (FIGURES_DIR, 'ablation_study.png',
         'Fig: Ablation study'),
        (FIGURES_DIR, 'augmentation_sensitivity.png',
         'Fig: Augmentation sensitivity'),
        (FIGURES_DIR, 'statistical_significance.png',
         'Fig: Statistical significance'),
        (FIGURES_DIR, 'calibration_analysis.png',
         'Fig: Calibration + ECE'),
        (FIGURES_DIR, 'roc_curves.png',
         'Fig: ROC curves'),
        (FIGURES_DIR, 'confidence_intervals.png',
         'Fig: CI error bars'),
        (FIGURES_DIR, 'hard_negative_analysis.png',
         'Fig: Hard-negative analysis'),
        (FIGURES_DIR, 'gradcam_all_classes.png',
         'Fig: Grad-CAM all classes'),
        (FIGURES_DIR, 'noise_robustness.png',
         'Fig: Noise robustness'),
        (FIGURES_DIR, 'prechf_analysis.png',
         'Fig: Pre-CHF analysis'),
        (FIGURES_DIR, 'statistical_validation.png',
         'Fig: 5-seed validation'),
        (FIGURES_DIR, 'pr_curve_prechf.png',
         'Fig: PR curve (Pre-CHF)'),
        (FIGURES_DIR, 'threshold_sensitivity.png',
         'Fig: Threshold sensitivity'),
    ]

    ok_count = 0
    for folder, fname, desc in file_map:
        full   = os.path.join(folder, fname)
        exists = os.path.exists(full)
        status = "✓" if exists else "✗"
        if exists:
            ok_count += 1
        print(f"  {status} {desc}")
        print(f"      {full}")

    print(f"\n  {ok_count}/{len(file_map)} files present")

    print("\n  REVIEWER CHECKLIST:")
    checks = [
        "Ablation (None/Mild/Full/No-FreqMask"
        "/No-Noise) with mean±std",
        "Augmentation saturation shown "
        "(anti-artifact evidence)",
        "Paired t-test + Wilcoxon p-values",
        "95% Bootstrap CIs on P/R/F1",
        "Reliability diagram + ECE per class",
        "ROC (OVR) + binary Pre-CHF vs NB AUC",
        "NB↔Pre-CHF confusion + confidence dist",
        "Grad-CAM physics band validation",
        "Noise robustness 50/60/70 dB",
        "Contribution section with real numbers",
        "Limitation sentence (augmented Pre-CHF)",
    ]
    for i, c in enumerate(checks, 1):
        print(f"    {i:2d}. {c}")

# ============================================================
# MERGED FROM run_kfold_pr_threshold.py
# (Pre-CHF detail, 5-seed statistical validation (Table 5),
#  stratified K-fold CV (Table 6), PR curve (Fig 15),
#  threshold sensitivity (Table 7))
# ============================================================

def step4b_prechf_analysis(results_json=None):
    print("\n" + "="*60)
    print("STEP 4B: PRE-CHF DETAILED ANALYSIS")
    print("="*60)

    ctx = load_best_model_and_test(results_json)
    if ctx is None:
        return None
    best      = ctx['best']
    spec_type = ctx['spec_type']
    y_pred    = ctx['y_pred']
    y_true    = ctx['y_true']
    print(f"  Best: {best['model_name']} + {spec_type}")

    cm     = confusion_matrix(y_true, y_pred,
                              labels=list(range(NUM_CLASSES)))
    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    prechf_m = report.get('Pre-CHF', {})
    recall   = prechf_m.get('recall', 0)
    f1       = prechf_m.get('f1-score', 0)
    print(f"  Pre-CHF: P={prechf_m.get('precision',0):.4f} "
          f"R={recall:.4f} F1={f1:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Pre-CHF Analysis — '
                 f'{best["model_name"]} + {spec_type}',
                 fontsize=13, fontweight='bold')
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    row_sum[row_sum == 0] = 1.0
    cm_norm = cm.astype('float') / row_sum
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                ax=axes[0], xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    prechf_idx = CLASS_NAMES.index('Pre-CHF')
    axes[1].bar(CLASS_NAMES, cm_norm[prechf_idx] * 100,
                color=[REGIME_COLORS[c] for c in CLASS_NAMES],
                edgecolor='black')
    axes[1].set_title('Pre-CHF Classification Distribution')
    axes[1].set_ylabel('% of Pre-CHF samples')
    axes[1].set_ylim(0, 110)
    for i, v in enumerate(cm_norm[prechf_idx]):
        axes[1].text(i, v*100+2, f'{v*100:.1f}%',
                     ha='center', fontweight='bold')

    metrics_data = {'Precision': [], 'Recall': [], 'F1': []}
    for cls in CLASS_NAMES:
        r = report.get(cls, {})
        metrics_data['Precision'].append(r.get('precision', 0))
        metrics_data['Recall'].append(r.get('recall', 0))
        metrics_data['F1'].append(r.get('f1-score', 0))
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    for i, (metric, vals) in enumerate(metrics_data.items()):
        axes[2].bar(x + (i-1)*w, vals, w, label=metric,
                    alpha=0.85, color=['#1f77b4','#ff7f0e','#2ca02c'][i],
                    edgecolor='black')
    axes[2].set_title('Per-Class Metrics')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CLASS_NAMES)
    axes[2].set_ylim(0, 1.15)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'prechf_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    analysis = {
        'model': best['model_name'], 'spec_type': spec_type,
        'prechf_precision': round(prechf_m.get('precision',0), 4),
        'prechf_recall'   : round(recall, 4),
        'prechf_f1'       : round(f1, 4),
        'full_report'     : report
    }
    with open(os.path.join(RESULTS_DIR, 'prechf_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved: prechf_analysis.json")
    return analysis


def step4d_statistical_validation(n_runs=5, best_config=None):
    print("\n" + "="*60)
    print(f"STEP 4D: STATISTICAL VALIDATION ({n_runs} runs)")
    print("="*60)

    if best_config is None:
        df = load_results_df()
        if df is None:
            return None
        best = df.loc[df['test_accuracy'].idxmax()]
        best_config = {
            'model_name': best['model_name'],
            'spec_type' : best['spec_type']
        }

    mn    = best_config['model_name']
    st    = best_config['spec_type']
    seeds = [42, 123, 456, 789, 1024,
             2024, 999, 314, 271, 100][:n_runs]

    print(f"  Config: {mn} + {st}")
    print(f"  Seeds : {seeds}")
    run_results = []

    for run_idx, seed in enumerate(seeds):
        print(f"\n  Run {run_idx+1}/{n_runs} (seed={seed})")
        set_all_seeds(seed)

        model      = BUILDERS[mn](input_shape=IMG_SIZE+(3,),
                                  num_classes=NUM_CLASSES)
        tg, vg, te = get_generators(st)
        if tg is None:
            tf.keras.backend.clear_session()
            continue
        cw = get_class_weights(tg)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        hist = model.fit(
            tg, epochs=30, validation_data=vg,
            callbacks=[EarlyStopping(monitor='val_accuracy',
                                     patience=8,
                                     restore_best_weights=True,
                                     verbose=0)],
            class_weight=cw, verbose=0)
        te.reset()
        _, acc = model.evaluate(te, verbose=0)
        te.reset()
        y_pred    = np.argmax(model.predict(te, verbose=0), axis=1)
        y_true    = te.classes
        report    = classification_report(
            y_true, y_pred, labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES, output_dict=True,
            zero_division=0)
        prechf_f1 = report.get('Pre-CHF', {}).get('f1-score', 0)
        run_results.append({
            'seed': seed,
            'accuracy': round(acc * 100, 3),
            'prechf_f1': round(prechf_f1, 4),
            'epochs': len(hist.history['accuracy'])
        })
        print(f"    Acc:{acc*100:.2f}% | Pre-CHF F1:{prechf_f1:.4f}")
        del model
        tf.keras.backend.clear_session()

    if not run_results:
        print("  ERROR: no successful runs")
        return None

    n_ok     = len(run_results)
    accs     = [r['accuracy']  for r in run_results]
    f1s      = [r['prechf_f1'] for r in run_results]
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs, ddof=1)) if n_ok > 1 else 0.0
    mean_f1  = float(np.mean(f1s))
    std_f1   = float(np.std(f1s, ddof=1))  if n_ok > 1 else 0.0
    # t-based CI is appropriate for small n (n=5 -> t=2.776, not 1.96)
    t_crit   = scipy_stats.t.ppf(0.975, df=n_ok-1) if n_ok > 1 else 0.0
    ci95_acc = t_crit * std_acc / np.sqrt(n_ok)
    ci95_f1  = t_crit * std_f1  / np.sqrt(n_ok)
    n_runs   = n_ok

    print(f"\n  SUMMARY ({n_runs} runs):")
    print(f"  Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Pre-CHF F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  95% CI accuracy : [{mean_acc-ci95_acc:.2f}%, "
          f"{mean_acc+ci95_acc:.2f}%]")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Statistical Validation ({n_runs} runs)\n'
                 f'{mn} + {st}', fontsize=12, fontweight='bold')
    run_nums = list(range(1, len(run_results)+1))
    for ax, vals, mean, std, label, color, fmt in [
        (axes[0], accs,  mean_acc, std_acc,
         'Accuracy (%)', 'blue',  'b-o'),
        (axes[1], f1s,   mean_f1,  std_f1,
         'Pre-CHF F1',   'green', 'g-s'),
    ]:
        ax.plot(run_nums, vals, fmt, linewidth=2, markersize=8)
        ax.axhline(y=mean, color='red', linestyle='--',
                   linewidth=2, label=f'Mean:{mean:.3f}')
        ax.fill_between(run_nums, mean-std, mean+std,
                        alpha=0.2, label=f'±1σ:{std:.3f}')
        ax.set_xlabel('Run'); ax.set_ylabel(label)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'statistical_validation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    stats = {
        'model': mn, 'spec_type': st, 'n_runs': n_runs,
        'accuracy' : {'mean': round(mean_acc, 3),
                      'std' : round(std_acc, 3),
                      'ci95': round(ci95_acc, 3)},
        'prechf_f1': {'mean': round(mean_f1, 4),
                      'std' : round(std_f1, 4),
                      'ci95': round(ci95_f1, 4)},
        'run_details': run_results
    }
    with open(os.path.join(RESULTS_DIR,
              'statistical_validation.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: statistical_validation.json")
    return stats


def _make_tf_dataset(paths, labels, batch_size, shuffle,
                     augment=False, seed=0):
    """
    Streaming tf.data pipeline (decode PNG -> resize -> [0,1]).
    Avoids materialising the whole fold in RAM (the previous
    implementation tried to hold ~30 GB of float32 images).
    """
    paths  = tf.constant(list(paths))
    labels = tf.constant(np.asarray(labels, dtype=np.int32))
    ds     = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed,
                        reshuffle_each_iteration=True)

    def _load(p, y):
        img = tf.io.read_file(p)
        img = tf.io.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        if augment:
            # Light on-the-fly augmentation, analogous to the
            # ImageDataGenerator settings used in get_generators().
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.clip_by_value(img, 0.0, 1.0)
        return img, tf.one_hot(y, NUM_CLASSES)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def step4e_kfold_cv(
        spec_type=None,
        n_splits=5,
        epochs=30):
    """
    Stratified K-fold CV on ORIGINAL (non-augmented) spectrograms
    pooled from the train+validation splits; the held-out test split
    is never touched.

    Fixes vs. the previous version:
      * Used SPLIT_FINAL/train, i.e. augmented copies of the same
        source image could land in both the train and val fold
        (leakage) -> now originals only.
      * Loaded every fold into RAM as float32 (~30 GB) -> now streams
        with tf.data.
      * Ignored its spec_type argument -> now honoured (falls back to
        the best model's spec when None).
    """
    print("\n" + "="*60)
    print("STEP 4E: STRATIFIED K-FOLD CV (originals only)")
    print("="*60)

    df = load_results_df()
    if df is None:
        return None
    best = df.loc[df['test_accuracy'].idxmax()]
    mn   = best['model_name']
    st   = spec_type or best['spec_type']
    print(f"  Model: {mn} | Spec: {st}")

    filepaths, labels = [], []
    for split in ['train', 'validation']:
        for cls_idx, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(SPLIT_ORIG, st, split, cls)
            if not os.path.exists(cls_dir):
                continue
            files = [os.path.join(cls_dir, f)
                     for f in os.listdir(cls_dir)
                     if f.endswith('.png') and not f.startswith('aug')]
            filepaths.extend(files)
            labels.extend([cls_idx] * len(files))

    filepaths = np.array(filepaths)
    labels    = np.array(labels)
    if len(filepaths) == 0:
        print(f"  ERROR: no original images under {SPLIT_ORIG}/{st}")
        return None
    print(f"  Pooled originals: {len(filepaths)} "
          f"({dict(zip(*np.unique(labels, return_counts=True)))})")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=42)

    fold_accs, fold_f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(filepaths, labels)):

        print(f"\n  Fold {fold+1}/{n_splits}")
        print(f"    Train: {len(train_idx)} | Val: {len(val_idx)}")
        set_all_seeds(fold)

        # Balanced class weights (originals are heavily imbalanced)
        cw_arr = compute_class_weight(
            'balanced', classes=np.arange(NUM_CLASSES),
            y=labels[train_idx])
        cw = dict(enumerate(cw_arr))

        model = BUILDERS[mn](input_shape=IMG_SIZE + (3,),
                             num_classes=NUM_CLASSES)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        train_ds = _make_tf_dataset(
            filepaths[train_idx], labels[train_idx],
            BATCH_SIZE, shuffle=True, augment=True, seed=fold)
        val_ds   = _make_tf_dataset(
            filepaths[val_idx], labels[val_idx],
            BATCH_SIZE, shuffle=False)

        model.fit(
            train_ds, validation_data=val_ds,
            epochs=epochs, class_weight=cw,
            callbacks=[EarlyStopping(monitor='val_accuracy',
                                     patience=8,
                                     restore_best_weights=True,
                                     verbose=0)],
            verbose=0)

        y_pred      = np.argmax(model.predict(val_ds, verbose=0), axis=1)
        y_true_fold = labels[val_idx]

        fold_acc = np.mean(y_pred == y_true_fold) * 100
        report   = classification_report(
            y_true_fold, y_pred, labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES, output_dict=True,
            zero_division=0)
        fold_f1  = report.get('Pre-CHF', {}).get('f1-score', 0)

        fold_accs.append(float(fold_acc))
        fold_f1s.append(float(fold_f1))
        print(f"    Acc: {fold_acc:.2f}% | Pre-CHF F1: {fold_f1:.4f}")

        del model
        tf.keras.backend.clear_session()

    if not fold_accs:
        print("  ERROR: no folds completed")
        return None

    mean_acc = safe_mean(fold_accs); std_acc = safe_std(fold_accs)
    mean_f1  = safe_mean(fold_f1s);  std_f1  = safe_std(fold_f1s)

    print(f"\n  SUMMARY ({n_splits}-fold CV, originals only)")
    print(f"  Accuracy  : {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Pre-CHF F1: {mean_f1:.4f} ± {std_f1:.4f}")

    kfold_stats = {
        'model': mn, 'spec_type': st,
        'n_splits': n_splits,
        'data': 'SPLIT_ORIG train+validation originals (no aug)',
        'n_samples': int(len(filepaths)),
        'accuracy': {
            'mean': round(mean_acc, 3),
            'std': round(std_acc, 3),
            'per_fold': [round(v, 3) for v in fold_accs]
        },
        'prechf_f1': {
            'mean': round(mean_f1, 4),
            'std': round(std_f1, 4),
            'per_fold': [round(v, 4) for v in fold_f1s]
        }
    }

    with open(os.path.join(RESULTS_DIR, 'kfold_cv.json'), 'w') as f:
        json.dump(kfold_stats, f, indent=2, cls=NumpyEncoder)

    print(f"  Saved: kfold_cv.json")
    return kfold_stats


def step4f_pr_curves(results_json=None):

    print("\n" + "="*60)
    print("STEP 4F: PR CURVES")
    print("="*60)

    ctx = load_best_model_and_test(results_json)
    if ctx is None:
        return None
    y_probs = ctx['y_probs']
    y_true  = ctx['y_true']

    prechf_idx = CLASS_NAMES.index('Pre-CHF')
    y_true_bin = (y_true == prechf_idx).astype(int)
    y_scores   = y_probs[:, prechf_idx]

    if y_true_bin.sum() == 0:
        print("  ERROR: no Pre-CHF samples in test set")
        return None

    precision, recall, _ = precision_recall_curve(y_true_bin, y_scores)
    pr_auc = average_precision_score(y_true_bin, y_scores)
    # Random-classifier baseline = class prevalence (was hardcoded 2.5%)
    prevalence = float(y_true_bin.mean())

    print(f"  PR-AUC: {pr_auc:.4f}  (prevalence={prevalence:.4f})")

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        recall,
        precision,
        linewidth=2.5,
        color='#1f77b4',
        label=f'PR-AUC = {pr_auc:.4f}'
    )

    ax.fill_between(recall, precision,
                    alpha=0.2, color='blue')

    ax.axhline(y=prevalence, color='red',
               linestyle='--', linewidth=1.5,
               label=f'Random baseline ({prevalence:.1%})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)

    ax.set_title('Precision-Recall Curve\nPre-CHF')

    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(os.path.join(d, 'pr_curve_prechf.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: pr_curve_prechf.png")

    out = {'pr_auc': round(float(pr_auc), 4),
           'prevalence': round(prevalence, 4),
           'model': ctx['best']['model_name'],
           'spec_type': ctx['spec_type']}
    with open(os.path.join(RESULTS_DIR, 'pr_curve.json'), 'w') as f:
        json.dump(out, f, indent=2)

    tf.keras.backend.clear_session()
    return out


def step4g_threshold_sensitivity(results_json=None):

    print("\n" + "="*60)
    print("STEP 4G: THRESHOLD SENSITIVITY")
    print("="*60)

    ctx = load_best_model_and_test(results_json)
    if ctx is None:
        return None
    y_probs = ctx['y_probs']
    y_true  = ctx['y_true']

    prechf_idx = CLASS_NAMES.index('Pre-CHF')
    y_true_bin = (y_true == prechf_idx).astype(int)
    y_scores   = y_probs[:, prechf_idx]

    if y_true_bin.sum() == 0:
        print("  ERROR: no Pre-CHF samples in test set")
        return None

    thresholds = np.arange(0.1, 1.0, 0.05)

    precision_list = []
    recall_list = []
    f1_list = []

    for th in thresholds:

        y_pred_bin = (y_scores >= th).astype(int)

        p = precision_score(
            y_true_bin, y_pred_bin, zero_division=0)

        r = recall_score(
            y_true_bin, y_pred_bin, zero_division=0)

        f1 = f1_score(
            y_true_bin, y_pred_bin, zero_division=0)

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)

    best_f1_idx = int(np.argmax(f1_list))
    best_thresh = thresholds[best_f1_idx]
    print(f"  Best threshold (F1): {best_thresh:.2f} "
          f"→ F1={f1_list[best_f1_idx]:.4f}")

    plt.figure(figsize=(8, 6))

    plt.plot(thresholds, precision_list, linewidth=2, label='Precision')
    plt.plot(thresholds, recall_list,    linewidth=2, label='Recall')
    plt.plot(thresholds, f1_list,        linewidth=2, label='F1-score')

    plt.axvline(x=best_thresh, color='red', linestyle='--',
                linewidth=1.5, label=f'Best F1 @ {best_thresh:.2f}')

    plt.xlabel('Decision Threshold')
    plt.ylabel('Metric')
    plt.title('Threshold Sensitivity\nPre-CHF')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    for d in [FIGURES_DIR, PAPER_FIGS]:
        plt.savefig(os.path.join(d, 'threshold_sensitivity.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: threshold_sensitivity.png")

    out = {
        'model'         : ctx['best']['model_name'],
        'spec_type'     : ctx['spec_type'],
        'best_threshold': round(float(best_thresh), 4),
        'best_f1'       : round(float(f1_list[best_f1_idx]), 4),
        'table': [
            {'threshold': round(float(t), 2),
             'precision': round(float(p), 4),
             'recall'   : round(float(r), 4),
             'f1'       : round(float(f), 4)}
            for t, p, r, f in zip(thresholds, precision_list,
                                  recall_list, f1_list)],
        'note': ('Threshold selected on the TEST set for illustration '
                 'only; for deployment, select on validation data.'),
    }
    with open(os.path.join(RESULTS_DIR,
                           'threshold_sensitivity.json'), 'w') as f:
        json.dump(out, f, indent=2)

    tf.keras.backend.clear_session()
    return out




# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Complete Boiling Pipeline '
                    '- Journal Revision')
    parser.add_argument(
        '--steps', nargs='+', default=['all'],
        choices=[
            'all',
            # Data prep
            'spectrogram', 'split', 'augment',
            'check',
            # Training
            'train',
            # Analysis
            'analysis', 'figures',
            # Revision
            'ablation', 'sensitivity',
            'significance', 'calibration',
            'roc', 'ci', 'hard_negative',
            'gradcam', 'noise',
            'contributions',
            # Merged from run_kfold_pr_threshold.py
            'prechf',       # Pre-CHF detailed confusion analysis
            'stats5seed',   # 5-seed statistical validation (paper Table 5)
            'kfold',        # stratified k-fold CV (paper Table 6)
            'pr',           # precision-recall curve (paper Figure 15)
            'threshold',    # decision-threshold sensitivity (paper Table 7)
        ],
        help='Steps to run')
    parser.add_argument(
        '--spec_types', nargs='+',
        default=SPEC_TYPES)
    parser.add_argument(
        '--models', nargs='+',
        default=list(BUILDERS.keys()))
    parser.add_argument(
        '--epochs', type=int, default=50)
    parser.add_argument(
        '--n_seeds', type=int, default=5)
    parser.add_argument(
        '--ablation_seeds', type=int, default=3)
    parser.add_argument(
        '--n_bootstrap', type=int, default=1000)
    parser.add_argument(
        '--noise_levels', nargs='+',
        type=int, default=[50, 60, 70])
    parser.add_argument(
        '--kfold_splits', type=int, default=5)
    parser.add_argument(
        '--stop_on_error', action='store_true',
        help='Abort on the first failing step instead of '
             'continuing with the remaining ones')

    args   = parser.parse_args()
    steps  = args.steps
    do_all = ('all' in steps)

    setup_gpu()

    rj = os.path.join(BASE_DIR, 'all_results.json')
    sj = os.path.join(RESULTS_DIR,
                      'statistical_validation.json')
    aj = os.path.join(RESULTS_DIR,
                      'ablation_study.json')

    failures = []

    def run(name, fn, *a, **kw):
        """Run a step; log + continue on failure (unless asked not to)."""
        if not (do_all or name in steps):
            return None
        try:
            return fn(*a, **kw)
        except Exception as e:
            import traceback
            print(f"\n  !!! STEP '{name}' FAILED: {e}")
            traceback.print_exc()
            failures.append(name)
            if args.stop_on_error:
                raise
            return None

    # ── Data preparation ──
    run('spectrogram', step1a_generate_spectrograms)
    run('split',       step1b_split_originals)
    run('augment',     step1c_augment_train_only)

    if do_all or 'check' in steps:
        ok = run('check', step2_check)
        if not ok and do_all:
            print("\n  Dataset check FAILED. Stopping.")
            return

    # ── Training ──
    run('train', step3_train_all,
        spec_types  = args.spec_types,
        model_names = args.models,
        epochs      = args.epochs)

    # ── Core analysis ──
    run('analysis', step4_analyze_results)

    # ── Paper figures ──
    run('figures', step5_missing_figures)

    # ── Get best config for revision steps ──
    def get_best():
        df = load_results_df(rj)
        if df is None:
            return 'Original_CNN', 'STFT'
        b = df.loc[df['test_accuracy'].idxmax()]
        return b['model_name'], b['spec_type']

    best_mn, best_st = get_best()

    # ── Revision analyses (training-heavy first) ──
    run('ablation', step_ablation_study,
        model_name = best_mn,
        spec_type  = best_st,
        epochs     = min(args.epochs, 40),
        n_seeds    = args.ablation_seeds)

    run('sensitivity', step_augmentation_sensitivity,
        model_name = best_mn,
        spec_type  = best_st,
        epochs     = min(args.epochs, 40),
        n_seeds    = args.ablation_seeds)

    run('significance', step_statistical_significance,
        results_json = rj,
        n_seeds      = args.n_seeds)

    # 5-seed validation must run BEFORE 'contributions', which reads
    # statistical_validation.json (previously ran after -> stale text)
    run('stats5seed', step4d_statistical_validation,
        n_runs=args.n_seeds,
        best_config={'model_name': best_mn,
                     'spec_type': best_st})

    run('kfold', step4e_kfold_cv,
        spec_type=best_st,
        n_splits=args.kfold_splits,
        epochs=min(args.epochs, 30))

    # ── Post-hoc analyses on the saved best model ──
    run('calibration',   step_calibration_analysis,   results_json=rj)
    run('roc',           step_roc_curves,             results_json=rj)
    run('ci',            step_confidence_intervals,
        results_json=rj, n_bootstrap=args.n_bootstrap)
    run('hard_negative', step_hard_negative_analysis, results_json=rj)
    run('gradcam',       step_gradcam_analysis,       results_json=rj)
    run('noise',         step_noise_robustness,
        results_json=rj, noise_levels=args.noise_levels)
    run('prechf',        step4b_prechf_analysis,      results_json=rj)
    run('pr',            step4f_pr_curves,            results_json=rj)
    run('threshold',     step4g_threshold_sensitivity, results_json=rj)

    # ── Contribution text LAST so it sees every result file ──
    run('contributions', step_contribution_section,
        results_json  = rj,
        stats_json    = sj,
        ablation_json = aj)

    print_final_summary()

    if failures:
        print(f"\n  {len(failures)} step(s) FAILED: "
              f"{', '.join(failures)}")
    else:
        print("\n  All requested steps completed.")


if __name__ == '__main__':
    main()
