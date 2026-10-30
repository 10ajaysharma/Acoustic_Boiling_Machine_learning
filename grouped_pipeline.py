#!/usr/bin/env python3
"""
================================================================
GROUPED / LEAKAGE-FREE PIPELINE  +  PHYSICS MODULE
Companion to acousticboiling.py.py  --  journal revision
================================================================

WHY THIS FILE EXISTS
--------------------
The original pipeline splits the 16,282 one-second clips with
`random.shuffle` at CLIP level.  Section 3.1 of the manuscript states
the clips were produced by "segmenting audio signals into one-second
files" after holding each power level steady for ~1000 s.  Consecutive
clips are therefore consecutive seconds of the SAME continuous
recording: same hydrophone placement, same vessel modes, same
background hum, same power step.

A random clip split puts second t in train and second t+1 in test.
The network can reach ~99.75% by recognising the *session*, not the
boiling regime.  Pre-CHF is only ~399 s total -- plausibly one or two
continuous runs -- so its 61 "unseen" test clips are interleaved with
its 279 training clips.

This file provides the fix:

  1. discover_groups()      -- recover session / power-step structure
  2. grouped_split()        -- split by SESSION, with a guard band
  3. generate_spectrograms()-- band-limited 0-1000 Hz (see note below)
  4. grouped_kfold()        -- GroupKFold on originals, streamed
  5. noise_robustness_wav() -- TRUE waveform-domain SNR
  6. wilson_ci()/report     -- honest small-sample intervals
  7. gradcam_resolution()   -- audits whether Grad-CAM can resolve
                               the frequency bands you claim
  8. physics.*              -- Minnaert / Fritz / Zuber / Taylor
                               computed, not quoted

BAND-LIMITING NOTE
------------------
With n_fft=512 at SR=10 kHz on a linear 0-5000 Hz axis, the entire
discriminative range (200-500 Hz) occupies 13 of 224 image rows (6%).
94% of every spectrogram is uninformative and GlobalAveragePooling
averages over it.  Setting FMAX_ANALYSIS=1000 Hz makes the physics
bands span ~50% of the image and gives Grad-CAM ~4.5 Hz per STFT bin.

USAGE
-----
  # Step 0 -- ALWAYS RUN THIS FIRST. Look at the report before anything else.
  python3 grouped_pipeline.py --steps inspect

  # then
  python3 grouped_pipeline.py --steps groups spectrogram split augment check
  python3 grouped_pipeline.py --steps physics
  python3 grouped_pipeline.py --steps kfold noise report
================================================================
"""

import os, re, json, math, shutil, random, argparse, warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG  (mirrors acousticboiling.py.py, with the fixes)
# ============================================================
DATA_DIR = os.path.join(os.path.expanduser("~"),
                        "Dataset_of_fluid_boiling",
                        "Dataset_sound_of_boiling")
BASE_DIR = os.path.join(os.path.expanduser("~"),
                        "coustic_boiling_grouped")

SPEC_DIR    = os.path.join(BASE_DIR, "Spectrograms")
SPLIT_ORIG  = os.path.join(BASE_DIR, "Split_Original")
SPLIT_FINAL = os.path.join(BASE_DIR, "Split_Final")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

for _d in (SPEC_DIR, SPLIT_ORIG, SPLIT_FINAL,
           RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

CLASS_NAMES = ['BKG', 'NB', 'Pre-CHF', 'TB']
FOLDERS     = CLASS_NAMES
SPEC_TYPES  = ['STFT', 'MEL', 'MFCC', 'CQT', 'CHROMA']
NUM_CLASSES = 4

SR      = 10000
N_FFT   = 512
HOP_LEN = 256
N_MELS  = 64
N_MFCC  = 40
N_CQT   = 72

# ---- THE BAND-LIMIT FIX ----
# 5000 Hz (Nyquist) wastes 94% of the image. 1000 Hz keeps a
# 2x margin above the highest physical band (400-500 Hz).
FMAX_ANALYSIS = 1000

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
TARGET_N   = 13000
SEED       = 42

# Guard band: clips dropped either side of every session boundary.
# Kills residual autocorrelation between adjacent train/test clips.
GUARD_CLIPS = 5

SPEC_CMAP = {'STFT': 'viridis', 'MEL': 'magma', 'MFCC': 'coolwarm',
             'CQT': 'inferno', 'CHROMA': 'plasma'}

PHYSICS_BANDS_HZ = {'BKG': (0, 100), 'NB': (200, 250),
                    'Pre-CHF': (200, 400), 'TB': (400, 500)}


# ============================================================
#  1. PHYSICS  --  computed, not quoted
# ============================================================
class BoilingPhysics:
    """
    Saturated pool boiling of distilled water at 1 atm on a
    25 mm smooth copper disc (the Sinha et al. apparatus).

    Every number below is DERIVED. Put these in the paper instead
    of repeating '200-250 Hz [3]' eight times.
    """
    # Fluid properties, saturated water @ 100 C, 101.325 kPa
    P0      = 101325.0     # Pa
    g       = 9.81         # m/s^2
    rho_l   = 957.9        # kg/m^3
    rho_v   = 0.5977       # kg/m^3
    sigma   = 0.0589       # N/m
    h_fg    = 2.257e6      # J/kg
    gamma   = 1.33         # polytropic exponent, vapour
    c_l     = 1543.0       # m/s, sound speed in water @100C
    D_heater = 0.025       # m

    # ---------- bubble acoustics ----------
    @classmethod
    def minnaert_coeff(cls):
        """f0 = K / R  with R in metres.  K = sqrt(3*gamma*P0/rho_l)/(2*pi)."""
        return math.sqrt(3 * cls.gamma * cls.P0 / cls.rho_l) / (2 * math.pi)

    @classmethod
    def minnaert_freq(cls, R):
        """Minnaert resonance [Hz] of a spherical gas/vapour bubble, radius R [m]."""
        return cls.minnaert_coeff() / R

    @classmethod
    def minnaert_radius(cls, f):
        """Inverse: bubble radius [m] that resonates at f [Hz]."""
        return cls.minnaert_coeff() / f

    @classmethod
    def fritz_departure_diameter(cls, theta_deg=45.0):
        """Fritz (1935) departure diameter [m]. theta = contact angle [deg]."""
        return (0.0208 * theta_deg *
                math.sqrt(cls.sigma / (cls.g * (cls.rho_l - cls.rho_v))))

    # ---------- hydrodynamic instability ----------
    @classmethod
    def taylor_critical_wavelength(cls):
        """Rayleigh-Taylor critical wavelength [m]."""
        return 2 * math.pi * math.sqrt(
            cls.sigma / (cls.g * (cls.rho_l - cls.rho_v)))

    @classmethod
    def taylor_most_dangerous_wavelength(cls):
        """Most-dangerous (fastest growing) RT wavelength [m] = sqrt(3)*lambda_c."""
        return 2 * math.pi * math.sqrt(
            3 * cls.sigma / (cls.g * (cls.rho_l - cls.rho_v)))

    # ---------- CHF ----------
    @classmethod
    def zuber_chf(cls):
        """Zuber (1959) hydrodynamic CHF correlation [W/m^2]."""
        return (0.131 * cls.rho_v * cls.h_fg *
                ((cls.sigma * cls.g * (cls.rho_l - cls.rho_v))
                 / cls.rho_v ** 2) ** 0.25)

    # ---------- reporting ----------
    @classmethod
    def report(cls):
        K       = cls.minnaert_coeff()
        d_fritz = cls.fritz_departure_diameter()
        R_fritz = d_fritz / 2
        f_fritz = cls.minnaert_freq(R_fritz)
        lam_c   = cls.taylor_critical_wavelength()
        lam_d   = cls.taylor_most_dangerous_wavelength()
        f_lam_d = cls.minnaert_freq(lam_d / 2)
        f_lam_c = cls.minnaert_freq(lam_c / 2)
        f_heat  = cls.minnaert_freq(cls.D_heater / 2)
        q_chf   = cls.zuber_chf()

        R_200 = cls.minnaert_radius(200)
        R_250 = cls.minnaert_radius(250)
        R_400 = cls.minnaert_radius(400)
        R_500 = cls.minnaert_radius(500)

        L = []
        add = L.append
        add("=" * 68)
        add("DERIVED ACOUSTIC PHYSICS -- saturated water, 1 atm, Cu disc")
        add("=" * 68)
        add(f"  Minnaert constant K        : {K:.4f}  (f0 = K/R, R in m)")
        add(f"  Zuber CHF                  : {q_chf/1e6:.3f} MW/m^2")
        add(f"  Fritz departure diameter   : {d_fritz*1e3:.2f} mm")
        add(f"  RT critical wavelength     : {lam_c*1e3:.1f} mm")
        add(f"  RT most-dangerous wavelen. : {lam_d*1e3:.1f} mm")
        add(f"  Heater diameter            : {cls.D_heater*1e3:.1f} mm")
        add("")
        add("  MINNAERT RESONANCE OF CANDIDATE SOURCES")
        add(f"    departure bubble  (D={d_fritz*1e3:5.2f} mm) -> {f_fritz:8.0f} Hz")
        add(f"    RT critical wave  (D={lam_c*1e3:5.1f} mm) -> {f_lam_c:8.0f} Hz")
        add(f"    RT most-dangerous (D={lam_d*1e3:5.1f} mm) -> {f_lam_d:8.0f} Hz")
        add(f"    heater-sized mass (D={cls.D_heater*1e3:5.1f} mm) -> {f_heat:8.0f} Hz")
        add("")
        add("  INVERSE: bubble diameter required for each observed band")
        add(f"    200 Hz -> D = {2*R_200*1e3:6.1f} mm")
        add(f"    250 Hz -> D = {2*R_250*1e3:6.1f} mm")
        add(f"    400 Hz -> D = {2*R_400*1e3:6.1f} mm")
        add(f"    500 Hz -> D = {2*R_500*1e3:6.1f} mm")
        add("")
        add("  >>> CONCLUSION FOR THE MANUSCRIPT <<<")
        add(f"  A {d_fritz*1e3:.2f} mm departure bubble resonates at ~{f_fritz/1000:.1f} kHz,")
        add(f"  which is {f_fritz/225:.0f}x above the observed 200-250 Hz NB band and")
        add(f"  is ALIASED AWAY by the SR=10 kHz / fmax=5 kHz acquisition.")
        add("  Section 6.1 as written ('Minnaert resonance ... results in")
        add("  emission peaks at 200-250 Hz for NB') is therefore NOT")
        add("  supported: reproducing 225 Hz needs a ~29 mm bubble, larger")
        add("  than the 25 mm heater.")
        add("")
        add("  The 200-500 Hz bands ARE consistent with collective vapour")
        add(f"  structures whose size is set by the RT wavelength")
        add(f"  ({lam_d*1e3:.0f} mm -> {f_lam_d:.0f} Hz) and by the heater scale")
        add(f"  ({cls.D_heater*1e3:.0f} mm -> {f_heat:.0f} Hz), i.e. coalesced vapour")
        add("  masses / film patches, not individual departing bubbles.")
        add("  This is a STRONGER and defensible physical claim.")
        add("=" * 68)
        return "\n".join(L)

    @classmethod
    def to_dict(cls):
        d_fritz = cls.fritz_departure_diameter()
        lam_d   = cls.taylor_most_dangerous_wavelength()
        return {
            'minnaert_K': cls.minnaert_coeff(),
            'zuber_chf_W_m2': cls.zuber_chf(),
            'fritz_departure_diameter_m': d_fritz,
            'f_departure_bubble_Hz': cls.minnaert_freq(d_fritz / 2),
            'RT_critical_wavelength_m': cls.taylor_critical_wavelength(),
            'RT_most_dangerous_wavelength_m': lam_d,
            'f_RT_mass_Hz': cls.minnaert_freq(lam_d / 2),
            'f_heater_scale_mass_Hz': cls.minnaert_freq(cls.D_heater / 2),
            'D_for_200Hz_m': 2 * cls.minnaert_radius(200),
            'D_for_500Hz_m': 2 * cls.minnaert_radius(500),
        }


def step_physics():
    print(BoilingPhysics.report())
    out = os.path.join(RESULTS_DIR, 'physics_derived.json')
    with open(out, 'w') as f:
        json.dump(BoilingPhysics.to_dict(), f, indent=2)
    txt = os.path.join(RESULTS_DIR, 'physics_report.txt')
    with open(txt, 'w', encoding='utf-8') as f:
        f.write(BoilingPhysics.report())
    print(f"\n  Saved: {out}\n  Saved: {txt}")
    return BoilingPhysics.to_dict()


# ============================================================
#  2. FILENAME INSPECTION   (RUN THIS FIRST)
# ============================================================
def step_inspect():
    """
    Print enough about the raw filenames to decide how sessions can
    be recovered. Nothing else in this file is trustworthy until you
    have read this output.
    """
    print("=" * 68)
    print("RAW FILENAME INSPECTION")
    print("=" * 68)

    if not os.path.isdir(DATA_DIR):
        print(f"  ERROR: {DATA_DIR} not found.")
        return None

    summary = {}
    for cls in FOLDERS:
        d = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(d):
            print(f"\n  {cls}: MISSING")
            continue
        wavs = sorted(f for f in os.listdir(d) if f.lower().endswith('.wav'))
        print(f"\n  {cls}: {len(wavs)} files")
        print("    first 5 :", wavs[:5])
        print("    last  5 :", wavs[-5:])

        # token/regex analysis
        nums = [re.findall(r'\d+', w) for w in wavs]
        ncnt = Counter(len(n) for n in nums)
        print(f"    numeric fields per name: {dict(ncnt)}")

        # Are the trailing numbers a dense 0..N-1 / 1..N run?
        contiguous = None
        if nums and all(len(n) >= 1 for n in nums):
            try:
                last = sorted(int(n[-1]) for n in nums)
                span = last[-1] - last[0] + 1
                contiguous = (span == len(last) and
                              len(set(last)) == len(last))
                gaps = [b - a for a, b in zip(last, last[1:]) if b - a > 1]
                print(f"    last-number range: {last[0]}..{last[-1]}  "
                      f"dense={contiguous}  gaps={len(gaps)}")
                if gaps:
                    print(f"      gap sizes (first 10): {gaps[:10]}")
                    print("      ^ gaps often mark POWER-STEP boundaries")
            except ValueError:
                pass

        stems = {re.sub(r'\d+', '#', w) for w in wavs}
        print(f"    distinct name templates: {len(stems)} -> "
              f"{list(stems)[:4]}")

        summary[cls] = {'n': len(wavs), 'templates': sorted(stems)[:8],
                        'dense_index': contiguous,
                        'first': wavs[:3], 'last': wavs[-3:]}

    print("\n" + "=" * 68)
    print("HOW TO READ THIS")
    print("=" * 68)
    print("""
  CASE A -- names contain a power/run token (e.g. 'P12', 'run3', '45W')
      -> best case. Use --group_mode filename and set --group_regex.

  CASE B -- a single dense numeric index per class (e.g. NB_0001..NB_9356)
      -> index == elapsed second. Sessions cannot be separated by name,
         but ORDER is preserved: use --group_mode blocks (contiguous
         blocks + guard band). This alone removes most leakage.

  CASE C -- dense index with visible GAPS
      -> gaps are almost certainly power-step boundaries. Use
         --group_mode gaps (each contiguous run = one session).

  CASE D -- hashes / random names, no order
      -> ordering is unrecoverable from names.
         Use --group_mode acoustic (change-point detection on slow
         acoustic drift). Weaker, but far better than random, and it
         must be declared as a limitation in the paper.
""")
    out = os.path.join(RESULTS_DIR, 'filename_inspection.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out}")
    return summary


# ============================================================
#  3. GROUP DISCOVERY
# ============================================================
def _sort_key(fname):
    """Natural sort: trailing integer if present, else lexicographic."""
    nums = re.findall(r'\d+', fname)
    return (int(nums[-1]) if nums else 0, fname)


def _groups_from_filename(wavs, regex):
    rx = re.compile(regex)
    out, unmatched = [], 0
    for w in wavs:
        m = rx.search(w)
        if m:
            out.append(m.group(1) if m.groups() else m.group(0))
        else:
            out.append('UNMATCHED')
            unmatched += 1
    if unmatched:
        print(f"      WARNING: {unmatched}/{len(wavs)} names did not "
              f"match {regex!r}")
    return out


def _groups_from_gaps(wavs):
    """Each maximal run of consecutive integer indices = one session."""
    idx = []
    for w in wavs:
        n = re.findall(r'\d+', w)
        idx.append(int(n[-1]) if n else None)
    if any(i is None for i in idx):
        return None
    gid, groups = 0, []
    for k, i in enumerate(idx):
        if k > 0 and i - idx[k - 1] > 1:
            gid += 1
        groups.append(f"g{gid:03d}")
    return groups


def _groups_from_blocks(wavs, block):
    """Contiguous fixed-size blocks of the ordered file list."""
    return [f"b{k // block:03d}" for k in range(len(wavs))]


def _groups_from_acoustic(paths, cls, min_len=60, z=3.0):
    """
    Last resort: detect session boundaries from slow drift in cheap
    acoustic descriptors (RMS, spectral centroid, low-band ratio).
    A boundary is declared where the frame-to-frame change in the
    smoothed feature vector exceeds z robust MADs.
    """
    try:
        import librosa
    except ImportError:
        print("      ERROR: librosa required for --group_mode acoustic")
        return None

    feats = []
    for k, p in enumerate(paths):
        try:
            y, _ = librosa.load(p, sr=SR, mono=True)
            S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
            f = np.fft.rfftfreq(len(y), 1 / SR)
            tot = S.sum() + 1e-12
            feats.append([
                np.sqrt(np.mean(y ** 2)),          # RMS
                float((S * f).sum() / tot),        # spectral centroid
                float(S[f < 100].sum() / tot),     # <100 Hz ratio
                float(S[(f >= 200) & (f < 500)].sum() / tot),
            ])
        except Exception:
            feats.append(feats[-1] if feats else [0, 0, 0, 0])
        if (k + 1) % 500 == 0:
            print(f"      {cls}: {k+1}/{len(paths)} scanned", end='\r')

    F = np.asarray(feats, dtype=float)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    # smooth
    w = 15
    ker = np.ones(w) / w
    Fs = np.vstack([np.convolve(F[:, j], ker, mode='same')
                    for j in range(F.shape[1])]).T
    d = np.linalg.norm(np.diff(Fs, axis=0), axis=1)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-9
    thr = med + z * 1.4826 * mad

    gid, groups, last = 0, ['g000'], 0
    for k in range(1, len(paths)):
        if d[k - 1] > thr and (k - last) >= min_len:
            gid += 1
            last = k
        groups.append(f"g{gid:03d}")
    print(f"      {cls}: {gid+1} sessions detected            ")
    return groups


def discover_groups(mode='blocks', regex=None, block=120):
    """
    Returns {class: [(filename, group_id), ...]} with files in
    acquisition order.
    """
    print("=" * 68)
    print(f"GROUP DISCOVERY  (mode={mode})")
    print("=" * 68)

    mapping = {}
    for cls in FOLDERS:
        d = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(d):
            print(f"  {cls}: MISSING - skipped")
            continue
        wavs = sorted((f for f in os.listdir(d)
                       if f.lower().endswith('.wav')), key=_sort_key)
        if not wavs:
            continue

        if mode == 'filename':
            if not regex:
                raise ValueError("--group_regex required for mode=filename")
            g = _groups_from_filename(wavs, regex)
        elif mode == 'gaps':
            g = _groups_from_gaps(wavs)
            if g is None:
                print(f"  {cls}: no numeric index -> falling back to blocks")
                g = _groups_from_blocks(wavs, block)
        elif mode == 'acoustic':
            g = _groups_from_acoustic(
                [os.path.join(d, w) for w in wavs], cls)
            if g is None:
                g = _groups_from_blocks(wavs, block)
        else:  # blocks
            g = _groups_from_blocks(wavs, block)

        g = [f"{cls}::{x}" for x in g]
        mapping[cls] = list(zip(wavs, g))
        sizes = Counter(g)
        print(f"  {cls:<8}: {len(wavs):>6} clips -> {len(sizes)} sessions "
              f"(min={min(sizes.values())}, max={max(sizes.values())})")

    out = os.path.join(RESULTS_DIR, 'groups.json')
    with open(out, 'w') as f:
        json.dump({c: {w: g for w, g in v} for c, v in mapping.items()},
                  f, indent=2)
    print(f"\n  Saved: {out}")
    return mapping


def load_groups():
    p = os.path.join(RESULTS_DIR, 'groups.json')
    if not os.path.exists(p):
        print(f"  ERROR: {p} missing -- run --steps groups first")
        return None
    with open(p) as f:
        raw = json.load(f)
    return {c: sorted(v.items(), key=lambda kv: _sort_key(kv[0]))
            for c, v in raw.items()}


# ============================================================
#  4. GROUPED SPLIT  (the actual fix)
# ============================================================
def grouped_split(train=0.70, val=0.15, test=0.15,
                  guard=GUARD_CLIPS, seed=SEED):
    """
    Assign whole SESSIONS to train/val/test.  No clip from a session
    ever appears in two partitions.  Additionally drop `guard` clips
    at each end of every session so that clips adjacent across a
    boundary are never split between partitions.
    """
    print("=" * 68)
    print("GROUPED SPLIT  (session-level, with guard band)")
    print(f"  ratio {train:.0%}/{val:.0%}/{test:.0%} | guard={guard} clips")
    print("=" * 68)

    mapping = load_groups()
    if mapping is None:
        return None

    rng = random.Random(seed)
    assign, stats = {}, {}

    for cls, pairs in mapping.items():
        by_g = defaultdict(list)
        for w, g in pairs:
            by_g[g].append(w)
        gids = sorted(by_g)

        subdivided = False
        if len(gids) < 3:
            subdivided = True
            print(f"\n  {'!'*62}")
            print(f"  {cls}: only {len(gids)} recording session(s) exist.")
            print(f"  A fully session-disjoint 3-way split is IMPOSSIBLE")
            print(f"  for this class. Falling back to contiguous")
            print(f"  pseudo-sessions with an ENLARGED guard band.")
            print(f"  ")
            print(f"  >>> This is a HARD LIMIT OF THE DATASET, not of the")
            print(f"      method. Results for {cls} remain OPTIMISTIC")
            print(f"      because train/test clips come from the same")
            print(f"      continuous recording. STATE THIS IN THE PAPER.")
            print(f"  {'!'*62}")
            flat = [w for g in gids for w in by_g[g]]
            # Fewer, larger pseudo-sessions => fewer boundaries => less
            # residual correlation. Enlarged guard applied below.
            n_sub = max(3, min(8, len(flat) // 60))
            size  = max(1, len(flat) // n_sub)
            by_g  = {f"{cls}::sub{i:02d}": flat[i*size:(i+1)*size]
                     for i in range(n_sub)}
            leftover = flat[n_sub*size:]
            if leftover:
                by_g[f"{cls}::sub{n_sub-1:02d}"].extend(leftover)
            gids = sorted(by_g)

        # Greedy size-balanced assignment over shuffled sessions
        order = gids[:]
        rng.shuffle(order)
        total = sum(len(by_g[g]) for g in gids)
        want  = {'train': train*total, 'validation': val*total,
                 'test': test*total}
        got   = {'train': 0, 'validation': 0, 'test': 0}
        chosen = {'train': [], 'validation': [], 'test': []}
        for g in sorted(order, key=lambda x: -len(by_g[x])):
            k = max(want, key=lambda s: want[s] - got[s])
            chosen[k].append(g)
            got[k] += len(by_g[g])

        # ensure no partition is empty
        for s in ('train', 'validation', 'test'):
            if not chosen[s]:
                donor = max(chosen, key=lambda k: len(chosen[k]))
                if len(chosen[donor]) > 1:
                    chosen[s].append(chosen[donor].pop())

        # Pseudo-sessions are contiguous slices of ONE recording, so
        # neighbouring clips across a boundary are highly correlated.
        # Use a much larger guard band there.
        g_eff = guard * 6 if subdivided else guard

        stats[cls] = {}
        for s in ('train', 'validation', 'test'):
            files = []
            for g in chosen[s]:
                clips = by_g[g]
                clips = clips[g_eff:-g_eff] if len(clips) > 2*g_eff + 1 \
                        else clips
                files.extend(clips)
            for w in files:
                assign[(cls, w)] = s
            stats[cls][s] = {'sessions': len(chosen[s]),
                             'clips': len(files)}
        stats[cls]['_subdivided'] = subdivided
        stats[cls]['_guard_used'] = g_eff
        stats[cls]['_true_sessions'] = len(set(g for _, g in pairs))

        print(f"\n  {cls}"
              f"{'  [PSEUDO-SESSIONS - OPTIMISTIC]' if subdivided else ''}")
        for s in ('train', 'validation', 'test'):
            d = stats[cls][s]
            print(f"    {s:<11}: {d['sessions']:>3} sessions, "
                  f"{d['clips']:>6} clips")
        print(f"    guard band : {g_eff} clips/session end")

    out = os.path.join(RESULTS_DIR, 'grouped_split_assignment.json')
    with open(out, 'w') as f:
        json.dump({f"{c}/{w}": s for (c, w), s in assign.items()},
                  f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'grouped_split_stats.json'), 'w') as f:
        json.dump({'ratios': {'train': train, 'val': val, 'test': test},
                   'guard_clips': guard, 'seed': seed, 'stats': stats},
                  f, indent=2)

    print(f"\n  Saved: {out}")

    weak = [c for c, d in stats.items() if d.get('_subdivided')]
    print("\n" + "=" * 68)
    print("  WHAT YOU CAN AND CANNOT CLAIM")
    print("=" * 68)
    clean = [c for c in stats if c not in weak]
    if clean:
        print(f"  FULLY SESSION-DISJOINT (trustworthy): {', '.join(clean)}")
    if weak:
        print(f"  PSEUDO-SESSIONS ONLY  (still optimistic): "
              f"{', '.join(weak)}")
        for c in weak:
            print(f"     {c}: {stats[c]['_true_sessions']} true "
                  f"recording session(s) in the whole dataset")
        print("\n  For these classes train and test clips originate from")
        print("  the SAME continuous recording. No split can fix that -")
        print("  only new experimental data can. The manuscript must say:")
        print("")
        print('    \"For the Pre-CHF regime the dataset contains too few')
        print('     independent recording sessions to permit a fully')
        print('     session-disjoint partition. Reported Pre-CHF metrics')
        print('     are therefore an upper bound, and quantify')
        print('     within-session discriminability rather than')
        print('     across-run generalisation.\"')
    print("=" * 68)
    print("  Quote THESE counts in Table 2 and state that partitioning")
    print("  was performed at session level, not clip level.")
    return assign


def load_assignment():
    p = os.path.join(RESULTS_DIR, 'grouped_split_assignment.json')
    if not os.path.exists(p):
        print(f"  ERROR: {p} missing -- run --steps split first")
        return None
    with open(p) as f:
        raw = json.load(f)
    return {(k.split('/', 1)[0], k.split('/', 1)[1]): v
            for k, v in raw.items()}


# ============================================================
#  5. BAND-LIMITED SPECTROGRAM GENERATION
# ============================================================
def generate_spectrograms(spec_types=None, fmax=FMAX_ANALYSIS):
    """WAV -> band-limited spectrogram PNGs, written straight into the
    session-based split tree."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import librosa, librosa.display
    except ImportError:
        print("  ERROR: pip install librosa")
        return None

    spec_types = spec_types or SPEC_TYPES
    assign = load_assignment()
    if assign is None:
        return None

    print("=" * 68)
    print(f"SPECTROGRAM GENERATION  (fmax={fmax} Hz)")
    n_rows_band = {c: (hi - lo) / fmax * IMG_SIZE[0]
                   for c, (lo, hi) in PHYSICS_BANDS_HZ.items()}
    print("  image rows spanned by each physics band:")
    for c, r in n_rows_band.items():
        print(f"    {c:<8} {PHYSICS_BANDS_HZ[c][0]:>4}-"
              f"{PHYSICS_BANDS_HZ[c][1]:<4} Hz -> {r:5.1f} / {IMG_SIZE[0]} rows")
    print(f"  (at fmax=5000 Hz these would be {fmax/5000:.2f}x smaller)")
    print("=" * 68)

    def _stft(y):
        return librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN)),
            ref=np.max)

    def _mel(y):
        return librosa.power_to_db(librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN,
            n_mels=N_MELS, fmax=fmax), ref=np.max)

    def _mfcc(y):
        return librosa.feature.mfcc(
            y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT,
            hop_length=HOP_LEN, fmax=fmax)

    def _cqt(y):
        return librosa.amplitude_to_db(np.abs(librosa.cqt(
            y, sr=SR, hop_length=HOP_LEN, n_bins=N_CQT,
            bins_per_octave=12, fmin=librosa.note_to_hz('C1'))),
            ref=np.max)

    def _chroma(y):
        return librosa.feature.chroma_stft(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN)

    GEN = {'STFT': _stft, 'MEL': _mel, 'MFCC': _mfcc,
           'CQT': _cqt, 'CHROMA': _chroma}

    def save(data, st, path):
        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        kw = dict(sr=SR, hop_length=HOP_LEN, x_axis='time',
                  cmap=SPEC_CMAP[st], ax=ax)
        if st == 'STFT':
            librosa.display.specshow(data, y_axis='hz', **kw)
            ax.set_ylim(0, fmax)          # <-- the band limit
        elif st == 'MEL':
            librosa.display.specshow(data, y_axis='mel', fmax=fmax, **kw)
        elif st == 'CQT':
            librosa.display.specshow(data, y_axis='cqt_hz', **kw)
            ax.set_ylim(0, fmax)
        elif st == 'CHROMA':
            librosa.display.specshow(data, y_axis='chroma', **kw)
        else:
            librosa.display.specshow(data, **kw)
        ax.axis('off')
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    total = errs = 0
    for cls in FOLDERS:
        src = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(src):
            continue
        wavs = [w for w in sorted(os.listdir(src), key=_sort_key)
                if w.lower().endswith('.wav') and (cls, w) in assign]
        print(f"\n  {cls}: {len(wavs)} assigned clips")
        for i, wf in enumerate(wavs):
            split = assign[(cls, wf)]
            try:
                y, _ = librosa.load(os.path.join(src, wf), sr=SR, mono=True)
            except Exception:
                errs += 1
                continue
            for st in spec_types:
                out_d = os.path.join(SPLIT_ORIG, st, split, cls)
                os.makedirs(out_d, exist_ok=True)
                out_p = os.path.join(out_d, Path(wf).stem + '.png')
                if os.path.exists(out_p):
                    continue
                try:
                    save(GEN[st](y), st, out_p)
                    total += 1
                except Exception:
                    errs += 1
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(wavs)}  err={errs}", end='\r')
        print(f"    done ({len(wavs)} clips)                      ")

    print(f"\n  PNGs written: {total} | errors: {errs}")
    return total


# ============================================================
#  6. AUGMENTATION  (train only) + LEAKAGE CHECK
# ============================================================
def augment_image(img, mode='full'):
    from PIL import Image, ImageEnhance
    if mode == 'none':
        return img
    arr = np.array(img).astype(np.float32)
    choice = (random.randint(0, 1) if mode == 'mild'
              else random.choice([0, 1, 3, 4, 5]) if mode == 'no_freqmask'
              else random.choice([0, 1, 2, 3, 5]) if mode == 'no_noise'
              else random.randint(0, 5))
    if choice == 0:
        return ImageEnhance.Brightness(img).enhance(random.uniform(.75, 1.25))
    if choice == 5:
        return ImageEnhance.Contrast(img).enhance(random.uniform(.8, 1.2))
    if choice == 1:
        arr = np.roll(arr, random.randint(-15, 15), axis=1)
    elif choice == 2:
        h = arr.shape[0]; mh = random.randint(5, 15)
        s = random.randint(0, max(0, h - mh)); arr[s:s+mh, :] = arr.mean()
    elif choice == 3:
        w = arr.shape[1]; mw = random.randint(5, 15)
        s = random.randint(0, max(0, w - mw)); arr[:, s:s+mw] = arr.mean()
    elif choice == 4:
        arr = np.clip(arr + np.random.normal(0, 8, arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def augment_train_only(target_n=TARGET_N, spec_types=None):
    from PIL import Image
    spec_types = spec_types or SPEC_TYPES
    random.seed(SEED); np.random.seed(SEED)

    print("=" * 68)
    print(f"AUGMENT TRAIN ONLY  (target {target_n}/class)")
    print("=" * 68)

    for st in spec_types:
        print(f"\n  {st}")
        for split in ('train', 'validation', 'test'):
            for cls in FOLDERS:
                src = os.path.join(SPLIT_ORIG, st, split, cls)
                dst = os.path.join(SPLIT_FINAL, st, split, cls)
                os.makedirs(dst, exist_ok=True)
                if not os.path.isdir(src):
                    continue
                orig = sorted(f for f in os.listdir(src)
                              if f.endswith('.png'))
                for f in orig:
                    p = os.path.join(dst, f)
                    if not os.path.exists(p):
                        shutil.copy2(os.path.join(src, f), p)
                if split != 'train':
                    continue
                have = len([f for f in os.listdir(dst)
                            if f.endswith('.png')])
                if have >= target_n or not orig:
                    print(f"    train/{cls:<8}: {have} (no aug needed)")
                    continue
                k = have
                while k < target_n:
                    sf = random.choice(orig)
                    dp = os.path.join(dst, f"aug{k:06d}_{sf}")
                    if os.path.exists(dp):
                        k += 1; continue
                    try:
                        augment_image(
                            Image.open(os.path.join(src, sf)).convert('RGB'),
                            'full').save(dp)
                        k += 1
                    except Exception:
                        break
                print(f"    train/{cls:<8}: {len(orig)} -> {k}")


def check_integrity():
    """Verify no augmented file and no session crosses partitions."""
    print("=" * 68)
    print("INTEGRITY CHECK")
    print("=" * 68)
    ok = True
    groups = load_groups()
    gmap = {}
    if groups:
        for cls, pairs in groups.items():
            for w, g in pairs:
                gmap[(cls, Path(w).stem)] = g

    for st in SPEC_TYPES:
        if not os.path.isdir(os.path.join(SPLIT_FINAL, st)):
            continue
        seen = defaultdict(set)
        print(f"\n  {st}")
        for split in ('train', 'validation', 'test'):
            tot = 0
            for cls in FOLDERS:
                p = os.path.join(SPLIT_FINAL, st, split, cls)
                if not os.path.isdir(p):
                    continue
                fs = [f for f in os.listdir(p) if f.endswith('.png')]
                tot += len(fs)
                if split != 'train':
                    aug = [f for f in fs if f.startswith('aug')]
                    if aug:
                        print(f"    LEAK: {len(aug)} aug files in "
                              f"{split}/{cls}")
                        ok = False
                for f in fs:
                    if f.startswith('aug'):
                        continue
                    g = gmap.get((cls, Path(f).stem))
                    if g:
                        seen[g].add(split)
            print(f"    {split:<11}: {tot}")
        cross = {g: s for g, s in seen.items() if len(s) > 1}
        if cross:
            print(f"    LEAK: {len(cross)} session(s) span partitions:")
            for g, s in list(cross.items())[:5]:
                print(f"       {g} -> {sorted(s)}")
            ok = False
        else:
            print("    session disjointness: OK")

    print(f"\n  RESULT: {'PASSED' if ok else 'FAILED'}")
    return ok


# ============================================================
#  7. GROUPED K-FOLD  (originals only, streamed)
# ============================================================
def grouped_kfold(spec_type='STFT', n_splits=5, epochs=30,
                  model_builder=None):
    import tensorflow as tf
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import classification_report
    from sklearn.utils.class_weight import compute_class_weight

    print("=" * 68)
    print(f"GROUPED K-FOLD CV  ({n_splits} folds, originals only)")
    print("=" * 68)

    groups = load_groups()
    if groups is None:
        return None
    gmap = {(c, Path(w).stem): g for c, pr in groups.items() for w, g in pr}

    paths, labels, gids = [], [], []
    for split in ('train', 'validation'):
        for ci, cls in enumerate(CLASS_NAMES):
            d = os.path.join(SPLIT_ORIG, spec_type, split, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if not f.endswith('.png') or f.startswith('aug'):
                    continue
                g = gmap.get((cls, Path(f).stem))
                if g is None:
                    continue
                paths.append(os.path.join(d, f))
                labels.append(ci)
                gids.append(g)

    paths = np.array(paths); labels = np.array(labels); gids = np.array(gids)
    if len(paths) == 0:
        print("  ERROR: no original images found")
        return None
    print(f"  {len(paths)} clips | {len(set(gids))} sessions")

    if model_builder is None:
        from tensorflow.keras import layers, models
        def model_builder(input_shape=IMG_SIZE + (3,), num_classes=4):
            return models.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv2D(32, 3, activation='relu', padding='same'),
                layers.BatchNormalization(), layers.MaxPooling2D(2),
                layers.Dropout(.25),
                layers.Conv2D(64, 3, activation='relu', padding='same'),
                layers.BatchNormalization(), layers.MaxPooling2D(2),
                layers.Dropout(.25),
                layers.Conv2D(128, 3, activation='relu', padding='same'),
                layers.BatchNormalization(), layers.MaxPooling2D(2),
                layers.Dropout(.3),
                layers.Conv2D(256, 3, activation='relu', padding='same'),
                layers.BatchNormalization(), layers.MaxPooling2D(2),
                layers.Dropout(.3),
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation='relu'), layers.Dropout(.5),
                layers.Dense(num_classes, activation='softmax')],
                name='Original_CNN')

    def ds(p, y, shuffle, aug=False):
        d = tf.data.Dataset.from_tensor_slices(
            (tf.constant(list(p)), tf.constant(y.astype(np.int32))))
        if shuffle:
            d = d.shuffle(len(p), seed=SEED)
        def load(path, lab):
            im = tf.io.decode_png(tf.io.read_file(path), channels=3)
            im = tf.cast(tf.image.resize(im, IMG_SIZE), tf.float32) / 255.
            if aug:
                im = tf.clip_by_value(
                    tf.image.random_contrast(
                        tf.image.random_brightness(im, .1), .9, 1.1), 0., 1.)
            return im, tf.one_hot(lab, NUM_CLASSES)
        return (d.map(load, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    accs, f1s, folds = [], [], []
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, va) in enumerate(gkf.split(paths, labels, groups=gids)):
        n_tr_g = len(set(gids[tr])); n_va_g = len(set(gids[va]))
        assert not (set(gids[tr]) & set(gids[va])), "group leak!"
        print(f"\n  Fold {k+1}/{n_splits}: train {len(tr)} clips "
              f"({n_tr_g} sess) | val {len(va)} clips ({n_va_g} sess)")
        print(f"    val class counts: "
              f"{dict(zip(*np.unique(labels[va], return_counts=True)))}")

        tf.random.set_seed(k); np.random.seed(k); random.seed(k)
        m = model_builder(input_shape=IMG_SIZE + (3,),
                          num_classes=NUM_CLASSES)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])

        present = np.unique(labels[tr])
        cw_arr = compute_class_weight('balanced', classes=present,
                                      y=labels[tr])
        cw = {int(c): float(w) for c, w in zip(present, cw_arr)}
        for c in range(NUM_CLASSES):
            cw.setdefault(c, 1.0)

        m.fit(ds(paths[tr], labels[tr], True, True),
              validation_data=ds(paths[va], labels[va], False),
              epochs=epochs, class_weight=cw, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor='val_accuracy', patience=8,
                  restore_best_weights=True, verbose=0)])

        yp = np.argmax(m.predict(ds(paths[va], labels[va], False),
                                 verbose=0), axis=1)
        yt = labels[va]
        acc = float(np.mean(yp == yt) * 100)
        rep = classification_report(yt, yp,
                                    labels=list(range(NUM_CLASSES)),
                                    target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        f1 = rep.get('Pre-CHF', {}).get('f1-score', 0.0)
        accs.append(acc); f1s.append(float(f1))
        folds.append({'fold': k + 1, 'accuracy': round(acc, 3),
                      'prechf_f1': round(float(f1), 4),
                      'train_sessions': n_tr_g, 'val_sessions': n_va_g,
                      'macro_f1': round(rep['macro avg']['f1-score'], 4)})
        print(f"    acc={acc:.2f}%  Pre-CHF F1={f1:.4f}  "
              f"macro F1={rep['macro avg']['f1-score']:.4f}")
        del m; tf.keras.backend.clear_session()

    out = {'spec_type': spec_type, 'n_splits': n_splits,
           'protocol': 'GroupKFold by recording session, originals only',
           'accuracy': {'mean': float(np.mean(accs)),
                        'std': float(np.std(accs, ddof=1))
                        if len(accs) > 1 else 0.0},
           'prechf_f1': {'mean': float(np.mean(f1s)),
                         'std': float(np.std(f1s, ddof=1))
                         if len(f1s) > 1 else 0.0},
           'per_fold': folds}
    print(f"\n  Accuracy   : {out['accuracy']['mean']:.2f}% "
          f"± {out['accuracy']['std']:.2f}%")
    print(f"  Pre-CHF F1 : {out['prechf_f1']['mean']:.4f} "
          f"± {out['prechf_f1']['std']:.4f}")
    print("\n  >>> THIS is the number to publish. If it is much lower than")
    print("      99.75%, the original result was session memorisation.")

    p = os.path.join(RESULTS_DIR, 'grouped_kfold.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {p}")
    return out


# ============================================================
#  8. TRUE WAVEFORM-DOMAIN NOISE ROBUSTNESS
# ============================================================
def add_noise_wav(y, snr_db, rng=None):
    """
    Additive white Gaussian noise at a genuine SNR.
        SNR_dB = 10*log10(P_signal / P_noise)
    Higher SNR_dB  ==>  LESS noise.  (The original code had this
    inverted and applied it to pixels.)
    """
    rng = rng or np.random
    ps = float(np.mean(y.astype(np.float64) ** 2))
    if ps <= 0:
        return y
    pn = ps / (10 ** (snr_db / 10.0))
    return (y + rng.normal(0, math.sqrt(pn), y.shape)).astype(np.float32)


def noise_robustness_wav(model_path, spec_type='STFT',
                         snr_levels=(30, 20, 10, 0),
                         fmax=FMAX_ANALYSIS, max_per_class=None):
    """
    Add noise to the WAVEFORM, regenerate the spectrogram, then
    classify. This is what "70 dB noise" should have meant.
    """
    import tensorflow as tf
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    try:
        import librosa, librosa.display
    except ImportError:
        print("  ERROR: pip install librosa"); return None

    print("=" * 68)
    print(f"NOISE ROBUSTNESS -- waveform domain, true SNR")
    print(f"  levels: {list(snr_levels)} dB  (higher = cleaner)")
    print("=" * 68)

    model = tf.keras.models.load_model(model_path, compile=False)
    assign = load_assignment()
    if assign is None:
        return None
    test = defaultdict(list)
    for (cls, w), s in assign.items():
        if s == 'test':
            test[cls].append(w)

    import tempfile
    tmp = tempfile.mkdtemp()

    def spec_png(y, path):
        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN)),
            ref=np.max)
        librosa.display.specshow(D, sr=SR, hop_length=HOP_LEN,
                                 x_axis='time', y_axis='hz',
                                 cmap=SPEC_CMAP[spec_type], ax=ax)
        ax.set_ylim(0, fmax); ax.axis('off')
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    from PIL import Image
    results = {}
    for snr in ('clean',) + tuple(snr_levels):
        rng = np.random.RandomState(0 if snr == 'clean' else int(snr))
        yt, yp = [], []
        for ci, cls in enumerate(CLASS_NAMES):
            wavs = sorted(test.get(cls, []), key=_sort_key)
            if max_per_class:
                wavs = wavs[:max_per_class]
            batch = []
            for w in wavs:
                try:
                    y, _ = librosa.load(
                        os.path.join(DATA_DIR, cls, w), sr=SR, mono=True)
                except Exception:
                    continue
                if snr != 'clean':
                    y = add_noise_wav(y, float(snr), rng)
                p = os.path.join(tmp, 'x.png')
                spec_png(y, p)
                batch.append(np.array(Image.open(p).convert('RGB')
                                      .resize(IMG_SIZE),
                                      dtype=np.float32) / 255.)
                if len(batch) == BATCH_SIZE:
                    yp.extend(np.argmax(model.predict(np.stack(batch),
                                                      verbose=0), axis=1))
                    yt.extend([ci] * len(batch)); batch = []
            if batch:
                yp.extend(np.argmax(model.predict(np.stack(batch),
                                                  verbose=0), axis=1))
                yt.extend([ci] * len(batch))

        yt = np.array(yt); yp = np.array(yp)
        acc = float(np.mean(yt == yp) * 100)
        rep = classification_report(yt, yp, labels=list(range(NUM_CLASSES)),
                                    target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        results[str(snr)] = {
            'overall_acc': round(acc, 2),
            'macro_f1': round(rep['macro avg']['f1-score'], 4),
            'per_class_recall': {c: round(rep[c]['recall'], 4)
                                 for c in CLASS_NAMES}}
        print(f"  SNR={str(snr):>5} dB -> acc={acc:6.2f}%  "
              f"macroF1={rep['macro avg']['f1-score']:.4f}  "
              f"PreCHF recall={rep['Pre-CHF']['recall']:.3f}")

    shutil.rmtree(tmp, ignore_errors=True)
    p = os.path.join(RESULTS_DIR, 'noise_robustness_waveform.json')
    with open(p, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {p}")
    return results


# ============================================================
#  9. HONEST SMALL-SAMPLE REPORTING
# ============================================================
def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion -- correct for small n,
    unlike the normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def gradcam_resolution(n_pool=4, fmax=FMAX_ANALYSIS, img_h=IMG_SIZE[0]):
    """
    How many Hz does one Grad-CAM cell cover? Decides whether the
    'attention localises to 200-250 Hz' claim is even possible.
    """
    rows = img_h // (2 ** n_pool)
    hz = fmax / rows
    print("=" * 68)
    print("GRAD-CAM FREQUENCY RESOLUTION AUDIT")
    print("=" * 68)
    print(f"  image height        : {img_h} px over 0-{fmax} Hz")
    print(f"  pooling layers      : {n_pool}  -> feature map {rows} rows")
    print(f"  Hz per Grad-CAM row : {hz:.0f} Hz")
    print("\n  band -> Grad-CAM rows")
    collide = defaultdict(list)
    for c, (lo, hi) in PHYSICS_BANDS_HZ.items():
        r0, r1 = int(lo / hz), int(min(hi, fmax) / hz)
        print(f"    {c:<8} {lo:>4}-{hi:<4} Hz -> rows {r0}-{r1}")
        for r in range(r0, r1 + 1):
            collide[r].append(c)
    bad = {r: cs for r, cs in collide.items() if len(cs) > 1}
    if bad:
        print("\n  WARNING: these rows are shared by several regimes:")
        for r, cs in sorted(bad.items()):
            print(f"    row {r}: {', '.join(cs)}")
        print("  -> Grad-CAM CANNOT distinguish these bands at this")
        print("     resolution. Do not claim it does.")
    else:
        print("\n  All physics bands land in distinct Grad-CAM rows.")
    print(f"\n  At fmax=5000 Hz the resolution would be "
          f"{5000/rows:.0f} Hz/row (all bands collapse to ~1 row).")
    return {'rows': rows, 'hz_per_row': hz,
            'colliding_rows': {str(k): v for k, v in bad.items()}}


def report_honest(results_json=None):
    """Print the numbers the paper should quote, with Wilson CIs."""
    print("=" * 68)
    print("HONEST REPORTING SUMMARY")
    print("=" * 68)

    st = os.path.join(RESULTS_DIR, 'grouped_split_stats.json')
    if os.path.exists(st):
        with open(st) as f:
            s = json.load(f)
        print("\n  Session-level split (quote in Table 2):")
        for cls, d in s['stats'].items():
            print(f"    {cls:<8} " + " | ".join(
                f"{k}: {v['clips']:>5} clips / {v['sessions']} sess"
                for k, v in d.items()))
        n_pc = s['stats'].get('Pre-CHF', {}).get('test', {}).get('clips', 0)
        if n_pc:
            print(f"\n  Pre-CHF test clips: n = {n_pc}")
            for r in (0.80, 0.90, 0.95):
                k = int(round(r * n_pc))
                lo, hi = wilson_ci(k, n_pc)
                print(f"    recall {r:.2f} -> 95% Wilson CI "
                      f"[{lo:.3f}, {hi:.3f}]  (width {hi-lo:.3f})")
            print("  -> ALWAYS report this interval next to any Pre-CHF F1.")

    kf = os.path.join(RESULTS_DIR, 'grouped_kfold.json')
    if os.path.exists(kf):
        with open(kf) as f:
            d = json.load(f)
        print(f"\n  Grouped K-fold (headline result):")
        print(f"    accuracy   {d['accuracy']['mean']:.2f}% "
              f"± {d['accuracy']['std']:.2f}%")
        print(f"    Pre-CHF F1 {d['prechf_f1']['mean']:.4f} "
              f"± {d['prechf_f1']['std']:.4f}")

    print("\n  Grad-CAM audit:")
    gradcam_resolution()

    print("""
  MANUSCRIPT EDITS IMPLIED BY THIS RUN
  ------------------------------------
   1. Sec 4.3: state partitioning is at SESSION level; report guard band.
   2. Sec 5/6 : replace all clip-split numbers with grouped results.
   3. Sec 6.1 : replace the Minnaert-for-NB claim (see physics report).
   4. Sec 5.9 : relabel noise axis as true waveform SNR (higher = cleaner).
   5. Sec 5.10: state Grad-CAM Hz/row; drop unresolvable band claims.
   6. Abstract: attach Wilson CI to Pre-CHF F1; report the MEAN, not
      the best seed; fix the CI currently mislabelled as F1.
   7. Table 11 / novelty table: remove 'first successful acoustic-only
      Pre-CHF detection' until grouped numbers support it.
""")


# ============================================================
#  MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description='Grouped / leakage-free boiling pipeline')
    ap.add_argument('--steps', nargs='+', default=['inspect'],
                    choices=['inspect', 'groups', 'split', 'spectrogram',
                             'augment', 'check', 'physics', 'kfold',
                             'noise', 'gradcam_audit', 'report', 'all'])
    ap.add_argument('--group_mode', default='blocks',
                    choices=['filename', 'gaps', 'blocks', 'acoustic'])
    ap.add_argument('--group_regex', default=None,
                    help=r"e.g. '(run\d+)' or '(P\d+)'")
    ap.add_argument('--block', type=int, default=120,
                    help='clips per pseudo-session for mode=blocks')
    ap.add_argument('--guard', type=int, default=GUARD_CLIPS)
    ap.add_argument('--fmax', type=int, default=FMAX_ANALYSIS)
    ap.add_argument('--spec_types', nargs='+', default=['STFT'])
    ap.add_argument('--spec_type', default='STFT')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--target_n', type=int, default=TARGET_N)
    ap.add_argument('--model_path', default=None)
    ap.add_argument('--snr_levels', nargs='+', type=float,
                    default=[30, 20, 10, 0])
    ap.add_argument('--max_per_class', type=int, default=None)
    a = ap.parse_args()
    S = a.steps; ALL = 'all' in S

    if ALL or 'inspect' in S:
        step_inspect()
    if ALL or 'physics' in S:
        step_physics()
    if ALL or 'groups' in S:
        discover_groups(a.group_mode, a.group_regex, a.block)
    if ALL or 'split' in S:
        grouped_split(guard=a.guard)
    if ALL or 'spectrogram' in S:
        generate_spectrograms(a.spec_types, a.fmax)
    if ALL or 'augment' in S:
        augment_train_only(a.target_n, a.spec_types)
    if ALL or 'check' in S:
        check_integrity()
    if ALL or 'gradcam_audit' in S:
        gradcam_resolution(fmax=a.fmax)
    if ALL or 'kfold' in S:
        grouped_kfold(a.spec_type, a.n_splits, a.epochs)
    if 'noise' in S:
        if not a.model_path:
            print("  --model_path required for --steps noise")
        else:
            noise_robustness_wav(a.model_path, a.spec_type,
                                 tuple(a.snr_levels), a.fmax,
                                 a.max_per_class)
    if ALL or 'report' in S:
        report_honest()


if __name__ == '__main__':
    main()
