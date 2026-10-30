import os, random
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

BASE   = "/home/r4/sharma/new_acoustic_boiling_result"
SPLIT  = f"{BASE}/outputSpectrogram/Split_Final"
OUTFIG = f"{BASE}/figures/fig_spectrogram_examples.png"

CLASSES  = ['BKG', 'NB', 'Pre-CHF', 'TB']
SPECTROS = ['STFT', 'MEL', 'MFCC', 'CQT', 'CHROMA']

CLASS_LABELS = {
    'BKG'    : 'Background\n(BKG)',
    'NB'     : 'Nucleate\nBoiling (NB)',
    'Pre-CHF': 'Pre-CHF',
    'TB'     : 'Transition\nBoiling (TB)',
}
SPEC_LABELS = {
    'STFT'  : 'STFT\n(Linear freq)',
    'MEL'   : 'MEL\n(Perceptual)',
    'MFCC'  : 'MFCC\n(Cepstral)',
    'CQT'   : 'CQT\n(Log freq)',
    'CHROMA': 'CHROMA\n(Pitch-class)',
}

random.seed(42)

fig = plt.figure(figsize=(18, 14))

# Leave space on left for row labels and top for title
left_margin  = 0.10
right_margin = 0.01
top_margin   = 0.88
bottom_margin = 0.02

nrows = len(CLASSES)
ncols = len(SPECTROS)
cell_w = (right_margin - left_margin + 1 - right_margin - (1 - top_margin)) / ncols
# Simpler: use GridSpec
from matplotlib.gridspec import GridSpec
gs = GridSpec(nrows, ncols, figure=fig,
              left=left_margin, right=1-right_margin,
              top=top_margin, bottom=bottom_margin,
              hspace=0.04, wspace=0.04)

for row, cls in enumerate(CLASSES):
    for col, spec in enumerate(SPECTROS):
        ax = fig.add_subplot(gs[row, col])

        # Try test first, then train
        for split in ['test', 'validation', 'train']:
            folder = os.path.join(SPLIT, spec, split, cls)
            if os.path.isdir(folder):
                break

        files = [f for f in os.listdir(folder) if f.endswith('.png')] \
                if os.path.isdir(folder) else []

        if files:
            img_path = os.path.join(folder, random.choice(files))
            img = mpimg.imread(img_path)
            ax.imshow(img, aspect='auto', origin='upper')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.set_facecolor('#eeeeee')

        ax.set_xticks([])
        ax.set_yticks([])

        # Column headers (top row only)
        if row == 0:
            ax.set_title(SPEC_LABELS[spec], fontsize=11,
                         fontweight='bold', pad=6)

        # Row labels (left column only) — use fig.text so they
        # sit in the left margin regardless of image content
        if col == 0:
            # get axes position in figure coords
            pos = ax.get_position()
            y_center = (pos.y0 + pos.y1) / 2
            fig.text(left_margin - 0.01, y_center,
                     CLASS_LABELS[cls],
                     ha='right', va='center',
                     fontsize=11, fontweight='bold',
                     multialignment='center')

fig.suptitle(
    'Spectrogram Representations for All Four Boiling Regimes\n'
    'SR = 10 kHz  |  N_FFT = 512  |  Hop = 256  |  224 × 224 pixels',
    fontsize=13, fontweight='bold', y=0.97)

plt.savefig(OUTFIG, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUTFIG}")
