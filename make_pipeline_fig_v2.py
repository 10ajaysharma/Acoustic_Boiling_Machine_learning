import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUTFIG = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/r4/sharma/new_acoustic_boiling_result/figures/fig_pipeline.png"

C_DATA, C_PROC, C_TRAIN = '#1565C0', '#2E7D32', '#E65100'
C_EVAL, C_EXT, C_FINAL  = '#6A1B9A', '#AD1457', '#1B5E20'

fig, ax = plt.subplots(figsize=(20, 13))
ax.set_xlim(0, 20); ax.set_ylim(0, 13); ax.axis('off')
fig.patch.set_facecolor('white')

def box(x, y, w, h, text, color, fs=9.5):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle='round,pad=0.05,rounding_size=0.12',
                 linewidth=1.3, edgecolor='#263238', facecolor=color, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='white', zorder=4, linespacing=1.35)

def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color='#37474F', lw=2,
                                mutation_scale=18), zorder=2)
    if label:
        ax.text((x1+x2)/2 + lx, (y1+y2)/2 + ly, label, fontsize=9,
                style='italic', color='#37474F', ha='center', va='center')

ax.text(10, 12.55, 'Methodology pipeline — acoustic boiling regime classification',
        ha='center', fontsize=15, fontweight='bold', color='#1a1a1a')

# Row 1: data -> spectrograms -> split
W1, H1, Y1 = 4.8, 1.2, 11.2
box(3.5, Y1, W1, H1, '1. DATA ACQUISITION\nSinha et al. (2023), IIT Patna\n16,282 WAV clips | 1 s | 10 kHz', C_DATA)
box(10.0, Y1, 5.6, H1, '2. SPECTROGRAM GENERATION\nSTFT | MEL | MFCC | CQT | CHROMA\n224×224 px | librosa | f_max 5 kHz', C_PROC)
box(16.5, Y1, W1, H1, '3. STRATIFIED SPLIT (seed 42)\n70% train | 15% val | 15% test\nsplit BEFORE augmentation', C_PROC)
arrow(5.9, Y1, 7.2, Y1)
arrow(12.8, Y1, 14.1, Y1)

# Row 2: training set / val-test sets
Y2 = 9.35
box(10.0, Y2, 5.6, H1, '4a. TRAINING SET\naugmented to 13,000 clips per class\n6 augmentation operations', C_PROC)
box(16.5, Y2, W1, H1, '4b. VALIDATION & TEST SETS\noriginal clips only (no augmentation)\n2,441 validation | 2,445 test', C_DATA)
arrow(15.0, Y1 - H1/2, 11.6, Y2 + H1/2)
arrow(16.5, Y1 - H1/2, 16.5, Y2 + H1/2)

# Row 3: training -> primary evaluation
Y3 = 7.5
box(10.0, Y3, 5.6, H1, '5. MODEL TRAINING\nOriginal CNN (0.52M) | ResNet50V2-FT (24.75M)\nAdam | ≤50 epochs | early stopping', C_TRAIN)
box(16.5, Y3, W1, H1, '6. PRIMARY EVALUATION\naccuracy | P / R / F1 | confusion matrix\n2,445 original test clips', C_EVAL)
arrow(10.0, Y2 - H1/2, 10.0, Y3 + H1/2)
arrow(16.5, Y2 - H1/2, 16.5, Y3 + H1/2)
arrow(12.8, Y3, 14.1, Y3, 'trained\nmodels', ly=0.38)

# Legend + note in the empty upper-left area
handles = [mpatches.Patch(color=c, label=l) for c, l in
           [(C_DATA, 'Data'), (C_PROC, 'Processing'), (C_TRAIN, 'Training'),
            (C_EVAL, 'Evaluation'), (C_EXT, 'Extended validation'), (C_FINAL, 'Result')]]
ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.015, 0.765),
          fontsize=10.5, frameon=True, framealpha=1, edgecolor='#B0BEC5',
          title='Legend', title_fontsize=11)
ax.text(0.35, 6.95, 'All evaluation uses original\n(non-augmented) test recordings.',
        fontsize=10, style='italic', color='#37474F', va='top')

# Evaluation frame with 12 modules
FX0, FX1, FY0, FY1 = 0.25, 19.75, 1.85, 6.35
ax.add_patch(FancyBboxPatch((FX0, FY0), FX1 - FX0, FY1 - FY0,
             boxstyle='round,pad=0.02,rounding_size=0.2', linewidth=1.5,
             edgecolor='#6A1B9A', facecolor='#F3E5F5', zorder=1, ls='--'))
ax.text(FX0 + 0.25, FY1 - 0.28, 'Evaluation and validation modules',
        fontsize=11, fontweight='bold', color='#4A148C', va='center')
arrow(16.5, Y3 - H1/2, 16.5, FY1)

xs = [1.85 + i * 3.26 for i in range(6)]
GW, GH = 3.0, 1.45
row_a = [('7. STATISTICAL VALIDATION\n5 seeds: 99.26 ± 0.29%\nPre-CHF F1 0.878 ± 0.057', C_EVAL),
         ('8. 5-FOLD CROSS-VALIDATION\n99.60 ± 0.30%\nPre-CHF F1 0.992 ± 0.006', C_EVAL),
         ('9. NOISE ROBUSTNESS\n50 / 60 / 70 dB Gaussian\nSTFT −0.28 | MFCC −4.95 pp', C_EVAL),
         ('10. PR-AUC & THRESHOLD\nPR-AUC 0.9917\nthreshold 0.75 (validation)', C_EVAL),
         ('11. GRAD-CAM\nlogit, class-averaged (n=30)\nattention in 0.2–2 kHz', C_EVAL),
         ('12. ROC & CALIBRATION\nPre-CHF vs NB AUC 0.9997\nECE 0.0014', C_EVAL)]
row_b = [('13. HARD-NEGATIVE ANALYSIS\nNB ↔ Pre-CHF confusion\n0.29% of NB + Pre-CHF clips', C_EVAL),
         ('14. AUGMENTATION SENSITIVITY\n1× → 46× Pre-CHF data\nF1 saturates at 10× (0.91)', C_EXT),
         ('15. ABLATION STUDY\n5 variants × 3 seeds\nfull augmentation best (0.943)', C_EXT),
         ('16. SIGNIFICANCE TESTS\npaired t-test + Wilcoxon\nCNN vs ResNet p = 2.9×10⁻⁶', C_EXT),
         ('17. BOOTSTRAP 95% CI\nPre-CHF F1 0.9508\n[0.908, 0.984]', C_EXT),
         ('18. SESSION-GROUPED CV\nsessions from file names\nsession-disjoint acc. 99.27%', C_EXT)]
for x, (t, c) in zip(xs, row_a):
    box(x, 4.95, GW, GH, t, c, fs=8.4)
for x, (t, c) in zip(xs, row_b):
    box(x, 2.9, GW, GH, t, c, fs=8.4)

# Final result
arrow(10.0, FY0, 10.0, 1.3)
box(10.0, 0.75, 16.5, 1.05,
    '19. FINAL MODEL — Original CNN + STFT\n'
    'test accuracy 99.75%  |  Pre-CHF F1 0.9508  |  PR-AUC 0.9917  |  0.52M parameters  |  84.9 ms per clip (CPU)',
    C_FINAL, fs=10.5)

plt.savefig(OUTFIG, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUTFIG}")
