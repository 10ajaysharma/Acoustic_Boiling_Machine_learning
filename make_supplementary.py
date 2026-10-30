#!/usr/bin/env python3
"""
Builds the Supplementary Material (Figures S1-S7) from the final figures.
Outputs (in the figures folder):
  Supplementary_Material.pdf          one A4 page per figure, with caption
  supplementary/Figure_S1.png ... S7  separate files for journal upload
"""
import os, sys, shutil, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

FIG = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/r4/sharma/new_acoustic_boiling_result/figures"
OUT_PDF = os.path.join(FIG, "Supplementary_Material.pdf")
OUT_DIR = os.path.join(FIG, "supplementary")
os.makedirs(OUT_DIR, exist_ok=True)

PAPER = ("Acoustic-Based Boiling Regime Classification Using Convolutional "
         "Neural Networks: A Systematic Spectrogram Analysis with Pre-CHF Detection")

ITEMS = [
    ("complete_analysis.png",
     "Overall results for all model-spectrogram configurations",
     "Summary of the ten configurations (Original CNN and ResNet50V2-FT, each with STFT, "
     "MEL, MFCC, CQT and CHROMA). Test accuracy heatmap, Pre-CHF F1 heatmap (prior-work "
     "value 0.000 for reference), accuracy versus CPU inference time with a 100 ms real-time "
     "line, and accuracy versus parameter count. All numbers are listed in Table 3 of the "
     "main text. Best configuration: Original CNN + STFT, 99.75% accuracy, Pre-CHF F1 = "
     "0.9508, 0.52M parameters."),
    ("cnn_spectrogram_comparison.png",
     "Original CNN across the five spectrogram representations",
     "(a) Test accuracy: STFT 99.75%, MEL 99.67%, MFCC 99.47%, CQT 99.22%, CHROMA 95.38%. "
     "(b) Pre-CHF F1 with the prior-work value (0.000) and a 0.90 reference line: STFT "
     "0.9508, MEL 0.9344, MFCC 0.9231, CQT 0.8550, CHROMA 0.1519. (c) Training time per "
     "representation."),
    ("statistical_significance.png",
     "Statistical significance tests",
     "Paired t-test and Wilcoxon signed-rank test over five paired retrainings. (Left) "
     "Original CNN with STFT versus MEL: 99.62% +/- 0.18% versus 99.66% +/- 0.20%, t-test "
     "p = 0.74, Wilcoxon p = 1.0 (not significant). (Right) Original CNN versus "
     "ResNet50V2-FT: 99.62% versus 96.16%, t-test p = 2.9 x 10^-6; the Wilcoxon p = 0.0625 "
     "is the smallest value attainable with five pairs."),
    ("confidence_intervals.png",
     "Bootstrap 95% confidence intervals",
     "Per-class precision, recall and F1 of Original CNN + STFT on the 2,445 original test "
     "clips, with 95% confidence intervals from 1,000 bootstrap resamples. Pre-CHF: "
     "precision 0.9508 [0.886, 1.000], recall 0.9508 [0.893, 1.000], F1 0.9508 [0.908, "
     "0.984]. NB F1 0.9985 [0.9973, 0.9995]; BKG and TB 1.000."),
    ("calibration_analysis.png",
     "Calibration analysis",
     "Original CNN + STFT. (Left) Reliability diagram. (Centre) Distribution of the maximum "
     "softmax confidence; almost all predictions are made with confidence close to 1. "
     "(Right) Expected Calibration Error per class: BKG 0.000, NB 0.002, Pre-CHF 0.002, "
     "TB 0.000; overall ECE = 0.0014."),
    ("roc_curves.png",
     "ROC curves",
     "Original CNN + STFT. (Left) One-vs-rest ROC curves: AUC = 1.000 for BKG, NB and TB and "
     "0.9997 for Pre-CHF. (Right) Binary Pre-CHF versus NB, the hardest pair: AUC = 0.9997."),
    ("hard_negative_analysis.png",
     "NB / Pre-CHF hard-negative analysis",
     "Original CNN + STFT. (Left) Normalised confusion between NB and Pre-CHF: 3 of 2,006 NB "
     "clips predicted as Pre-CHF (0.15%) and 3 of 61 Pre-CHF clips predicted as NB (4.92%). "
     "(Right) Predicted-confidence distributions for correctly classified and confused "
     "clips. No Pre-CHF clip is confused with BKG or TB."),
]

A4 = (8.27, 11.69)
missing = []

with PdfPages(OUT_PDF) as pdf:
    # Cover page
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.86, "Supplementary Material", ha='center', fontsize=20, weight='bold')
    fig.text(0.5, 0.80, textwrap.fill(PAPER, 70), ha='center', va='top',
             fontsize=12, style='italic')
    fig.text(0.12, 0.66, "Contents", fontsize=14, weight='bold')
    for i, (_, title, _) in enumerate(ITEMS, 1):
        fig.text(0.12, 0.62 - (i - 1) * 0.035, f"Figure S{i}.  {title}", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)

    for i, (fname, title, caption) in enumerate(ITEMS, 1):
        path = os.path.join(FIG, fname)
        if not os.path.exists(path):
            missing.append(fname)
            print(f"  MISSING: {fname} (Figure S{i} skipped)")
            continue
        shutil.copy(path, os.path.join(OUT_DIR, f"Figure_S{i}.png"))
        img = mpimg.imread(path)
        h, w = img.shape[:2]
        aspect = h / w
        wf = 0.86
        hf = wf * A4[0] * aspect / A4[1]
        if hf > 0.66:
            hf = 0.66
            wf = hf * A4[1] / (A4[0] * aspect)
        left, top = (1 - wf) / 2, 0.92
        bottom = top - hf
        fig = plt.figure(figsize=A4)
        fig.text(0.5, 0.955, f"Figure S{i}. {title}", ha='center', va='top',
                 fontsize=13, weight='bold')
        ax = fig.add_axes([left, bottom, wf, hf])
        ax.imshow(img)
        ax.axis('off')
        fig.text(0.08, bottom - 0.02, textwrap.fill(f"Figure S{i}. " + caption, 98),
                 va='top', fontsize=9.5, linespacing=1.4)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)
        print(f"  Figure S{i}: {fname}")

    info = pdf.infodict()
    info['Title'] = "Supplementary Material - " + PAPER
    info['Author'] = "A.P. Sharma"

print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_DIR}/Figure_S1.png ... Figure_S{len(ITEMS)}.png")
if missing:
    print(f"WARNING: {len(missing)} figure(s) missing: {missing}")
