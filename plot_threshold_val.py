#!/usr/bin/env python3
"""Figure 12: Pre-CHF threshold sensitivity on the validation set."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/r4/sharma/new_acoustic_boiling_result'
d = json.load(open(f'{BASE}/results/threshold_val_tuned.json'))
c = d['validation_curve']; t = [r['threshold'] for r in c]
plt.figure(figsize=(8, 5))
plt.plot(t, [r['precision'] for r in c], 'o-', label='Precision')
plt.plot(t, [r['recall'] for r in c], 's-', label='Recall')
plt.plot(t, [r['f1'] for r in c], '^-', label='F1-score')
plt.axvline(d['chosen_threshold'], color='r', ls='--',
            label=f"Selected = {d['chosen_threshold']} (val F1 = {d['validation_at_chosen']['f1']})")
plt.xlabel('Decision threshold'); plt.ylabel('Metric'); plt.ylim(0, 1.05)
plt.title('Pre-CHF threshold sensitivity (validation set)')
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(f'{BASE}/figures/threshold_sensitivity.png', dpi=150)
print('Saved threshold_sensitivity.png')
