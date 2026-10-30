# Acoustic Boiling Regime Classification — final code

Code that produced every result in the manuscript *Acoustic-Based Boiling Regime
Classification Using Convolutional Neural Networks: A Systematic Spectrogram
Analysis with Pre-CHF Detection* (A.P. Sharma, Jožef Stefan Institute).

All files are flat on purpose: copy them into `~/Dataset_of_fluid_boiling/` on
Skuta and submit jobs from there (`sbatch <job>.sh`). Outputs go to
`/home/r4/sharma/new_acoustic_boiling_result/` (set by `BASE_DIR` in
`complete_pipeline.py`). Raw WAV data: `~/Dataset_of_fluid_boiling/Dataset_sound_of_boiling/`.

Environment: `module load python/anaconda3-3.2022.10`,
`source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh`,
`conda activate acoustic_boiling` (TensorFlow 2.20 / Keras 3). CPU only.

## Files

| File | Role | Manuscript item |
| --- | --- | --- |
| complete_pipeline.py | Main pipeline (spectrograms, split, augmentation, training, all analyses). Includes k-fold memory fix, robust PR loading, sensitivity fix, arrow-style fix, Keras-3 Grad-CAM fix | Tables 3–7, 9–12, 14; Figs 1, 5–11, 13, 16, S1–S7 |
| run_only_sens.py | Augmentation sensitivity sweep only | Table 11, Fig 15 |
| tune_threshold_val.py | Pre-CHF threshold chosen on validation, reported on test | Table 8 |
| plot_threshold_val.py | Threshold figure from the validation sweep | Fig 12 |
| gradcam_logit.py | Logit-based, class-averaged Grad-CAM with frequency profile | Table 17, Fig 14 |
| grouped_pipeline.py | Session inspection and session-grouped CV | Tables 1, 13 |
| fix_model_paths.py | Points all_results.json to the copied original models | Methodology 3.8 |
| make_pipeline_fig_v2.py | Methodology pipeline figure | Fig 2 |
| make_spectrogram_fig_v2.py | Spectrogram example grid | Fig 3 |
| make_supplementary.py | Supplementary_Material.pdf and Figure_S1–S7.png | S1–S7 |
| job_step1_spectrogram.sh … job_step6_kfold_pr_threshold.sh | Main pipeline jobs | — |
| job_sens_fix.sh | Sensitivity sweep job | — |
| job_stft_validation.sh | CI, calibration, ROC, hard-negative on STFT | Table 14 |
| job_regen_figs.sh | Regenerates model-dependent figures with original models | Figs 5–8, 13 |
| job_grouped.sh | Session-grouped CV | Table 13 |
| original/acousticboiling_py.py | Original training script (models in `New folder/`) | Table 3 source |

## Order of execution

1. `sbatch job_step1_spectrogram.sh` → `job_step2_split_aug.sh` → `job_step3_train.sh` → `job_step4_analysis.sh`
2. `sbatch job_step5_revision.sh` and `sbatch job_sens_fix.sh`
3. `sbatch job_step6_kfold_pr_threshold.sh` (needs `--mem=180G`)
4. Copy original models and all_results.json from `New folder/`, then `python3 fix_model_paths.py`
5. `python3 tune_threshold_val.py`, `python3 plot_threshold_val.py`
6. `sbatch job_stft_validation.sh`, `sbatch job_regen_figs.sh`
7. `python3 gradcam_logit.py`, `python3 make_pipeline_fig_v2.py`, `python3 make_spectrogram_fig_v2.py`, `python3 make_supplementary.py`
8. `python3 grouped_pipeline.py --steps inspect`, then `sbatch job_grouped.sh`

The paper's headline numbers (99.75%, Pre-CHF F1 0.9508) come from the original
models; retraining with `job_step3_train.sh` gives similar but not identical
values (run-to-run variance, quantified by the five-seed validation).

`ALL_CODE_combined.txt` contains every file above in one text file.
