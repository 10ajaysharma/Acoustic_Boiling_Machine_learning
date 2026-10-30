#!/bin/bash
#SBATCH --partition=r4
#SBATCH --job-name=regen_figs
#SBATCH --output=logs/regen_figs_%j.out
#SBATCH --error=logs/regen_figs_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
# Regenerates model-dependent figures with the ORIGINAL models.
# First run used:  --steps analysis            (Figures 5, 6, 8)
# Second run used: --steps prechf gradcam noise (Figures 7, 13 + noise tables)
# Do NOT add 'figures' here: it would overwrite fig_pipeline.png.

module load python/anaconda3-3.2022.10
source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh
conda activate acoustic_boiling
cd "$HOME/Dataset_of_fluid_boiling"
python3 -u complete_pipeline.py --steps prechf gradcam noise --noise_levels 50 60 70
cat ~/new_acoustic_boiling_result/results/prechf_analysis.json
