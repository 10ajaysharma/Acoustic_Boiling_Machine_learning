#!/bin/bash
#SBATCH --partition=r4
#SBATCH --job-name=grouped_cv
#SBATCH --output=logs/grouped_%j.out
#SBATCH --error=logs/grouped_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
# Session-grouped CV (Table 13). Output: ~/coustic_boiling_grouped/

mkdir -p logs
module load python/anaconda3-3.2022.10
source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh
conda activate acoustic_boiling
cd "$HOME/Dataset_of_fluid_boiling"

echo "Start: $(date)"
python3 -u grouped_pipeline.py \
    --steps groups split spectrogram augment check kfold \
    --group_mode filename \
    --group_regex "^(\d+Wcm2|Transition)" \
    --guard 5 \
    --fmax 5000 \
    --spec_types STFT \
    --spec_type STFT \
    --n_splits 5 \
    --epochs 30
echo "Done: $(date)"
