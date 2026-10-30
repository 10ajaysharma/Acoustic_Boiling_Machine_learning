#!/bin/bash
#SBATCH --partition=r4
#SBATCH --job-name=boiling_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Submit from ~/Dataset_of_fluid_boiling :
#   cd ~/Dataset_of_fluid_boiling && sbatch job_step3_train.sh
#
# NOTE: only Original_CNN and ResNet50V2_FT are wired up in
# complete_pipeline.py's BUILDERS dict (matches the paper's Table 3 —
# 2 models x 5 spectrogram types = 10 experiments). EfficientNetB0 /
# MobileNetV3 were an earlier, abandoned plan and are intentionally
# left out here.

echo "=============================="
echo "STEP 3: TRAIN (10 EXPERIMENTS)"
echo "Start: $(date)"
echo "=============================="

mkdir -p logs
module load python/anaconda3-3.2022.10
source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh
conda activate acoustic_boiling
cd "$HOME/Dataset_of_fluid_boiling"

python3 complete_pipeline.py \
    --steps train \
    --epochs 50 \
    --models Original_CNN ResNet50V2_FT \
    --spec_types STFT MEL MFCC CQT CHROMA

echo "Training Done: $(date)"
