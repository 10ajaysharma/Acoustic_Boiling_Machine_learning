#!/bin/bash
#SBATCH --partition=r4_cpu
#SBATCH --job-name=boiling_spectrogram
#SBATCH --output=logs/spectrogram_%j.out
#SBATCH --error=logs/spectrogram_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0

# NOTE: submit this from inside ~/Dataset_of_fluid_boiling, e.g.:
#   cd ~/Dataset_of_fluid_boiling && sbatch job_step1_spectrogram.sh
# (the logs/ path above is relative to wherever you run `sbatch` from)

echo "=============================="
echo "STEP 1: SPECTROGRAM GENERATION"
echo "Start: $(date)"
echo "=============================="

mkdir -p logs
module load python/anaconda3-3.2022.10
source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh
conda activate acoustic_boiling
cd "$HOME/Dataset_of_fluid_boiling"

python3 complete_pipeline.py --steps spectrogram

echo "Spectrogram Done: $(date)"
