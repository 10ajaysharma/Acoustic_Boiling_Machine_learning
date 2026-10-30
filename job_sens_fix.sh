#!/bin/bash
#SBATCH --partition=r4
#SBATCH --job-name=sens_fix
#SBATCH --output=logs/sens_fix_%j.out
#SBATCH --error=logs/sens_fix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

mkdir -p logs
module load python/anaconda3-3.2022.10
source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh
conda activate acoustic_boiling
cd "$HOME/Dataset_of_fluid_boiling/Dataset_sound_of_boiling"

python3 run_only_sens.py

echo "Sensitivity fix run done: $(date)"
