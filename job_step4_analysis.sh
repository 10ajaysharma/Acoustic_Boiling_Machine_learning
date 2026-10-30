#!/bin/bash
#SBATCH --partition=r4
#SBATCH --job-name=boiling_analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Submit from ~/Dataset_of_fluid_boiling :
#   cd ~/Dataset_of_fluid_boiling && sbatch job_step4_analysis.sh

echo "=============================="
echo "STEP 4: ANALYSIS + FIGURES"
echo "Start: $(date)"
echo "=============================="

mkdir -p logs
module load python/anaconda3-3.2022.10
source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh
conda activate acoustic_boiling
cd "$HOME/Dataset_of_fluid_boiling"

python3 complete_pipeline.py --steps analysis figures

echo "Analysis Done: $(date)"
