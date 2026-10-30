#!/bin/bash
#SBATCH --partition=r4
#SBATCH --job-name=boiling_revision
#SBATCH --output=logs/revision_%j.out
#SBATCH --error=logs/revision_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Submit from ~/Dataset_of_fluid_boiling :
#   cd ~/Dataset_of_fluid_boiling && sbatch job_step5_revision.sh

echo "=============================="
echo "STEP 5: JOURNAL REVISION"
echo "Start: $(date)"
echo "=============================="

mkdir -p logs
module load python/anaconda3-3.2022.10
source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh
conda activate acoustic_boiling
cd "$HOME/Dataset_of_fluid_boiling"

python3 complete_pipeline.py \
    --steps ablation sensitivity \
            significance calibration \
            roc ci hard_negative \
            gradcam noise contributions \
    --n_seeds 5 \
    --ablation_seeds 3 \
    --n_bootstrap 1000 \
    --noise_levels 50 60 70

echo "=============================="
echo "Revision Done: $(date)"
echo "=============================="
