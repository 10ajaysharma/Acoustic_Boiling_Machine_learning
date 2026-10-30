#!/bin/bash
#SBATCH --partition=r4_cpu
#SBATCH --job-name=boiling_split_aug
#SBATCH --output=logs/split_aug_%j.out
#SBATCH --error=logs/split_aug_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0

# Submit from ~/Dataset_of_fluid_boiling :
#   cd ~/Dataset_of_fluid_boiling && sbatch job_step2_split_aug.sh

echo "=============================="
echo "STEP 2: SPLIT + AUGMENT + CHECK"
echo "Start: $(date)"
echo "=============================="

mkdir -p logs
module load python/anaconda3-3.2022.10
source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh
conda activate acoustic_boiling
cd "$HOME/Dataset_of_fluid_boiling"

python3 complete_pipeline.py --steps split
python3 complete_pipeline.py --steps augment
python3 complete_pipeline.py --steps check

echo "Split/Augment Done: $(date)"
