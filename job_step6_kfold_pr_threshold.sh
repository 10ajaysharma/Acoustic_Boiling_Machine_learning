#!/bin/bash
#SBATCH --partition=r4
#SBATCH --job-name=boiling_kfold_pr
#SBATCH --output=logs/kfold_pr_%j.out
#SBATCH --error=logs/kfold_pr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=30-00:00:00

# NEW — this step didn't exist in your old job scripts. It runs the
# functionality that used to live only in the separate
# run_kfold_pr_threshold.py, now merged into complete_pipeline.py:
#   prechf     -> detailed Pre-CHF confusion breakdown
#   stats5seed -> 5-seed statistical validation   (paper Table 5)
#   kfold      -> stratified 5-fold CV, real retrain per fold (Table 6)
#   pr         -> precision-recall curve / PR-AUC  (paper Figure 15)
#   threshold  -> decision-threshold sensitivity   (paper Table 7)
#
# kfold and stats5seed each retrain the best model 5 times from
# scratch, so this is one of the slower jobs — run it after
# job_step3_train.sh has produced all_results.json and the trained
# .h5 models.
#
# Submit from ~/Dataset_of_fluid_boiling :
#   cd ~/Dataset_of_fluid_boiling && sbatch job_step6_kfold_pr_threshold.sh

echo "=============================="
echo "STEP 6: K-FOLD CV / PR-AUC / THRESHOLD / 5-SEED STATS"
echo "Start: $(date)"
echo "=============================="

mkdir -p logs
module load python/anaconda3-3.2022.10
source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh
conda activate acoustic_boiling
cd "$HOME/Dataset_of_fluid_boiling"

python3 complete_pipeline.py \
    --steps prechf stats5seed kfold pr threshold \
    --n_seeds 5

echo "=============================="
echo "Step 6 Done: $(date)"
echo "=============================="
