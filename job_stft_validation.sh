#!/bin/bash
#SBATCH --partition=r4
#SBATCH --job-name=stft_val
#SBATCH --output=logs/stft_val_%j.out
#SBATCH --error=logs/stft_val_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
# Bootstrap CI, calibration, ROC and hard-negative on the ORIGINAL STFT model.

mkdir -p logs
module load python/anaconda3-3.2022.10
source /home/opt/anaconda/3.2022/10/etc/profile.d/conda.sh
conda activate acoustic_boiling
cd "$HOME/Dataset_of_fluid_boiling"

RES="$HOME/new_acoustic_boiling_result/results"
FIG="$HOME/new_acoustic_boiling_result/figures"

BEST=$(python3 -c "
import json, pandas as pd
df = pd.DataFrame(json.load(open('$HOME/new_acoustic_boiling_result/all_results.json')))
b = df.loc[df['test_accuracy'].idxmax()]
print(b['model_name'] + '+' + b['spec_type'])
")
echo "Best config: $BEST"
if [ "$BEST" != "Original_CNN+STFT" ]; then echo "ERROR: not STFT, stopping"; exit 1; fi

for f in confidence_intervals calibration roc hard_negative; do
  [ -f "$RES/$f.json" ] && cp "$RES/$f.json" "$RES/${f}_MEL_old.json"; done
for f in confidence_intervals calibration_analysis roc_curves hard_negative_analysis; do
  [ -f "$FIG/$f.png" ] && cp "$FIG/$f.png" "$FIG/${f}_MEL_old.png"; done

python3 -u complete_pipeline.py --steps ci calibration roc hard_negative --n_bootstrap 1000

for f in confidence_intervals calibration roc hard_negative; do
  echo "===== $f ====="; cat "$RES/$f.json"; echo; done
echo "Done: $(date)"
