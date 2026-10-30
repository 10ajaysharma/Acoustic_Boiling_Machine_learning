#!/usr/bin/env python3
"""
Step 3.8 of the methodology: after copying the ORIGINAL all_results.json and
models from 'New folder' into new_acoustic_boiling_result/, point every
model_path to the new models folder and verify that the best model is
Original_CNN + STFT.

Copy first (on Skuta):
  cp "$HOME/Dataset_of_fluid_boiling/Dataset_sound_of_boiling/New folder/all_results.json" ~/new_acoustic_boiling_result/
  cp "$HOME/Dataset_of_fluid_boiling/Dataset_sound_of_boiling/New folder/models/"*_best.h5 ~/new_acoustic_boiling_result/models/
"""
import json, os
import pandas as pd

BASE = '/home/r4/sharma/new_acoustic_boiling_result'
RJ = f'{BASE}/all_results.json'
results = json.load(open(RJ))
for r in results:
    new = os.path.join(BASE, 'models', os.path.basename(r.get('model_path', '')))
    if r.get('model_path') != new:
        print(f"Fixed: {r.get('model_path')}\n   -> {new}")
    r['model_path'] = new
json.dump(results, open(RJ, 'w'), indent=2)

df = pd.DataFrame(results)
print(df[['model_name', 'spec_type', 'test_accuracy', 'prechf_f1']]
      .sort_values('test_accuracy', ascending=False).to_string(index=False))
missing = [p for p in df['model_path'] if not os.path.exists(p)]
print('All model files present' if not missing else f'MISSING: {missing}')
