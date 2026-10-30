#!/usr/bin/env python3
from complete_pipeline import step_augmentation_sensitivity

if __name__ == "__main__":
    step_augmentation_sensitivity(
        model_name='Original_CNN',
        spec_type='STFT',
        epochs=40,
        n_seeds=3
    )
