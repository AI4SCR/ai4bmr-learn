import pandas as pd
import numpy as np
from ai4bmr_learn.data.splits import generate_splits
rng = np.random.RandomState(42)
metadata = pd.DataFrame({
    'dx_name': rng.choice(['a', 'b', 'c'], size=1000, replace=True),
    'patient_id': rng.choice([f'patient_{i}' for i in range(20)], size=1000, replace=True),
})

metadata = pd.DataFrame({
    'dx_name': rng.choice(['a', 'b', 'c', pd.NA], size=1000, replace=True),
    'patient_id': rng.choice([f'patient_{i}' for i in range(20)], size=1000, replace=True),
})

test_size=0.2
val_size = 0.2
target_column_name='dx_name'
group_column_name='patient_id'
generate_splits(metadata, test_size=0.2, val_size = 0.2, target_column_name='dx_name')
generate_splits(metadata, test_size=0.2, val_size = 0.2, target_column_name='dx_name', group_column_name='patient_id')
generate_splits(metadata, test_size=0.2, val_size = 0.2, target_column_name='dx_name', stratify=True, group_column_name='patient_id')
generate_splits(metadata, test_size=0.5, val_size = 0.2, target_column_name='dx_name', stratify=True, use_filtered_targets_for_train=True, group_column_name='patient_id')
metadata = generate_splits(metadata, test_size=0.5, val_size = 0.2, target_column_name='dx_name', stratify=True, use_filtered_targets_for_train=True, group_column_name=None, encode_targets=True)
