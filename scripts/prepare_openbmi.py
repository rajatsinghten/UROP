"""Export OpenBMI (MOABB Lee2019_MI) following the DBConformer paper:
4-40 Hz band-pass, resample to 256 Hz, 20 sensorimotor channels, [0, 4] s,
both sessions, training runs only (54 subjects x 2 sessions x 100 trials).

Subjects are exported one at a time (resumable) and concatenated at the end.
Usage: python -I prepare_openbmi.py <out_dir>
"""
import sys
import os
import numpy as np
import pandas as pd
from moabb.datasets import Lee2019_MI
from moabb.paradigms import MotorImagery

CHANNELS = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']

out = sys.argv[1]
parts = os.path.join(out, 'per_subject')
os.makedirs(parts, exist_ok=True)

ds = Lee2019_MI()
paradigm = MotorImagery(n_classes=2, fmin=4, fmax=40, resample=256, channels=CHANNELS)
for s in ds.subject_list:
    f = os.path.join(parts, f'S{s:02d}_X.npy')
    if os.path.exists(f):
        continue
    X, labels, meta = paradigm.get_data(dataset=ds, subjects=[s])
    np.save(os.path.join(parts, f'S{s:02d}_labels.npy'), labels)
    meta.to_csv(os.path.join(parts, f'S{s:02d}_meta.csv'))
    np.save(f, X.astype(np.float32))
    print(f'S{s:02d}', X.shape, dict(zip(*np.unique(labels, return_counts=True))), flush=True)

Xs, ys, metas = [], [], []
for s in ds.subject_list:
    Xs.append(np.load(os.path.join(parts, f'S{s:02d}_X.npy')))
    ys.append(np.load(os.path.join(parts, f'S{s:02d}_labels.npy'), allow_pickle=True))
    metas.append(pd.read_csv(os.path.join(parts, f'S{s:02d}_meta.csv'), index_col=0))
X, y, meta = np.concatenate(Xs), np.concatenate(ys), pd.concat(metas, ignore_index=True)
np.save(os.path.join(out, 'X.npy'), X)
np.save(os.path.join(out, 'labels.npy'), y)
meta.to_csv(os.path.join(out, 'meta.csv'))
print('OpenBMI', X.shape, dict(zip(*np.unique(y, return_counts=True))), flush=True)
