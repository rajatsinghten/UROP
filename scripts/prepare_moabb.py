"""Export MOABB MI datasets to the X.npy / labels.npy / meta.csv layout used by
DBConformer, MVCNet and MIRepNet (MOABB default MotorImagery paradigm, 8-32 Hz,
native sampling rate, dataset-default trial interval).

Usage: python -I prepare_moabb.py <out_root> [DatasetName ...]
"""
import sys
import os
import numpy as np
from moabb.datasets import BNCI2014_001, BNCI2014_002, BNCI2014_004, BNCI2015_001, Zhou2016
from moabb.paradigms import MotorImagery

DATASETS = {
    'BNCI2014001': (BNCI2014_001, 4),
    'BNCI2014002': (BNCI2014_002, 2),
    'BNCI2014004': (BNCI2014_004, 2),
    'BNCI2015001': (BNCI2015_001, 2),
    'Zhou2016': (Zhou2016, 3),
}

out_root = sys.argv[1]
names = sys.argv[2:] or list(DATASETS)
for name in names:
    cls, n_classes = DATASETS[name]
    ds = cls()
    paradigm = MotorImagery(n_classes=n_classes)
    X, labels, meta = paradigm.get_data(dataset=ds, subjects=ds.subject_list)
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, 'X.npy'), X.astype(np.float32))
    np.save(os.path.join(out, 'labels.npy'), labels)
    meta.to_csv(os.path.join(out, 'meta.csv'))
    print(name, X.shape, np.unique(labels, return_counts=True), flush=True)
