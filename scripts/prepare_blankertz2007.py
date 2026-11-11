"""Export BCI Competition IV dataset 1 (Blankertz2007) calibration data to the
layout DBConformer expects (data/MI1/X-7.npy, labels-7.npy):
8-30 Hz band-pass on the continuous signal, epochs [0.5, 3.5] s after cue,
kept at 1000 Hz (DBConformer's loader downsamples by 4 to 250 Hz itself).
Subjects in order a..g (a and f: left hand / foot, others: left / right hand;
c, d, e are artificially generated per the competition description).

Usage: python -I prepare_blankertz2007.py <dir_with_BCICIV_calib_ds1*_1000Hz.mat> <out_dir>
"""
import sys
import os
import numpy as np
import pandas as pd
import scipy.io
import mne

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
Xs, ys, rows = [], [], []
for subj in 'abcdefg':
    m = scipy.io.loadmat(os.path.join(src, f'BCICIV_calib_ds1{subj}_1000Hz.mat'), struct_as_record=False, squeeze_me=True)
    fs = int(m['nfo'].fs)
    assert fs == 1000
    cnt = 0.1 * m['cnt'].astype(np.float64).T  # (channels, samples), in microvolts
    cnt = mne.filter.filter_data(cnt, fs, 8, 30, verbose=False)
    pos, y = m['mrk'].pos.astype(int), m['mrk'].y.astype(int)
    start, stop = int(0.5 * fs), int(3.5 * fs)
    X = np.stack([cnt[:, p + start:p + stop] for p in pos]).astype(np.float32)
    classes = list(m['nfo'].classes)
    Xs.append(X)
    ys.append(y)
    rows += [{'subject': subj, 'label': lab, 'class': classes[0] if lab == -1 else classes[1]} for lab in y]
    print(subj, X.shape, classes, dict(zip(*np.unique(y, return_counts=True))), flush=True)

np.save(os.path.join(out, 'X-7.npy'), np.concatenate(Xs))
np.save(os.path.join(out, 'labels-7.npy'), np.concatenate(ys))
pd.DataFrame(rows).to_csv(os.path.join(out, 'meta-7.csv'))
print('Blankertz2007', np.concatenate(Xs).shape, flush=True)
