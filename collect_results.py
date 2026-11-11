"""Summarize DBConformer runs against the paper (Table IV, arXiv 2506.21140v2).

Usage: python collect_results.py <dir with <setting>_<dataset>.out files>
Each .out ends with: per-subject means, method name, overall mean, std over seeds.
"""
import sys
import os
import re
import glob

PAPER = {  # (setting, dataset): (mean, std)
    ('CO', 'BNCI2014001'): (80.16, 1.8), ('CO', 'BNCI2014004'): (75.09, 1.5),
    ('CO', 'Zhou2016'): (82.49, 2.4), ('CO', 'MI1-7'): (87.21, 1.3),
    ('CO', 'BNCI2014002'): (79.14, 0.2), ('CO', 'OpenBMI'): (72.12, 0.1),
    ('CV', 'BNCI2014001'): (83.66, 0.4), ('CV', 'BNCI2014004'): (74.17, 0.6),
    ('CV', 'Zhou2016'): (91.54, 0.2), ('CV', 'MI1-7'): (90.33, 0.5),
    ('CV', 'BNCI2014002'): (81.36, 0.3), ('CV', 'OpenBMI'): (72.89, 0.2),
    ('LOSO', 'BNCI2014001'): (77.67, 0.6), ('LOSO', 'BNCI2014004'): (69.85, 0.5),
    ('LOSO', 'Zhou2016'): (85.37, 0.9), ('LOSO', 'MI1-7'): (76.33, 0.7),
    ('LOSO', 'BNCI2014002'): (77.17, 0.8), ('LOSO', 'OpenBMI'): (73.61, 0.2),
}

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs_out'
# logs_out/<variant>/<setting>_<dataset>.out ; flat files directly in logs_out count as variant "baseline"
rows = []
for f in sorted(glob.glob(os.path.join(out_dir, '*.out')) + glob.glob(os.path.join(out_dir, '*', '*.out'))):
    variant = os.path.basename(os.path.dirname(f)) if os.path.dirname(f) != out_dir.rstrip('/\\') else 'baseline'
    setting, data = os.path.basename(f)[:-4].split('_', 1)
    lines = [l.strip() for l in open(f, errors='replace') if l.strip()]
    seeds = [float(x) for x in re.findall(r'Avg acc:\s+([0-9.]+)', '\n'.join(lines))]
    done = len(lines) >= 3 and lines[-3].startswith('DBConformer') and lines[-3].endswith(f'_{data}')
    mean, std = (float(lines[-2]), float(lines[-1])) if done else (None, None)
    rows.append((setting, data, variant, mean, std, seeds))

base = {(s, d): m for s, d, v, m, _, _ in rows if v == 'baseline' and m is not None}
print(f"{'Setting':<5} {'Dataset':<12} {'Variant':<16} {'Ours':>14} {'vs our baseline':>16} {'Paper':>14}  Seeds")
for setting, data, variant, mean, std, seeds in sorted(rows, key=lambda r: (r[0], r[1], r[2] != 'baseline', r[2])):
    paper = PAPER.get((setting, data))
    ours = f'{mean:.2f} ± {std:.2f}' if mean is not None else f'running ({len(seeds)}/5)'
    b = base.get((setting, data))
    delta = f'{mean - b:+.2f}' if (mean is not None and b is not None and variant != 'baseline') else ''
    ref = f'{paper[0]:.2f} ± {paper[1]:.1f}' if paper else '-'
    print(f"{setting:<5} {data:<12} {variant:<16} {ours:>14} {delta:>16} {ref:>14}  {', '.join(f'{x:.2f}' for x in seeds)}")
