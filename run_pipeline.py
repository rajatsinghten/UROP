"""Download + prepare datasets, then train DBConformer (baseline + improvement variants) on the local GPU.

Default mode "improve": LOSO on BNCI2014001 and BNCI2014004, baseline vs each improvement flag vs combinations
(small downloads, no OpenBMI). Mode "reproduce": the paper-table baseline jobs on all datasets.

Resumable: finished datasets and finished training jobs are skipped on re-run.
Called by setup_and_run.ps1 with the venv's Python; can also be run directly:

  python run_pipeline.py                        # improve mode: prepare data, then baseline + all variants
  python run_pipeline.py --mode reproduce       # paper baseline jobs on all datasets (OpenBMI ~60 GB)
  python run_pipeline.py --variants baseline ema cosine   # only these variants (names in VARIANTS)
  python run_pipeline.py --settings LOSO CO     # add within-subject (CO) runs
  python run_pipeline.py --skip-train           # data only
  python run_pipeline.py --skip-download        # train only
  python run_pipeline.py --jobs CO:BNCI2014001 LOSO:Zhou2016
  python run_pipeline.py --no-openbmi           # skip the ~60 GB OpenBMI download and its jobs
  python run_pipeline.py --clean-openbmi-raw    # delete OpenBMI raw downloads after processing
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, 'DBConformer')
DATA = os.path.join(REPO, 'data')                  # layout the DBConformer code expects
MNE_DATA = os.path.join(ROOT, 'raw_downloads', 'mne_data')
BLANKERTZ_DIR = os.path.join(ROOT, 'raw_downloads', 'blankertz2007')
BLANKERTZ_URL = 'https://www.bbci.de/competition/download/competition_iv/BCICIV_1calib_1000Hz_mat.zip'
LOGS = os.path.join(ROOT, 'logs_out')

# dataset name used by the training scripts -> file that exists once preparation finished
READY_FILE = {
    'BNCI2014001': 'BNCI2014001/meta.csv', 'BNCI2014002': 'BNCI2014002/meta.csv',
    'BNCI2014004': 'BNCI2014004/meta.csv', 'Zhou2016': 'Zhou2016/meta.csv',
    'MI1-7': 'MI1/meta-7.csv', 'OpenBMI': 'OpenBMI/meta.csv',
}
# Light jobs first so results appear early; OpenBMI last (54 subjects, slowest).
DEFAULT_JOBS = ['CO:BNCI2014001', 'CO:BNCI2014004', 'CO:Zhou2016', 'LOSO:Zhou2016',
                'CO:MI1-7', 'LOSO:MI1-7', 'CO:BNCI2014002', 'LOSO:BNCI2014002',
                'CO:OpenBMI', 'LOSO:OpenBMI']
ALREADY_DONE_ON_MAC = ['LOSO:BNCI2014001', 'LOSO:BNCI2014004']

# Improvement variants = extra CLI flags for DBConformer_<setting>.py. Each run = 5 seeds x all subjects (one baseline-run cost,
# except ens3 = 3x). Order: baseline first, single-flag ablations, combinations, expensive ones last.
TRAIN_COMBO = ['--label_smoothing', '0.1', '--ema_decay', '0.995', '--lr_sched', 'cosine', '--warmup_epochs', '5',
               '--weight_decay', '0.01', '--optimizer', 'adamw', '--grad_clip', '1.0']
DATA_COMBO = ['--aug', 'shift,noise,segrec', '--tta', 'shift']
VARIANTS = {
    'baseline': [],
    'val_trials': ['--val', 'trials'],
    'val_subject': ['--val', 'subject'],
    'aug_shift': ['--aug', 'shift'],
    'aug_noise': ['--aug', 'noise'],
    'aug_segrec': ['--aug', 'segrec'],
    'aug_combo': ['--aug', 'shift,noise,segrec'],
    'tta': ['--tta', 'shift'],
    'label_smoothing': ['--label_smoothing', '0.1'],
    'ema': ['--ema_decay', '0.995'],
    'swa': ['--swa_epochs', '10'],
    'cosine': ['--lr_sched', 'cosine', '--warmup_epochs', '5'],
    'adamw': ['--weight_decay', '0.01', '--optimizer', 'adamw'],
    'grad_clip': ['--grad_clip', '1.0'],
    'tf_drop': ['--tf_drop', '0.3'],                          # exploratory architecture tweak
    'train_combo': TRAIN_COMBO,
    'data_combo': DATA_COMBO,
    'all_trials': TRAIN_COMBO + DATA_COMBO + ['--val', 'trials'],
    'all_ens3': TRAIN_COMBO + DATA_COMBO + ['--val', 'trials', '--ens', '3'],
    'ens3': ['--ens', '3'],
}
IMPROVE_DATASETS = ['BNCI2014001', 'BNCI2014004']


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def ready(name):
    return os.path.exists(os.path.join(DATA, READY_FILE[name]))


def child_env():
    env = dict(os.environ)
    env['MNE_DATA'] = MNE_DATA            # MOABB/MNE download cache lives inside this folder
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    return env


def run(cmd, cwd=ROOT, out=None):
    log('RUN ' + ' '.join(cmd))
    if out:
        with open(out, 'w', encoding='utf-8') as f:
            return subprocess.call(cmd, cwd=cwd, env=child_env(), stdout=f, stderr=subprocess.STDOUT)
    return subprocess.call(cmd, cwd=cwd, env=child_env())


def prepare_moabb(name):
    if ready(name):
        log(f'{name}: already prepared')
        return True
    rc = run([sys.executable, os.path.join(ROOT, 'scripts', 'prepare_moabb.py'), DATA, name])
    return rc == 0 and ready(name)


def download(url, dest):
    """Resumable HTTP download."""
    part = dest + '.part'
    for attempt in range(1, 6):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0                      # server ignored Range -> restart
                with open(part, 'ab' if have else 'wb') as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            os.replace(part, dest)
            return True
        except Exception as e:
            log(f'download attempt {attempt} failed: {e}')
            time.sleep(5 * attempt)
    return False


def prepare_blankertz():
    if ready('MI1-7'):
        log('MI1-7 (Blankertz2007): already prepared')
        return True
    os.makedirs(BLANKERTZ_DIR, exist_ok=True)
    zpath = os.path.join(BLANKERTZ_DIR, 'BCICIV_1calib_1000Hz_mat.zip')
    if not os.path.exists(os.path.join(BLANKERTZ_DIR, 'BCICIV_calib_ds1g_1000Hz.mat')):
        if not os.path.exists(zpath) and not download(BLANKERTZ_URL, zpath):
            return False
        with zipfile.ZipFile(zpath) as z:
            z.extractall(BLANKERTZ_DIR)
    rc = run([sys.executable, os.path.join(ROOT, 'scripts', 'prepare_blankertz2007.py'),
              BLANKERTZ_DIR, os.path.join(DATA, 'MI1')])
    return rc == 0 and ready('MI1-7')


def prepare_openbmi(clean_raw):
    if ready('OpenBMI'):
        log('OpenBMI: already prepared')
        return True
    rc = run([sys.executable, os.path.join(ROOT, 'scripts', 'prepare_openbmi.py'),
              os.path.join(DATA, 'OpenBMI')])
    ok = rc == 0 and ready('OpenBMI')
    if ok:  # per-subject intermediates are only needed to resume
        shutil.rmtree(os.path.join(DATA, 'OpenBMI', 'per_subject'), ignore_errors=True)
        if clean_raw:
            for d in os.listdir(MNE_DATA) if os.path.isdir(MNE_DATA) else []:
                if 'lee2019' in d.lower():
                    shutil.rmtree(os.path.join(MNE_DATA, d), ignore_errors=True)
    return ok


def job_done(setting, data, variant):
    f = os.path.join(LOGS, variant, f'{setting}_{data}.out')
    if not os.path.exists(f):
        return False
    lines = [l.strip() for l in open(f, errors='replace') if l.strip()]
    return len(lines) >= 3 and lines[-3].startswith('DBConformer') and lines[-3].endswith(f'_{data}')


def check_gpu(allow_cpu):
    code = ('import torch;print(torch.cuda.is_available(),'
            'torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout.strip()
    if out.startswith('True'):
        log('GPU: ' + out[5:])
        return
    log('No CUDA GPU visible to PyTorch (check NVIDIA driver / CUDA build of torch).')
    if not allow_cpu:
        sys.exit('Aborting. Re-run with --allow-cpu to train on CPU anyway (very slow).')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['improve', 'reproduce'], default='improve')
    ap.add_argument('--variants', nargs='*', help='subset of: ' + ' '.join(VARIANTS))
    ap.add_argument('--settings', nargs='*', default=['LOSO'], choices=['LOSO', 'CO'], help='improve mode only')
    ap.add_argument('--datasets', nargs='*', default=IMPROVE_DATASETS, help='improve mode only')
    ap.add_argument('--skip-download', action='store_true')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--jobs', nargs='*', help='reproduce mode: e.g. CO:BNCI2014001 LOSO:OpenBMI')
    ap.add_argument('--all', action='store_true', help='reproduce mode: also rerun the 2 LOSO jobs already done on the Mac')
    ap.add_argument('--no-openbmi', action='store_true')
    ap.add_argument('--clean-openbmi-raw', action='store_true')
    ap.add_argument('--allow-cpu', action='store_true')
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    os.makedirs(MNE_DATA, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    log(f'Free disk space: {free_gb:.0f} GB')

    # job = (setting, dataset, variant)
    if args.mode == 'reproduce':
        names = args.jobs or (ALREADY_DONE_ON_MAC + DEFAULT_JOBS if args.all else DEFAULT_JOBS)
        jobs = [(j.split(':')[0], j.split(':')[1], 'baseline') for j in names]
    else:
        variants = args.variants or list(VARIANTS)
        unknown = [v for v in variants if v not in VARIANTS]
        if unknown:
            sys.exit(f'Unknown variants {unknown}; choose from {list(VARIANTS)}')
        jobs = [(st, d, v) for v in variants for d in args.datasets for st in args.settings]
    if args.no_openbmi:
        jobs = [j for j in jobs if j[1] != 'OpenBMI']
    jobs = [j for j in jobs if j[1] in READY_FILE or sys.exit(f'Unknown dataset {j[1]}')]
    needed = [d for d in READY_FILE if d in {j[1] for j in jobs}]
    log(f'{len(jobs)} training jobs over datasets {needed}')

    if not args.skip_download:
        failed = []
        steps = [(n, lambda n=n: prepare_moabb(n)) for n in
                 ['BNCI2014001', 'BNCI2014004', 'BNCI2014002', 'Zhou2016'] if n in needed]
        if 'MI1-7' in needed:
            steps.append(('MI1-7', prepare_blankertz))
        if 'OpenBMI' in needed:
            steps.append(('OpenBMI', lambda: prepare_openbmi(args.clean_openbmi_raw)))
        for name, fn in steps:
            log(f'=== Preparing {name} ===')
            if not fn():
                failed.append(name)
                log(f'!! {name} failed; its jobs will be skipped (re-run the script to retry)')
        log('Data preparation finished' + (f'; failed: {failed}' if failed else ' (all OK)'))

    if args.skip_train:
        return
    check_gpu(args.allow_cpu)
    for setting, data, variant in jobs:
        job = f'{setting}:{data}:{variant}'
        if not ready(data):
            log(f'SKIP {job}: dataset not prepared')
        elif job_done(setting, data, variant):
            log(f'SKIP {job}: already finished')
        else:
            log(f'START {job}')
            os.makedirs(os.path.join(LOGS, variant), exist_ok=True)
            t = time.time()
            rc = run([sys.executable, '-u', f'DBConformer_{setting}.py', '0', data] + VARIANTS[variant], cwd=REPO,
                     out=os.path.join(LOGS, variant, f'{setting}_{data}.out'))
            log(f'END {job} exit={rc} after {(time.time() - t) / 60:.1f} min')
            subprocess.call([sys.executable, os.path.join(ROOT, 'collect_results.py'), LOGS])
    log('ALL DONE. Results table: python collect_results.py logs_out')


if __name__ == '__main__':
    main()
