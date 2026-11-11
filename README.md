# DBConformer baseline: Windows one-click package (NVIDIA GPU)

Reproduces the DBConformer paper baselines (arXiv 2506.21140, Table IV). The zip has **no data**;
the script downloads and prepares every dataset itself, then trains on your GPU.

## Run
1. Unzip anywhere with enough free space (about 10 GB in the default mode).
2. Double-click `START_HERE.bat` (or run `powershell -ExecutionPolicy Bypass -File setup_and_run.ps1`).

It does, in order:
1. Finds Python 3.10-3.12 (installs 3.12 with winget if missing) and creates `.venv`.
2. Installs CUDA PyTorch (tries CUDA 12.6, then 12.4) and the other requirements; checks the GPU is visible.
3. Downloads and exports the datasets it needs into `DBConformer/data/` (default mode: BNCI2014001 and BNCI2014004 via MOABB, small).
4. Trains on GPU 0, one job after another, and prints a table of each variant against your own baseline after every job.

## Modes
- **`improve` (default):** LOSO on BNCI2014001 and BNCI2014004. Runs the baseline and every improvement variant (5 seeds x all subjects each).
- **`--mode reproduce`:** the original paper-table baseline jobs on all datasets (OpenBMI is about 60 GB raw).

## Improvement variants (flags of `DBConformer_LOSO.py` / `DBConformer_CO.py`)
| Variant | Flags | Idea |
|---|---|---|
| `baseline` | none | Released code, last-epoch test accuracy |
| `val_trials` / `val_subject` | `--val trials` / `--val subject` | Pick the best-validation checkpoint (validation from source data only) |
| `aug_shift`, `aug_noise`, `aug_segrec`, `aug_combo` | `--aug ...` | EEG augmentation on training batches |
| `tta` | `--tta shift` | Average over time-shifted test copies |
| `label_smoothing`, `ema`, `swa`, `cosine`, `adamw`, `grad_clip` | see `VARIANTS` in `run_pipeline.py` | Training setup |
| `tf_drop` | `--tf_drop 0.3` | Exploratory architecture tweak |
| `train_combo`, `data_combo`, `all_trials`, `all_ens3` | combinations | `all_ens3` adds a 3-model ensemble (3x cost) |
| `ens3` | `--ens 3` | 3-model ensemble (3x cost) |

Pick a subset: `START_HERE.bat --variants baseline ema cosine train_combo`. Add within-subject runs: `--settings LOSO CO`.
No test labels are used for any choice. Gains are unmeasured until you run this; differences under about 0.5 points are probably noise.
`val_*` trains on less data than the baseline, so compare it with the baseline before crediting other flags.

## Other options
| Option | Effect |
|---|---|
| `--skip-download` / `--skip-train` | Only train / only prepare data |
| `--jobs CO:BNCI2014001 ...` | reproduce mode: only these jobs |
| `--allow-cpu` | Train on CPU if no CUDA GPU is found (very slow) |
| `--no-openbmi`, `--clean-openbmi-raw`, `--all` | reproduce mode only |

## Resuming and results
- Safe to stop and re-run: prepared datasets and finished jobs are skipped.
- Logs: `logs_out/<variant>/<setting>_<dataset>.out`. Table: `python collect_results.py logs_out` (use `.venv\Scripts\python`).

## Notes
- `DBConformer/` is the upstream code plus local patches (device selection, optional imports, OpenBMI support,
  portable folder creation instead of `mkdir -p`). No model or training logic changed.
- Released-code values are used. CO runs lr 1e-4 / 200 epochs; the paper text says 1e-3 / 100.
- Needs a recent NVIDIA driver (CUDA 12.6 build needs about 560 or newer). Update it if the script says CUDA is unavailable.
- Citations required for the datasets: BCI Competition IV, OpenBMI (Lee et al., GigaScience 2019), MOABB datasets per their pages.
