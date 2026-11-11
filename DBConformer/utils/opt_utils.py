'''
Optional improvements on top of the DBConformer baseline training loop.
Every option is OFF by default; with all defaults fit()/predict_proba() reproduce the baseline behaviour
(train for max_epoch, report target accuracy of the last epoch).

  --val {none,trials,subject}  validation split carved from SOURCE data only, best-validation checkpoint is restored
  --aug shift,noise,chdrop,segrec   training-batch augmentation (any subset, comma separated)
  --ens N                      average softmax of N independently seeded models
  --tta {none,shift}           test-time augmentation (average softmax over time-shifted copies)
Training-setup options (label smoothing, EMA/SWA, LR schedule, weight decay, grad clip, --tf_drop) live in
utils/train_setup.py and are applied inside fit(); with --val the EMA/SWA weights are what gets validated.

No target label is used for any decision (checkpoint selection uses source validation data only).
'''
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils.device import DEVICE
from utils.train_setup import set_lr, WeightAverager
from utils.utils import cal_acc_comb

AUG_CHOICES = ['shift', 'noise', 'chdrop', 'segrec']


def add_opt_args(parser):
    parser.add_argument('--val', default='none', choices=['none', 'trials', 'subject'],
                        help='validation split from source data: trials = stratified random trials, '
                             'subject = hold out one whole source subject (LOSO only)')
    parser.add_argument('--val_frac', type=float, default=0.15, help='fraction of source trials held out (--val trials)')
    parser.add_argument('--aug', default='', help='comma list from: ' + ','.join(AUG_CHOICES))
    parser.add_argument('--aug_p', type=float, default=0.5, help='per-sample probability of applying each augmentation')
    parser.add_argument('--shift_max', type=float, default=0.1, help='max time shift as fraction of trial length')
    parser.add_argument('--noise_std', type=float, default=0.1, help='gaussian noise std relative to per-channel std')
    parser.add_argument('--chdrop_p', type=float, default=0.1, help='per-channel drop probability')
    parser.add_argument('--seg_n', type=int, default=8, help='number of segments for segmentation-and-recombination')
    parser.add_argument('--ens', type=int, default=1, help='number of independently seeded models to average')
    parser.add_argument('--tta', default='none', choices=['none', 'shift'])
    parser.add_argument('--tta_shift', type=float, default=0.05, help='TTA shift as fraction of trial length (views: -s, 0, +s)')
    return parser


def apply_opt_args(args, cli):
    '''copy parsed options into args and return the method-name tag ('' for the baseline)'''
    args.val, args.val_frac = cli.val, cli.val_frac
    args.aug_list = [a for a in cli.aug.split(',') if a]
    for a in args.aug_list:
        assert a in AUG_CHOICES, 'unknown augmentation ' + a
    args.aug_p, args.shift_max, args.noise_std = cli.aug_p, cli.shift_max, cli.noise_std
    args.chdrop_p, args.seg_n = cli.chdrop_p, cli.seg_n
    args.ens, args.tta, args.tta_shift = cli.ens, cli.tta, cli.tta_shift
    tag = ''
    if args.val != 'none':
        tag += '-val' + ('subj' if args.val == 'subject' else '')
    if args.aug_list:
        tag += '-aug.' + '.'.join(args.aug_list)
    if args.ens > 1:
        tag += '-ens' + str(args.ens)
    if args.tta != 'none':
        tag += '-tta'
    args.tag = tag
    return tag


# ---------------------------------------------------------------- augmentation (x: B x C x T, on DEVICE)
def _shift(x, s):
    B, C, T = x.shape
    idx = torch.arange(T, device=x.device)[None, :] + s[:, None]
    valid = ((idx >= 0) & (idx < T)).to(x.dtype)[:, None, :]
    out = x.gather(2, idx.clamp(0, T - 1)[:, None, :].expand(B, C, T))
    return out * valid  # zero padding


def augment(x, y, args):
    B, C, T = x.shape
    dev = x.device
    p = args.aug_p
    if 'segrec' in args.aug_list:  # segmentation and recombination within class
        bounds = np.linspace(0, T, args.seg_n + 1).astype(int)
        out = x.clone()
        for c in torch.unique(y):
            ids = torch.where(y == c)[0]
            if len(ids) < 2:
                continue
            use = torch.rand(len(ids), device=dev) < p
            for j in range(args.seg_n):
                donor = ids[torch.randint(len(ids), (len(ids),), device=dev)]
                sl = slice(bounds[j], bounds[j + 1])
                seg = torch.where(use[:, None, None], x[donor][:, :, sl], x[ids][:, :, sl])
                out[ids, :, sl] = seg
        x = out
    if 'shift' in args.aug_list:
        m = max(1, int(args.shift_max * T))
        s = torch.randint(-m, m + 1, (B,), device=dev) * (torch.rand(B, device=dev) < p).long()
        x = _shift(x, s)
    if 'noise' in args.aug_list:
        sd = x.std(dim=2, keepdim=True)
        on = (torch.rand(B, 1, 1, device=dev) < p).to(x.dtype)
        x = x + on * args.noise_std * sd * torch.randn_like(x)
    if 'chdrop' in args.aug_list:
        keep = torch.rand(B, C, 1, device=dev) >= args.chdrop_p
        keep = keep | (torch.rand(B, 1, 1, device=dev) >= p) | (keep.sum(1, keepdim=True) == 0)  # never drop all
        x = x * keep.to(x.dtype)
    return x


# ---------------------------------------------------------------- validation split (source data only)
def make_val_split(dset_loaders, args, member):
    full = dset_loaders.setdefault('source_full', dset_loaders['source'])  # unsplit source, reused by every member
    Xs, Ys = full.dataset.tensors
    n = len(Ys)
    rs = np.random.RandomState(args.SEED * 1000 + member * 10 + args.idt)  # independent of the global RNG
    if args.val == 'subject':
        assert not args.trials_arr and args.__dict__.get('scenario', 'LOSO') == 'LOSO', \
            '--val subject needs equal-size source subjects (LOSO, not Zhou2016)'
        per = n // (args.N - 1)
        k = rs.randint(args.N - 1)
        val_mask = np.zeros(n, dtype=bool)
        val_mask[k * per:(k + 1) * per] = True
    else:
        y = Ys.cpu().numpy()
        val_mask = np.zeros(n, dtype=bool)
        for c in np.unique(y):
            ids = rs.permutation(np.where(y == c)[0])
            val_mask[ids[:max(1, int(round(args.val_frac * len(ids))))]] = True
    tr_i = torch.from_numpy(np.where(~val_mask)[0]).to(Xs.device)
    va_i = torch.from_numpy(np.where(val_mask)[0]).to(Xs.device)
    old = full
    dset_loaders['source'] = Data.DataLoader(Data.TensorDataset(Xs[tr_i], Ys[tr_i]), batch_size=old.batch_size,
                                             shuffle=True, drop_last=old.drop_last)
    return Xs[va_i], Ys[va_i]


# ---------------------------------------------------------------- inference
def _fwd(model, x, args):
    if args.backbone == 'ADFCNN' or 'Conformer' in args.backbone:
        x = x.unsqueeze(1)  # B x 1 x C x T (same as unsqueeze(3)+permute in the baseline)
    return model(x)[1]


def predict_proba(model, X, args, tta='none'):
    '''softmax probabilities (N x classes, cpu) for X (N x C x T)'''
    model.eval()
    T = X.shape[2]
    if tta == 'shift':
        s = max(1, int(args.tta_shift * T))
        offsets = [-s, 0, s]
    else:
        offsets = [0]
    out = []
    with torch.no_grad():
        for i in range(0, len(X), args.batch_size * 3):
            xb = X[i:i + args.batch_size * 3].to(DEVICE)
            p = 0
            for o in offsets:
                xv = xb if o == 0 else _shift(xb, torch.full((len(xb),), o, device=xb.device, dtype=torch.long))
                p = p + nn.Softmax(dim=1)(_fwd(model, xv, args).float().cpu())
            out.append(p / len(offsets))
    return torch.cat(out, 0)


def acc_from_probs(probs, y, args):
    pred = probs.argmax(1).numpy()
    y = y.cpu().numpy()
    acc = balanced_accuracy_score(y, pred) if args.paradigm == 'ERP' else accuracy_score(y, pred)
    return acc * 100


# ---------------------------------------------------------------- training loop
def fit(args, dset_loaders, base_network, optimizer_f, optimizer_c, criterion, member=0):
    '''Baseline loop + optional validation/checkpoint selection + augmentation.
    Returns the last-epoch target accuracy (baseline path) or None when --val is on.'''
    use_val = args.val != 'none'
    if use_val:
        Xv, Yv = make_val_split(dset_loaders, args, member)
        print('source train/val trials:', len(dset_loaders['source'].dataset), len(Xv))
    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    best = None  # (val acc, -val loss, epoch, state)
    averager = WeightAverager(base_network, args, args.max_epoch) if (args.ema_decay > 0 or args.swa_epochs > 0) else None
    eval_net = lambda: averager.model if (averager is not None and averager.active) else base_network
    base_network.train()
    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = next(iter_source)
        if inputs_source.size(0) == 1:
            continue
        for opt in (optimizer_f, optimizer_c):
            if opt is not None:
                set_lr(opt, args, iter_num, max_iter, len(dset_loaders["source"]))
        iter_num += 1
        if args.aug_list:
            inputs_source = augment(inputs_source, labels_source, args)
        if 'ADFCNN' in args.backbone or 'Conformer' in args.backbone or args.backbone == 'DBConformer':
            inputs_source = inputs_source.unsqueeze_(3)
            inputs_source = inputs_source.permute(0, 3, 1, 2)
        features_source, outputs_source = base_network(inputs_source)
        classifier_loss = criterion(outputs_source, labels_source)
        optimizer_f.zero_grad()
        if optimizer_c is not None:
            optimizer_c.zero_grad()
        classifier_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(base_network.parameters(), args.grad_clip)
        optimizer_f.step()
        if optimizer_c is not None:
            optimizer_c.step()
        if averager is not None and averager.ema:
            averager.step_ema(base_network)
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            ep = int(iter_num // len(dset_loaders["source"]))
            if averager is not None and not averager.ema:
                base_network.eval()
                averager.epoch_end_swa(base_network, ep)
            if use_val:
                pv = predict_proba(eval_net(), Xv, args)
                va = acc_from_probs(pv, Yv, args)
                vl = float(nn.functional.nll_loss(torch.log(pv + 1e-8), Yv.cpu()))
                if best is None or (va, -vl) > (best[0], best[1]):
                    best = (va, -vl, ep, {k: v.detach().clone() for k, v in eval_net().state_dict().items()})
                args.log.record('Task: {}, Iter:{}/{}; Val Acc = {:.2f}%, Val Loss = {:.4f}'.format(
                    args.task_str, ep, args.max_epoch, va, vl))
            else:
                base_network.eval()
                acc_t_te, _ = cal_acc_comb(dset_loaders["target-online"], eval_net(), args=args)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, ep, int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
            base_network.train()
    if use_val:
        base_network.load_state_dict(best[3])
        args.log.record('Task: {}, restored best-val epoch {} (val acc {:.2f}%)'.format(args.task_str, best[2], best[0]))
        print('restored best-val epoch', best[2], 'val acc {:.2f}'.format(best[0]))
        return None
    if averager is not None and averager.active:  # final prediction/checkpoint use the averaged weights
        base_network.load_state_dict(averager.model.state_dict())
    return acc_t_te
