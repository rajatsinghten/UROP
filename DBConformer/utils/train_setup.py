'''
Optional training-setup options (all default to the exact baseline behaviour).
  --label_smoothing, --ema_decay / --swa_epochs, --lr_sched cosine|repo (+ --warmup_epochs),
  --weight_decay (--optimizer adamw), --grad_clip, --tf_drop (EXPLORATORY architecture tweak).
No test labels are used by any option: averaged weights are simply evaluated at the end.
'''
import math
import copy
import torch


def add_train_args(parser):
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--ema_decay', type=float, default=0.0, help='0 = off, e.g. 0.999')
    parser.add_argument('--swa_epochs', type=int, default=0, help='0 = off; average weights over last N epochs')
    parser.add_argument('--lr_sched', default='none', choices=['none', 'cosine', 'repo'])
    parser.add_argument('--warmup_epochs', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--grad_clip', type=float, default=0.0, help='max grad norm, 0 = off')
    parser.add_argument('--tf_drop', type=float, default=0.5, help='EXPLORATORY: transformer dropout (baseline 0.5)')
    return parser


def apply_extra(args, cli):
    """copy the training-setup options into args and return the method-name tag ('' for the baseline)"""
    if cli.ema_decay > 0 and cli.swa_epochs > 0:
        raise SystemExit('use either --ema_decay or --swa_epochs, not both')
    for k in ('label_smoothing', 'ema_decay', 'swa_epochs', 'lr_sched', 'warmup_epochs',
              'weight_decay', 'optimizer', 'grad_clip', 'tf_drop'):
        setattr(args, k, getattr(cli, k))
    tag = ''
    if cli.label_smoothing > 0: tag += '-ls' + str(cli.label_smoothing)
    if cli.ema_decay > 0: tag += '-ema' + str(cli.ema_decay)
    if cli.swa_epochs > 0: tag += '-swa' + str(cli.swa_epochs)
    if cli.lr_sched != 'none': tag += '-' + cli.lr_sched + ('w' + str(cli.warmup_epochs) if cli.warmup_epochs > 0 else '')
    if cli.weight_decay > 0: tag += '-' + cli.optimizer + 'wd' + str(cli.weight_decay)
    if cli.grad_clip > 0: tag += '-gc' + str(cli.grad_clip)
    if cli.tf_drop != 0.5: tag += '-tfd' + str(cli.tf_drop)
    return tag


def make_optimizer(params, args):
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)


def set_lr(optimizer, args, iter_num, max_iter, iters_per_epoch):
    '''iter_num: 0-based index of the step about to be taken.'''
    if args.lr_sched == 'none':
        return
    warm = int(args.warmup_epochs * iters_per_epoch)
    if iter_num < warm:
        f = (iter_num + 1) / warm
    elif args.lr_sched == 'cosine':
        f = 0.5 * (1 + math.cos(math.pi * (iter_num - warm) / max(1, max_iter - warm)))
    else:  # 'repo': decay of utils.utils.lr_scheduler (gamma=10, power=0.75) without its SGD side effects
        f = (1 + 10 * iter_num / max_iter) ** (-0.75)
    for g in optimizer.param_groups:
        g['lr'] = args.lr * f


class WeightAverager:
    '''EMA (ema_decay>0) or uniform SWA over the last swa_epochs epochs. Buffers (BN stats) are averaged too.'''
    def __init__(self, model, args, max_epoch):
        self.ema = args.ema_decay > 0
        self.decay = args.ema_decay
        self.swa_start = max_epoch - args.swa_epochs  # epochs completed before averaging starts
        self.model = copy.deepcopy(model).eval()
        self.n = 0
        self.active = self.ema

    @torch.no_grad()
    def step_ema(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))  # warm-up so early EMA is not dominated by init
        for pe, p in zip(self.model.state_dict().values(), model.state_dict().values()):
            if pe.dtype.is_floating_point:
                pe.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                pe.copy_(p)

    @torch.no_grad()
    def epoch_end_swa(self, model, epoch):
        if epoch <= self.swa_start:
            return
        self.active = True
        self.n += 1
        for pe, p in zip(self.model.state_dict().values(), model.state_dict().values()):
            if pe.dtype.is_floating_point:
                pe.add_((p.detach() - pe) / self.n)  # running mean
            else:
                pe.copy_(p)
