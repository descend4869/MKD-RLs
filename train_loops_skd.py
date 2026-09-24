"""Training loops for single-teacher DKD and DIST."""

import time

import torch
import torch.nn.functional as F

from utils import AverageMeter, DIST, adjust_lr, correct_num


def _reduce_logit(logit, mask):
    """Collapse a class subset into one logit, while retaining its mass."""
    return logit.masked_fill(~mask, float("-inf")).logsumexp(dim=1, keepdim=True)


def _dkd_loss(logits_s, logits_t, target, alpha, beta, temperature):
    """Decoupled KD loss from CVPR 2022, returned per sample."""
    n, classes = logits_s.shape
    target_mask = torch.zeros_like(logits_s, dtype=torch.bool)
    target_mask.scatter_(1, target.unsqueeze(1), True)
    non_target_mask = ~target_mask

    # TCKD: binary target-vs-rest distributions.
    s_binary = torch.cat((
        logits_s.gather(1, target.unsqueeze(1)),
        _reduce_logit(logits_s, non_target_mask)), dim=1)
    t_binary = torch.cat((
        logits_t.gather(1, target.unsqueeze(1)),
        _reduce_logit(logits_t, non_target_mask)), dim=1)
    tckd = F.kl_div(
        F.log_softmax(s_binary / temperature, dim=1),
        F.softmax(t_binary / temperature, dim=1),
        reduction="none").sum(1) * temperature ** 2

    # NCKD: distribution over non-target classes only.  The target logit is
    # masked in both networks, hence no target-class knowledge is repeated.
    # A finite sentinel avoids 0 * (-inf) in some PyTorch KLDiv versions.
    s_non_target = logits_s.masked_fill(target_mask, -1e9)
    t_non_target = logits_t.masked_fill(target_mask, -1e9)
    nckd = F.kl_div(
        F.log_softmax(s_non_target / temperature, dim=1),
        F.softmax(t_non_target / temperature, dim=1),
        reduction="none").sum(1) * temperature ** 2
    return alpha * tckd + beta * nckd


def _train_single(train_loader, model, criterion_ce, optimizer, epoch, device,
                  args, teacher_model, method):
    meters = [AverageMeter(name, ':.4e') for name in
              ('train_loss', 'train_loss_cls', 'train_loss_kd')]
    train_loss, train_loss_cls, train_loss_kd = meters
    top1_num, top5_num, total = 0, 0, 0
    lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        _, logits = model(inputs, is_feat=True)
        with torch.no_grad():
            _, teacher_logits = teacher_model(inputs, is_feat=True)

        loss_cls = criterion_ce(logits, targets)
        if method == 'dkd':
            loss_kd_each = _dkd_loss(logits, teacher_logits, targets,
                                     args.dkd_alpha, args.dkd_beta, args.kd_T)
        else:
            inter_loss, intra_loss = DIST(tau=args.kd_T)(logits, teacher_logits)
            # utils.DIST returns per-sample inter/intra losses.
            loss_kd_each = inter_loss * args.dist_beta + intra_loss * args.dist_gamma
        loss_kd = loss_kd_each.mean()
        loss = loss_cls + args.kd_weight * loss_kd
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        train_loss.update(loss.item(), bs)
        train_loss_cls.update(loss_cls.item(), bs)
        train_loss_kd.update(loss_kd.item(), bs)
        top1, top5 = correct_num(logits, targets, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += bs
        if args.rank == 0:
            print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, '
                  'CLS Loss:{:.2f}, KD Loss:{:.2f}, Top-1 Acc:{:.2f}'.format(
                      epoch, batch_idx, len(train_loader), lr,
                      time.time() - start_time, train_loss_cls.avg,
                      train_loss_kd.avg, (top1_num / total * 100.).item()))

    acc1, acc5 = top1_num / total, top5_num / total
    if args.rank == 0:
        args.logger.info('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}\n'
                         ' Train_loss:{:.5f}\t Train_loss_cls:{:.5f}'
                         '\t Train_loss_kd:{:.5f}\nTrain top-1 accuracy:{:.2f}'
                         .format(epoch, lr, time.time() - start_time,
                                 train_loss.avg, train_loss_cls.avg,
                                 train_loss_kd.avg, acc1 * 100.))
    return acc1, acc5


def train_dkd(train_loader, model, criterion_list, optimizer, epoch, device,
              args, teacher_model):
    return _train_single(train_loader, model, criterion_list[0], optimizer,
                         epoch, device, args, teacher_model, 'dkd')


def train_dist(train_loader, model, criterion_list, optimizer, epoch, device,
               args, teacher_model):
    return _train_single(train_loader, model, criterion_list[0], optimizer,
                         epoch, device, args, teacher_model, 'dist')
