import torch
import os
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from bisect import bisect_right
import logging


__all__ = ['cal_param_size', 'cal_multi_adds', 'get_data_folder', 'CrossEntropyLoss_label_smooth', 
           'adjust_lr', 'DistillKL', 'DIST', 'set_logger']


def cal_param_size(model):
    return sum([i.numel() for i in model.parameters()])


count_ops = 0
def measure_layer(layer, x, multi_add=1):
    delta_ops = 0
    type_name = str(layer)[:str(layer).find('(')].strip()
    # print(type_name)
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) //
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) //
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w // layer.groups * multi_add

    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        #bias_ops = layer.bias.numel()
        delta_ops = weight_ops + 0#bias_ops

    global count_ops
    count_ops += delta_ops
    return

def is_leaf(module):
    return sum(1 for x in module.children()) == 0


def should_measure(module):
    if is_leaf(module):
        return True
    return False

def cal_multi_adds(model, shape=(1,3,32,32)):
    global count_ops
    count_ops = 0
    data = torch.zeros(shape)

    def new_forward(m):
        def lambda_forward(x):
            measure_layer(m, x)
            return m.old_forward(x)
        return lambda_forward

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, unreduce=False):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        if unreduce: 
            loss = (nn.KLDivLoss(reduction='none')(p_s, p_t) * (self.T**2)).sum(-1)
        else:
            loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss



def adjust_lr(optimizer, epoch, args):
    cur_lr = 0.
    if args.lr_type == 'multistep':
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)
    elif args.lr_type == 'cosine':
        cur_lr = args.init_lr * 0.5 * (1. + math.cos(np.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

        return cur_lr
    
def adjust_lr_vit(optimizer, epoch, args):
    cur_lr = 0.

    # Warm-up phase
    if epoch < args.warmup_epochs:
        cur_lr = args.init_lr * (epoch + 1) / args.warmup_epochs
    else:
        # Learning rate adjustment after warm-up
        if args.lr_type == 'multistep':
            cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)
        elif args.lr_type == 'cosine':
            cur_lr = args.init_lr * 0.5 * (1. + math.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    # Update optimizer learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

    return cur_lr



def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t)


def intra_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s.transpose(0, 1), y_t.transpose(0, 1)).mean()


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return inter_loss, intra_loss


def set_logger(filename, name='experiments'):
    logger = logging.getLogger(name) 
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')


    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter) 

    console = logging.StreamHandler() 
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    
    return logger

def corrupt_teacher_supervision(
        teacher_logits,
        teacher_features,
        teacher_embeddings,
        corruption_rate,
        teacher_idx=0):
    """
    Replace part of one teacher's supervision with another sample's supervision. 
    """

    if corruption_rate <= 0:
        return

    if not 0.0 <= corruption_rate <= 1.0:
        raise ValueError('corruption_rate must be in [0, 1]')

    if not 0 <= teacher_idx < len(teacher_logits):
        raise IndexError('teacher_idx is out of range')

    batch_size = teacher_logits[teacher_idx].size(0)
    if batch_size < 2:
        return

    device = teacher_logits[teacher_idx].device

    # Select the samples whose teacher supervision will be corrupted.
    corrupt_mask = torch.rand(batch_size, device=device) < corruption_rate
    if not corrupt_mask.any():
        return

    # A random non-zero cyclic shift guarantees donor_index[i] != i.
    shift = torch.randint(1, batch_size, (), device=device)
    donor_index = (
        torch.arange(batch_size, device=device) + shift
    ) % batch_size

    def replace_samples(tensor_list):
        original = tensor_list[teacher_idx]
        corrupted = original.clone()
        corrupted[corrupt_mask] = original[donor_index[corrupt_mask]]
        return corrupted

    teacher_logits[teacher_idx] = replace_samples(teacher_logits)
    teacher_features[teacher_idx] = replace_samples(teacher_features)
    teacher_embeddings[teacher_idx] = replace_samples(teacher_embeddings)


def compute_and_plot_feature_similarity_map(model, feat_trans, teacher_models,
                                            data_loader, device, num_classes,
                                            teacher_names, save_path):
    """
    Compute class-wise aligned feature cosine similarities and save a heatmap.
    """
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    modules = [model, feat_trans] + list(teacher_models)
    training_states = [module.training for module in modules]
    for module in modules:
        module.eval()

    similarity_sums = torch.zeros(num_classes, len(teacher_models), device=device)
    class_counts = torch.zeros(num_classes, device=device)

    try:
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                student_features, _ = model(inputs, is_feat=True)
                aligned_student_features = feat_trans(student_features[-2])
                class_counts.index_add_(
                    0, targets, torch.ones_like(targets, dtype=torch.float32))

                for teacher_idx, teacher_model in enumerate(teacher_models):
                    teacher_features, _ = teacher_model(inputs, is_feat=True)
                    similarities = F.cosine_similarity(
                        aligned_student_features[teacher_idx].flatten(1),
                        teacher_features[-2].flatten(1), dim=1, eps=1e-8)
                    similarity_sums[:, teacher_idx].index_add_(0, targets, similarities)
    finally:
        for module, was_training in zip(modules, training_states):
            module.train(was_training)

    similarity_map = (similarity_sums / class_counts.clamp_min(1).unsqueeze(1)).cpu().numpy()
    class_names = getattr(data_loader.dataset, 'classes', None)
    if class_names is None or len(class_names) != num_classes:
        class_names = ['C{}'.format(class_idx) for class_idx in range(num_classes)]
    teacher_labels = [
        'T{} ({})'.format(teacher_idx + 1, teacher_names[teacher_idx])
        for teacher_idx in range(len(teacher_models))
    ]

    figure_height = min(max(8, 0.08 * num_classes + 2), 11)
    figure, axis = plt.subplots(
        figsize=(max(6.5, 1.6 * len(teacher_models)), figure_height))
    image = axis.imshow(similarity_map, aspect='auto', cmap='coolwarm', vmin=0.7, vmax=0.95)
    axis.set_xlabel('Teacher')
    axis.set_ylabel('Class')
    axis.set_xticks(range(len(teacher_models)))
    axis.set_xticklabels(teacher_labels, rotation=30, ha='right')
    axis.set_yticks(range(num_classes))
    axis.set_yticklabels(class_names, fontsize=5)
    axis.set_title('Class-wise student-teacher feature similarity')
    figure.colorbar(image, ax=axis, label='Mean cosine similarity')
    figure.tight_layout()
    figure.savefig(save_path, dpi=300)
    plt.close(figure)

    return similarity_map
