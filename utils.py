import copy
import torch
import torch.nn as nn
import math

def check_cuda(use_gpu, gpu_id):
    """ Check CUDA availability """
    device = torch.device('cpu')
    use_cuda = use_gpu and gpu_id >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda:%d' % gpu_id)
    return device


def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for _ in range(k)
    )


def src_dot_dst(src_field, dst_field, out_field, alpha_val=None):
    def func(edges):
        scores = edges.src[src_field] * edges.dst[dst_field]
        if alpha_val is not None:
            u, v, eid = edges.edges()
            alpha_multiplier = edges.src[alpha_val][u,v].unsqueeze(-1).unsqueeze(-1).expand(scores.size()[0],
                                                                                            scores.size()[1],
                                                                                            scores.size()[2])
            return {out_field: scores * alpha_multiplier}
        else:
            return {out_field: scores}
    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func


def u_copy_v(src_field, out_field):
    def func(edges):
        return {out_field: edges.src[src_field]}
    return func


# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func


def exp_real(field, out_field='score_soft', L=1e-5):
    def func(edges):
        # clamp for softmax numerical stability
        return {out_field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func


def exp_fake(field, out_field='score_soft', L=1e-5):
    def func(edges):
        # clamp for softmax numerical stability
        return {out_field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func


def exp(field, out_field='score_soft', L=1e-5):
    def func(edges):
        # clamp for softmax numerical stability
        return {out_field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func



class EpsilonGreedyStrategy(object):
    """
    Class implementing the epsilon greedy strategy with exponential decay.
    """
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        """
        Computes the current exploration rate.

        Parameters
        ----------
        current_step : int
            current step

        Returns
        -------
        float
            exploration rate
        """
        return self.end + (self.start - self.end) * math.exp(-1 * current_step * self.decay)


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
        #fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} avg: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)