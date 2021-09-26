import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn._reduction as _Reduction
import numpy as np

class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)

class NewLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(NewLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return -(input[:, 0]) + np.log(np.exp(input[:, 0]) + np.exp(input[:, 1]))
        # return F.cross_entropy(input, target, weight=self.weight,
        #                        ignore_index=self.ignore_index, reduction=self.reduction)

class MyNLLLoss(nn.Module):

    def forward(self, pre, y_true, weight=1):
        '''
        对于 利好 判断为 利空的，或 利空判断为 利好的，设置惩罚权重大一点
        :param pre: 经过 logsoftmax之后的输出值
        :param y_true: 真实的标签值
        :param weight: 利好利空的惩罚权重
        :return:
        '''
        size = pre.size(0)
        all_loss = torch.tensor(0.)
        for i in range(size):
            if abs(pre[i].argmax(-1) - y_true[i]) == 2:
                all_loss.add_(weight * pre[i][y_true[i]])
            else:
                all_loss.add_(pre[i][y_true[i]])

        return -all_loss / size