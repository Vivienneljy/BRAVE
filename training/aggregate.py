import copy

import numpy as np
import sklearn.metrics.pairwise as smp
import torch

'''
不同的聚合方式
'''


def aggregate(grad_list, global_model, args):
    if args.aggregation == 'trim':  # 选取聚合函数
        return trim(grad_list, global_model)
    elif args.aggregation == 'simple_mean':
        return simple_mean(grad_list, global_model)
    elif args.aggregation == 'median':
        return median(grad_list, global_model)
    elif args.aggregation == 'krum':
        return krum(grad_list, global_model)
    else:
        raise NotImplementedError


# FedAvg
def simple_mean(grad_list, w):  # w:model.state_dict()
    mean_nd = torch.mean(torch.concat(grad_list, dim=1), dim=-1, keepdim=True)  # 参数均值：FedAvg
    w_avg = copy.deepcopy(w)
    idx = 0
    for key, value in w_avg.items():
        v_size = value.reshape(-1, 1).size()[0]
        w_avg[key] = value + mean_nd[idx:(idx + v_size)].reshape(value.shape)
        idx += v_size

    return w_avg

# trimmed mean
def trim(grad_list, w, b=0):
    w_trim = copy.deepcopy(w)
    # sort
    sorted_array = torch.sort(torch.concat(grad_list, dim=1), dim=-1).values
    # trim
    n = len(grad_list)
    m = n - b * 2
    trim_nd = torch.mean(sorted_array[:, b:(b + m)], dim=-1, keepdim=True)  # 去头去尾求均值
    # update global model
    idx = 0
    for key, value in w_trim.items():
        v_size = value.reshape(-1, 1).size()[0]
        w_trim[key] = value + trim_nd[idx:(idx + v_size)].reshape(value.shape)
        idx += v_size

    return w_trim


def median(grad_list, w):
    w_median = copy.deepcopy(w)
    if len(grad_list) % 2 == 1:  # 判断奇偶求中位数
        median_nd = torch.concat(grad_list, dim=1).sort(dim=-1).values[:, len(grad_list) // 2]  # 求中位数
    else:
        median_nd = torch.concat(grad_list, dim=1).sort(dim=-1).values[:,
                    len(grad_list) // 2: len(grad_list) // 2 + 1].mean(dim=-1, keepdim=True)
    idx = 0
    for key, value in w_median.items():
        v_size = value.reshape(-1, 1).size()[0]
        w_median[key] = value + median_nd[idx:(idx + v_size)].reshape(value.shape)
        idx += v_size

    return w_median


# 计算每一维度欧氏距离之和
def score(gradient, v, f):
    num_neighbours = v.shape[1] - 2 - f
    sorted_distance = torch.square(v - gradient).sum(dim=0).sort().values
    return torch.sum(sorted_distance[1:(1 + num_neighbours)]).item()


def nearest_distance(gradient, c_p):
    sorted_distance = torch.square(c_p - gradient).sum(dim=1).sort(dim=0).values
    return sorted_distance[1].item()


def krum(grad_list, w, b=0):
    w_krum = copy.deepcopy(w)
    num_params = len(grad_list)
    q = b
    if num_params <= 2:
        # if there are too few clients, randomly pick one as Krum aggregation result
        random_idx = np.random.choice(num_params)
        krum_nd = torch.reshape(grad_list[random_idx], shape=(-1, 1))
    else:
        if num_params - b - 2 <= 0:
            q = num_params - 3
        v = torch.concat(grad_list, dim=1)
        scores = torch.tensor([score(gradient, v, q) for gradient in grad_list])  # 计算每个参数与n-k-2轮全局模型的欧氏距离
        min_idx = int(scores.argmin(dim=0).item())  # 最接近的参数
        krum_nd = torch.reshape(grad_list[min_idx], shape=(-1, 1))  # 选择最接近n-k-2轮全局模型更新的客户端更新作为本轮全局更新

    idx = 0
    for key, value in w_krum.items():
        v_size = value.reshape(-1, 1).size()[0]
        w_krum[key] = value + krum_nd[idx:(idx + v_size)].reshape(value.shape)
        idx += v_size

    return w_krum
