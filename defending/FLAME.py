import numpy as np
import torch
from sklearn.cluster._hdbscan import hdbscan

from defending.BRAVE import anomalous_cluster_detection
from util.param_utils import *


class FLAME():
    def __init__(self, grads, device, args):
        self.device = device
        self.num_clients = len(grads)
        self.args = args
        self.grads = grads

    def detection(self, memo, global_model):  # 动态聚类
        # 计算成对余弦相似度
        cos_list = []
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        for i in range(self.num_clients):
            cos_i = []
            for j in range(self.num_clients):
                cos_ij = 1 - cos(self.grads[i], self.grads[j])
                cos_i.append(cos_ij.item())
            cos_list.append(cos_i)
        # 聚类
        cluster = hdbscan.HDBSCAN(min_cluster_size=self.num_clients // 2 + 1, min_samples=1,
                                  allow_single_cluster=True).fit(cos_list)
        labels = cluster.labels_
        # 判断恶意客户端：最大簇
        benign_client, malicious_client = [], []
        self.norm_list = np.array([])
        max_cluster_index = 0
        if self.args.idea1:
            models = []
            for g in self.grads:
                models.append(update_to_model(global_model, g))
            max_cluster_index = anomalous_cluster_detection(models, memo, labels, self.device, self.args)
        else:
            max_num_in_cluster = 0
            if labels.max() < 0:  # 全是良性客户端
                labels = np.zeros_like(labels)
            else:  # 存在恶意客户端
                for index_cluster in range(labels.max() + 1):  # 计算最大簇
                    if len(labels[labels == index_cluster]) > max_num_in_cluster:
                        max_cluster_index = index_cluster
                        max_num_in_cluster = len(labels[labels == index_cluster])

        for i in range(len(labels)):
            if labels[i] == max_cluster_index:
                benign_client.append(i)
                # self.norm_list = np.append(self.norm_list,torch.norm(local_grad_vector[i],p=2))  # consider BN
                self.norm_list = np.append(self.norm_list,
                                           torch.norm(self.grads[i].view(-1), p=2).item())  # no consider BN
            else:
                malicious_client.append(i)
        return malicious_client, benign_client

    def adaptive_clipping(self, benign_grads):  # 动态裁剪:聚合前
        self.clip_value = np.median(self.norm_list)
        for i in range(len(benign_grads)):
            gama = self.clip_value / self.norm_list[i]
            if gama < 1:
                benign_grads[i] *= gama
        return benign_grads

    def adaptive_noising(self, global_weights, noise):  # 适应性噪声：聚合后
        for key, var in global_weights.items():
            temp = copy.deepcopy(var)
            temp = temp.normal_(mean=0, std=noise * self.clip_value)
            var += temp
        return global_weights


def adaptive_clipping(grad_list, args):
    norm_list = np.array([torch.norm(g.view(-1), p=2).item() for g in grad_list])
    clip_value = np.median(norm_list)
    for i in range(len(grad_list)):
        gama = clip_value / norm_list[i]
        if gama < 1:
            grad_list[i] *= gama
    return grad_list
