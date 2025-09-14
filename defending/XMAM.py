from collections import Counter

import numpy as np
import torch
from sklearn.cluster._hdbscan import hdbscan
from sklearn.decomposition import PCA

from defending.BRAVE import anomalous_cluster_detection
from util.param_utils import update_to_model


class XMAM():
    def __init__(self, grads, device, args):
        self.device = device
        self.args = args
        self.grads = grads
        self.num_clients = len(grads)

    def detection(self, memo, global_model):
        benign_client, malicious_client = [], []
        if self.args.dataset == 'cifar':
            shape = [1, 3, 32, 32]
        elif self.args.dataset == 'mnist':
            shape = [1, 1, 28, 28]
        else:
            raise NotImplementedError()
        x_ray = torch.ones(shape, device=torch.device(self.device))
        client_SLPDs = []
        models = []
        for g in self.grads:
            models.append(update_to_model(global_model, g))
        for net_index, net in enumerate(models):  # 客户端模型对随机矩阵的预测
            net.eval()
            SLPD_now = net(x_ray)
            SLPD_now = SLPD_now.detach().cpu().numpy()
            SLPD_now = SLPD_now[0]
            client_SLPDs.append(SLPD_now)
        client_SLPDs = np.array(client_SLPDs)

        # delete abnormal SLPD value like nan etc
        client_remain = []
        jjj = 0
        for i in range(self.num_clients):
            for j in range(len(client_SLPDs[i])):
                jjj = j
                if np.isnan(client_SLPDs[i][j]):
                    break
            if jjj == len(client_SLPDs[i]) - 1:
                client_remain.append(i)
            else:
                malicious_client.append(i)
        client_num_remain = len(client_remain)

        # cluster SLPDs
        pca = PCA(n_components=3)
        X_new = pca.fit_transform(client_SLPDs[client_remain])
        cluster = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster_labels = cluster.fit_predict(X_new)

        if self.args.idea1:
            majority = anomalous_cluster_detection(models, memo, cluster_labels, self.device, self.args)
        else:
            majority = Counter(cluster_labels)
            majority = majority.most_common()[0][0]

        # select benign client
        for i in range(client_num_remain):
            if cluster_labels[i] == majority:
                benign_client.append(client_remain[i])
            else:
                malicious_client.append(client_remain[i])

        return malicious_client, benign_client
