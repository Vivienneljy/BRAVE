import numpy as np
import sklearn.metrics.pairwise as smp
from sklearn.cluster import KMeans

from defending.BRAVE import anomalous_cluster_detection
from util.param_utils import *


class LFighter():
    def __init__(self, grads, device, args):
        self.device = device
        self.num_clients = len(grads)
        self.args = args
        self.grads = grads

    def detection(self, memo, global_model):
        # 提取模型参数
        ls = layer_size(global_model)
        dw = [None for i in range(self.num_clients)]  # 记录每个客户端的参数更新
        db = [None for i in range(self.num_clients)]
        for i in range(self.num_clients):  # 计算客户端模型在最后一层的更新
            dw[i] = (self.grads[i][-(ls[-1] + ls[-2]):-ls[-1]]).cpu().numpy().reshape(ls[-1], -1)
            db[i] = (self.grads[i][-ls[-1]:]).view(-1).cpu().numpy()
        dw = np.asarray(dw)
        db = np.asarray(db)

        # 聚类
        if len(db[0]) <= 2:  # 对于二分类，两个label一个是源标签，一个是目标标签
            data = []
            for i in range(self.num_clients):  # 将模型参数展平
                data.append(dw[i].reshape(-1))
        else:  # 对于多分类，将梯度值最高的两个神经元作为潜在源和目标类神经元
            norms = np.linalg.norm(dw, axis=-1)
            num_classes = ls[-1]
            memory = np.zeros([num_classes])
            memory = np.sum(norms, axis=0)
            memory += np.sum(abs(db), axis=0)
            max_two_freq_classes = memory.argsort()[-2:]  # 梯度值最高的两个类
            data = []
            for i in range(self.num_clients):
                data.append(dw[i][max_two_freq_classes].reshape(-1))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)  # 对最后一层参数进行聚类
        labels = kmeans.labels_
        clusters = {0: [], 1: []}

        # 计算可疑分数
        good_cl = 0
        for i, l in enumerate(labels):
            clusters[l].append(data[i])
        if self.args.idea1:
            models = []
            for g in self.grads:
                models.append(update_to_model(global_model, g))
            good_cl = anomalous_cluster_detection(models, memo, labels, self.device, self.args)
        else:
            cs0, cs1 = self.clusters_dissimilarity(clusters)
            if cs0 < cs1:  # 可疑分数小的簇是恶意的
                good_cl = 1

        # 分类
        ben, mal = [], []
        for i in range(len(labels)):
            if labels[i] == good_cl:
                ben.append(i)
            else:
                mal.append(i)
        return mal, ben

    def clusters_dissimilarity(self, clusters):
        n0 = len(clusters[0])
        n1 = len(clusters[1])
        smp.distance_metrics()
        cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)  # 计算成对余弦相似度
        cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
        mincs0 = np.min(cs0, axis=1)  # 计算最大夹角
        mincs1 = np.min(cs1, axis=1)
        ds0 = n0 / self.num_clients * (1 - np.mean(mincs0))  # 利用簇密度计算可疑分数：越密集，越可疑
        ds1 = n1 / self.num_clients * (1 - np.mean(mincs1))
        return ds0, ds1
