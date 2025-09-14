import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from defending.BRAVE import anomalous_cluster_detection
from util.param_utils import update_to_model


class DPFLA:
    def __init__(self, grads, device, args):
        self.device = device
        self.num_client = len(grads)
        self.grads = grads
        self.args = args

    def detection(self, memo, global_model):
        if self.args.dataset == 'mnist':
            p = 50
        elif self.args.dataset == 'cifar':
            p = 84
        P = generate_orthogonal_matrix(n=p)
        W = generate_orthogonal_matrix(n=self.num_client * self.num_client)
        Ws = [W[:, e * self.num_client: e * self.num_client + self.num_client][0, :].reshape(-1, 1) for e in range(self.num_client)]

        param_diff_mask = []
        detect_res_list = []
        start_idx = -10 * (p + 1)

        for idx in range(10):  # 对于每一个神经元
            # 计算每个本地模型的权重与全局模型最后一层权重之间的梯度差 (每一维度上的欧式距离)
            for i in range(self.num_client):
                gradient = self.grads[i][start_idx : start_idx + p].cpu().numpy()
                X_mask = Ws[i] @ gradient.reshape(1, -1) @ P  # 矩阵乘法
                param_diff_mask.append(X_mask)

            Z_mask = sum(param_diff_mask)
            U_mask, sigma, VT_mask = svd(Z_mask)

            G = Ws[0]
            for idx in range(self.num_client):  # 拼接所有客户端的W矩阵
                if idx == 0:
                    continue
                G = np.concatenate((G, Ws[idx]), axis=1)

            U = np.linalg.inv(G) @ U_mask
            U = U[:, :2]
            res = U * sigma[:2]
            detect_res_list.append(res)
            start_idx += p

        coefficient_list, score_list = batch_detect_outliers_kmeans(detect_res_list)

        max_sc = max(coefficient_list)
        max_sc_idx = coefficient_list.index(max_sc)
        scores = score_list[max_sc_idx] if max_sc >= 0.70 else np.ones(self.num_client, dtype=int)

        # 返回得分列表
        majority = 1
        if self.args.idea1 and max_sc >= 0.70:
            models = []
            for g in self.grads:
                models.append(update_to_model(global_model, g))
            majority = anomalous_cluster_detection(models, memo, scores, self.device, self.args)
        mal = np.where(scores == 1-majority)[0]
        ben = np.where(scores == majority)[0]
        return mal, ben


def generate_orthogonal_matrix(n):
    q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')  # 对随机矩阵进行正交分解
    return q

def batch_detect_outliers_kmeans(list, n_clusters=2):
    # 初始化K-means模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    coefficient_list = []
    score_list = []

    for data in list:
        # 训练模型
        kmeans.fit(data)
        # 预测聚类标签
        labels = kmeans.predict(data)
        # 计算轮廓系数
        coefficient = silhouette_score(data, labels)
        coefficient_list.append(coefficient)

        scores = labels
        if sum(labels) < len(data) / 2:
            scores = 1 - labels
        else:
            scores = labels

        score_list.append(scores)
    return coefficient_list, score_list


def svd(x):
    m, n = x.shape
    if m >= n:
        return np.linalg.svd(x)
    else:
        u, s, v = np.linalg.svd(x.T)
        return v.T, s, u.T

