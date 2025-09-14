import numpy as np
import torch


def distribute_dataset(train_dataset, args, classes_per_client=1, samples_per_class=582):
    num_users = args.num_users
    dd_type = args.data_distribution
    alpha = args.alpha
    seed = {"numpy":args.random_seed,"torch_cpu":args.manual_seed_cpu,"torch_gpu":args.manual_seed_gpu}
    # 数据分布
    if dd_type == 'IID':
        user_groups = sample_dirichlet(train_dataset, num_users, seed, alpha=100000000)
    elif dd_type == 'NON_IID':
        user_groups = sample_dirichlet(train_dataset, num_users, seed, alpha=alpha)
    elif dd_type == 'EXTREME_NON_IID':
        user_groups = sample_extreme(train_dataset, num_users, classes_per_client, samples_per_class, seed)
    return user_groups


# 迪利克雷分布
def sample_dirichlet(dataset, num_users, seed, alpha=1):
    np.random.seed(seed['numpy'])  # np
    torch.manual_seed(seed['torch_cpu'])  # cpu
    torch.cuda.manual_seed(seed['torch_gpu'])  # gpu

    classes = {}  # 每一类的数据
    for idx, x in enumerate(dataset):  # 将数据按label进行划分
        _, label = x
        if type(label) == torch.Tensor:
            label = label.item()
        if label in classes:
            classes[label].append(idx)
        else:
            classes[label] = [idx]
    num_classes = len(classes.keys())

    user_groups = {i: [] for i in range(num_users)}  # 客户端拥有的数据
    for n in range(num_classes):  # 遍历每一类
        np.random.shuffle(classes[n])
        class_size = len(classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(num_users * [alpha]))  # 每个客户端可以拥有的数据比例
        for user in range(num_users):  # 遍历每个客户端
            num_imgs = int(round(sampled_probabilities[user]))  # round：四舍五入
            if num_imgs > 0:
                user_groups[user].extend(classes[n][:num_imgs])
            classes[n] = classes[n][min(len(classes[n]), num_imgs):]  # 去除已经分配的数据

    for user in range(num_users):
        user_groups[user] = set(user_groups[user])
    return user_groups


# 极端分布不均的情况
def sample_extreme(dataset, num_users, classes_per_client, samples_per_class, seed):
    np.random.seed(seed['numpy'])  # np
    torch.manual_seed(seed['torch_cpu'])  # cpu
    torch.cuda.manual_seed(seed['torch_gpu'])  # gpu

    num_classes = len(dataset.classes)
    if num_users > int(num_classes / classes_per_client):
        exit('num_users > num_classes')
    n = len(dataset)
    user_groups = {i: [] for i in range(num_users)}
    idxs = np.arange(n)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    label_indices = {l: [] for l in range(num_classes)}  # 记录每一类的数据
    for l in label_indices:
        label_idxs = np.where(labels == l)
        label_indices[l] = list(idxs[label_idxs])
    labels = [i for i in range(num_classes)]

    for i in range(num_users):  # 遍历每个客户端
        user_labels = np.random.choice(labels, classes_per_client, replace=False)  # 每个客户端拥有哪两类数据
        for l in user_labels:  # 遍历每个类
            lab_idxs = label_indices[l]  # 每一类所拥有的数据量
            # lab_idxs = label_indices[l][:samples_per_class]  # 每一类所拥有的数据量
            user_groups[i].extend(lab_idxs)
            labels.remove(l)  # label_indices[l] = list(set(label_indices[l]) - set(lab_idxs))  # 去除已分配的数据
            # if len(label_indices[l]) < samples_per_class:  # 若剩余数据量不足以分配给另一个客户端，则删除剩余数据
            #     labels = list(set(labels) - set([l]))

    for user in range(num_users):
        user_groups[user] = set(user_groups[user])
    return user_groups
