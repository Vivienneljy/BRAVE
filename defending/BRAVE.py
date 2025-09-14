import os
from torchvision import datasets, transforms
import numpy as np
import sklearn.metrics.pairwise as smp
import torch
from sklearn.cluster import MiniBatchKMeans
from util.param_utils import update_to_model


class BRAVE():
    def __init__(self, grads, device, args):
        self.device = device
        self.num_client = len(grads)
        self.grads = grads
        self.args = args

    def detection(self, memo, global_model):
        # cluster
        grad_list = [item.view(-1).cpu().numpy() for item in self.grads]
        cs = 1 / (1 + smp.euclidean_distances(grad_list))
        cluster = MiniBatchKMeans(n_clusters=2).fit(cs)
        labels = cluster.predict(cs)
        clu1 = np.where(np.array(labels) == 0)[0]
        clu2 = np.where(np.array(labels) == 1)[0]

        good_cl = 0
        if self.args.idea1:
            models = []
            for g in self.grads:
                models.append(update_to_model(global_model, g))
            good_cl = anomalous_cluster_detection(models, memo, labels, self.device, self.args)
        else:
            if len(clu1) < len(clu2):
                good_cl = 1

        if good_cl == 0:
            mal, ben = clu2, clu1
        else:
            mal, ben = clu1, clu2

        return mal, ben

def anomalous_cluster_detection(models, memo, labels, device, args):
    # load auxiliay data
    if os.path.exists('./{}/vitual_data.pt'.format(args.log)):
        virtual_data = torch.load('./{}/vitual_data.pt'.format(args.log))
    else:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.manual_seed_cpu)  # cpu
        torch.cuda.manual_seed(args.manual_seed_gpu)  # gpu

        if args.dataset == 'cifar':
            mean, std = 0.4734, 0.2507
            shape = [args.auxiliary_data_size, 3, 32, 32]
        elif args.dataset == 'mnist':
            mean, std = 0.1307, 0.3081
            shape = [args.auxiliary_data_size, 1, 28, 28]
        else:
            raise NotImplementedError()

        if args.auxiliary_data == 'ones':  # 全1矩阵
            virtual_data = torch.ones(size = shape, device=device)
        elif args.auxiliary_data == 'random':  # 随机矩阵
            virtual_data = torch.normal(mean=mean, std=std, size=shape, device=device)
        elif args.auxiliary_data == 'real':  # 相同形状的真实数据
            transform = transforms.Compose([transforms.ToTensor()])
            if args.dataset == 'cifar':  # cifar10 --> cifar100
                data = datasets.CIFAR100(args.auxiliary_data_path, train=False, download=True, transform=transform)
            elif args.dataset == 'mnist':  # mnist --> fashionmnist
                data = datasets.FashionMNIST(args.auxiliary_data_path, train=False, download=True, transform=transform)
            else:
                raise NotImplementedError()
            choose_data = np.random.choice(len(data), args.auxiliary_data_size, replace=False)
            virtual_data = torch.stack([data[i][0] for i in choose_data]).to(device)
        else:
            raise NotImplementedError()
        torch.save(virtual_data, './{}/vitual_data.pt'.format(args.log))

    # anomaly detection
    outputs = []
    var_min, good_cl = 1000, 0
    for i, model in enumerate(models):
        model.eval()
        output = model(virtual_data)
        outputs.append(np.exp(output.detach().cpu().numpy()))
    vars = np.mean(np.var(outputs,axis=2),axis=1)
    memo.print_brave_anormaly_score(vars)

    for index_cluster in set(labels):
        if index_cluster == 2:
            continue
        mean = np.mean(vars[np.where(labels == index_cluster)])
        if mean < var_min:
            good_cl = index_cluster
            var_min = mean

    return good_cl