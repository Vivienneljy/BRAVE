import argparse


def args_parser():

    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--aggregation', help="aggregation rule", default='simple_mean', type=str,
                        choices=['simple_mean', 'trim', 'krum', 'median'])
    parser.add_argument('--iteration', type=int, default=300,
                        help="number of rounds of training")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--local_ep', type=int, default=3,
                        help="the number of local epochs: E")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users joined in communication: K")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--data_distribution', type=str, default='IID', choices=['IID', 'NON_IID', 'EXTREME_NON_IID'],
                        help='client data distribution')  # IID
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='NON_IID setting:The larger alpha,the smoother the data distribution')  # 1.0

    # model arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # 0.01
    parser.add_argument('--gpu', default=True, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")  # cuda:0
    parser.add_argument('--first_time', type=int, default=1, help='first time to communicate:1(Y)/0(N)')
    parser.add_argument('--checkpoint', type=str, default=r'', help="the path of checkpoint")

    # attack arguments
    parser.add_argument('--num_atk', help="number of attackers,note:num_atk < frac * num_users", default=70, type=int)
    parser.add_argument('--source_label', help='source class (-1: random choose)', default=6, type=int)  # 6
    parser.add_argument('--target_label', help='target class (-1: random choose)', default=2, type=int)  # 2

    # defence arguments
    parser.add_argument('--defense', type=str, default='brave', help="type of defense",
                        choices=['no', 'brave', 'flame', 'xmam', 'lfighter', 'dpfla'])
    parser.add_argument('--auxiliary_data', type=str, default='real', help='auxiliary preprocessing',
                        choices=['real', 'random', 'ones'])  # fmnist cifar100
    parser.add_argument('--auxiliary_data_path', type=str, default=r'D:/Datasets/FashionMNIST', help='auxiliary preprocessing')  # CIFAR100 FashionMNIST
    parser.add_argument('--auxiliary_data_size', type=int, default=100, help='size of auxiliary preprocessing')
    parser.add_argument('--idea1', type=bool, default=True, help="anomalous cluster detection")
    parser.add_argument('--noise', type=float, default=0.001)

    # seed arguments
    parser.add_argument('--random_seed', type=int, default=903, help='random seed')  # 903
    parser.add_argument('--manual_seed_cpu', type=int, default=313, help='random seed of cpu')  # 313
    parser.add_argument('--manual_seed_gpu', type=int, default=322, help='random seed of gpu')  # 322

    # log arguments
    parser.add_argument('--save', type=bool, default=False, help='whether save log and model')
    parser.add_argument('--log', type=str, default='Test_part10_atk70_s6t2_mnist',
                        help='the experiment name of logs')
    parser.add_argument('--print_every', type=int, default=2, help='print performance frequence')

    # data  arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of preprocessing:fmnist or mnist or cifar")
    parser.add_argument('--data_path', type=str, default=r'D:/Datasets/MNIST',
                        help='get preprocessing automatically when None')  # CIFAR10 MNIST

    args = parser.parse_args()
    return args
