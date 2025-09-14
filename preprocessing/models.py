import torch.nn.functional as F
from torch import nn


def get_model(args):
    if args.dataset == 'mnist':
        model = CNNMnist(args=args)
    elif args.dataset == 'cifar':
        model = CNNCifar(args=args)
    else:
        raise NotImplementedError
    return model

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.feature_fc1 = None
        self.feature_fc2 = None
        self.feature_fc3 = None
        # self.feature_fc1_graph = None
        # self.feature_fc2_graph = None
        # self.feature_fc3_graph = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        # self.feature_fc1_graph = x
        self.feature_fc1 = x.cpu().detach().numpy()
        x = F.relu(self.fc1(x))
        # self.feature_fc2_graph = x
        self.feature_fc2 = x.cpu().detach().numpy()
        x = F.relu(self.fc2(x))
        # self.feature_fc3_graph = x
        self.feature_fc3 = x.cpu().detach().numpy()
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.feature_fc1 = None
        self.feature_fc2 = None

    def forward(self, x):
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        self.feature_fc1 = x.cpu().detach().numpy()
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        self.feature_fc2 = x.cpu().detach().numpy()
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
