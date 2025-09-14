import copy

import torch
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label % 10)

# label flipping attack
class Label_Flipping_Attack(Dataset):
    def __init__(self, dataset, idxs, source_class=None, target_class=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.source_class = source_class
        self.target_class = target_class

    def __getitem__(self, item):
        image, label = copy.deepcopy(self.dataset[self.idxs[item]])
        if label == self.source_class:
            target = self.target_class
        else:
            target = label
        return torch.as_tensor(image), torch.as_tensor(target)

    def __len__(self):
        return len(self.idxs)

def local_loader(train_dataset, user_groups, args):
    train_loaders, test_loaders = {},{}
    for i in range(args.num_users):
        if i < args.num_atk:
            train_loaders[i], test_loaders[i] = malicious_loader(train_dataset, list(user_groups[i]), args)
        else:
            train_loaders[i], test_loaders[i] = benign_loader(train_dataset, list(user_groups[i]), args)
    return train_loaders, test_loaders

def global_loader(test_dataset, args):
    # benign dataloader
    benign_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    # malicious dataloader
    idxs_test = [i for i in range(len(test_dataset))]
    mal_dataset = Label_Flipping_Attack(test_dataset, idxs_test, args.source_label, args.target_label)
    malicious_dataloader = DataLoader(mal_dataset, batch_size=128, shuffle=False)
    return benign_dataloader, malicious_dataloader

def benign_loader(dataset, idxs, args):
    # split indexes for train, and test (90, 10)
    idxs_train = idxs[:int(0.9 * len(idxs))]
    idxs_test = idxs[int(0.9 * len(idxs)):]
    # get preprocessing
    train_dataset = DatasetSplit(dataset, idxs_train)
    test_dataset = DatasetSplit(dataset, idxs_test)
    # get dataloader
    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
    return trainloader, testloader


def malicious_loader(dataset, idxs, args):
    # split indexes for train, and test (90, 10)
    idxs_train = idxs[:int(0.9 * len(idxs))]
    idxs_test = idxs[int(0.9 * len(idxs)):]
    # data poisoning
    train_dataset = Label_Flipping_Attack(dataset, idxs_train, args.source_label, args.target_label)
    test_dataset = DatasetSplit(dataset, idxs_test)
    # get dataloader
    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
    return trainloader, testloader