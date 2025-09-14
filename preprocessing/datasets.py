from torchvision import datasets, transforms

from preprocessing.sample import distribute_dataset

'''
    split preprocessing into train_dataset and test_dataset, and use test_dataset to test global model, 
    then split train_data into /train/ and /test/, and use /test/ to test local model  
    
    train_dataset: train data
    test_dataset: test data 
    user_groups: dict[user_id]=user_dataset
'''


def get_dataset(args):
    """
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_path
    if args.dataset == 'cifar':
        return get_dataset_cifar(args, data_dir)
    elif args.dataset == 'mnist':
        return get_dataset_mnist(args, data_dir)
    else:
        raise NotImplementedError()


def get_dataset_cifar(args, data_dir):
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4940, 0.4850, 0.4504), (0.2467, 0.2429, 0.2616))])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=train_transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=test_transform)

    # sample training data amongst users
    user_groups = distribute_dataset(train_dataset, args)

    return train_dataset, test_dataset, user_groups


def get_dataset_mnist(args, data_dir):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1326,), (0.3106,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)

    # sample training data amongst users Data distribution
    user_groups = distribute_dataset(train_dataset, args)

    return train_dataset, test_dataset, user_groups
