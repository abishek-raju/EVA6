from torchvision import datasets, transforms


def train_dataset(train_transform, dataset : str, 
                  root_dir : str  = '../data')->"dataset_obj":
    """
    Function which returns the dataset object according to the dataset
    parameter.
    
    Params:
        train_transform : Transform's compose object.
        dataset : MNIST/CIFAR10.
        root_dir : path to store data.
    """
    if dataset == "MNIST":
        tr_dataset = datasets.MNIST(root_dir, train=True, download=True,
                       transform = train_transform)
    elif dataset == "CIFAR10":
        tr_dataset = datasets.CIFAR10(root_dir, train=True,
                                        download=True, transform = train_transform)
    return tr_dataset

def test_dataset(test_transform, dataset : str, 
                  root_dir : str  = '../data')->"dataset_obj":
    """
    Function which returns the dataset object according to the dataset
    parameter.
    
    Params:
        test_transform : Transform's compose object.
        dataset : MNIST/CIFAR10.
        root_dir : path to store data.
    """
    if dataset == "MNIST":
        ts_dataset = datasets.MNIST(root_dir, train=False, download=True,
                       transform = test_transform)
    elif dataset == "CIFAR10":
        ts_dataset = datasets.CIFAR10(root_dir, train=False,
                                        download=True, transform = test_transform)
    return ts_dataset