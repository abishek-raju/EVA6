from torchvision import datasets, transforms


def train_dataset(train_transform, root_dir = '../data'):

    tr_dataset = datasets.MNIST(root_dir, train=True, download=True,
                   transform = train_transform)
    return tr_dataset

def train_dataset(test_transform, root_dir = '../data'):

    ts_dataset = datasets.MNIST(root_dir, train=False, download=True,
                   transform = test_transform)
    return ts_dataset