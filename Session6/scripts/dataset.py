


def train_dataset(train_transform, root_dir = '../data'):

    tr_dataset = datasets.MNIST(root_dir, train=True, download=True,
                   transform = train_transform)
    return tr_dataset