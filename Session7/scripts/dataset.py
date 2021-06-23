from torchvision import datasets, transforms
import numpy as np


class cifar10:
    """
    cifar10 dataset class which call the transformations to augment the data
    """
    def __init__(self, root_dir : str  = '../data',
                 train : bool = False , download : bool = True,
                 transform : "transform_obj" = None)->"dataset_obj":
        self.transform = transform
        self.cifar_ = datasets.CIFAR10(root_dir, train=train,
                                        download=download)
    def __getitem__(self,index):
        image, label = self.cifar_[index]
        if self.transform:
            return self.transform(image = numpy.asarray(image))["image"],label
        else:
            return image,label
    def __len__(self):
        return len(self.cifar_.data)

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
        tr_dataset = cifar10(root_dir, train=True,
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
        ts_dataset = cifar10(root_dir, train=False,
                                        download=True, transform = test_transform)
    return ts_dataset