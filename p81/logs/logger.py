# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter


class log_training_params:
    def __init__(self,exp_name : str = "Experiment_1",max_misclassified_images : int = 20,
                 tensorboard_root : str = "runs/"):
        self.max_misclassified_images = max_misclassified_images
        self.exp_name = exp_name
        self.tensorboard_root = tensorboard_root
        self._train_loss = []
        self._test_loss = []
        self._train_accuracy = []
        self._test_accuracy = []
        self.writer = SummaryWriter(tensorboard_root+exp_name)
    @property
    def train_test_loss(self):
        return self._train_loss,self._test_loss
    @train_test_loss.setter
    def train_test_loss(self, loss : "tuple_tr_loss_tst_loss_epoch"):
        self._train_loss.append(loss[0])
        self._test_loss.append(loss[1])
        self.writer.add_scalars('Loss',
        {
        'train_loss': loss[0],
        'test_loss': loss[1],
        },loss[2])
    
    @property
    def train_test_accuracy(self):
        return self._train_accuracy,self._test_accuracy
    @train_test_accuracy.setter
    def train_test_accuracy(self, acc : "tuple_tr_acc_tst_acc_epoch"):
        self._train_accuracy.append(acc[0])
        self._test_accuracy.append(acc[1])
        self.writer.add_scalars('Accuracy',
        {
        'train_accuracy': acc[0],
        'test_accuracy': acc[1],
        },acc[2])
    
    def flush(self):
        self.writer.flush()