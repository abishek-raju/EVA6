# -*- coding: utf-8 -*-
from torch.utils.tensorboard import SummaryWriter
from ..utils.get_misclassified_images import fig2img,image_grid

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
        
        self._misclassified_images = []
        self._misclassified_labels = []
        self._misclassified_preds = []

        self._misclassified_grid_image = None
        
        self._classified_images = []
        self._classified_labels = []
        self._classified_preds = []

        self._classified_grid_image = None
        
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
    
    @property
    def misclassified_len(self):
        return len(self._misclassified_images)
    
    @property
    def misclassified_images(self):
        return (self._misclassified_images,
        self._misclassified_labels,
        self._misclassified_preds)
    @misclassified_images.setter
    def misclassified_images(self, misclassified : "image,label,preds"):
        self._misclassified_images = misclassified[0]
        self._misclassified_labels = misclassified[1]
        self._misclassified_preds = misclassified[2]
        self.misclassified_grid_image = fig2img(image_grid(self._misclassified_images,self._misclassified_labels,self._misclassified_preds))
        
    @property
    def misclassified_grid_image(self):
        return self._misclassified_grid_image 
    @misclassified_grid_image.setter
    def misclassified_grid_image(self, img : "image"):
        self._misclassified_grid_image = img
        self.writer.add_image("Misclassified Images", img)
        self.flush()
#    @property
#    def misclassified_len(self):
#        return len(self._misclassified_images)
#    
#    @property
#    def misclassified_images(self):
#        return (self._misclassified_images,
#        self._misclassified_labels,
#        self._misclassified_preds)
#    @misclassified_images.setter
#    def misclassified_images(self, misclassified : "image,label,preds"):
#        if self.misclassified_len < self.max_misclassified_images:
#            if len(misclassified[0]) < (self.max_misclassified_images - self.misclassified_len):
#                max_threshold = len(misclassified[0]) - 1
#            else:
#                max_threshold = (self.max_misclassified_images - self.misclassified_len)
#            self._misclassified_images.append(misclassified[0][:max_threshold])
#            self._misclassified_labels.append(misclassified[1][:max_threshold])
#            self._misclassified_preds.append(misclassified[2][:max_threshold])
    
    @property
    def classified_images(self):
        return (self._classified_images,
        self._classified_labels,
        self._classified_preds)
    @classified_images.setter
    def classified_images(self, classified : "image,label,preds"):
        self._classified_images = classified[0]
        self._classified_labels = classified[1]
        self._classified_preds = classified[2]
        self.classified_grid_image = fig2img(image_grid(self._classified_images,self._classified_labels,self._classified_preds))
        
    @property
    def classified_grid_image(self):
        return self._classified_grid_image 
    @classified_grid_image.setter
    def classified_grid_image(self, img : "image"):
        self._classified_grid_image = img
        self.writer.add_image("Correctly Classified Images", img)
        self.flush()
        
        