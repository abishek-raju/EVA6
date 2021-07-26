try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
import os
from natsort import natsorted
import zipfile
import urllib.request
from skimage import io
import cv2
from PIL import Image
import torch
class Tiny_image_net_Dataset_200:
    def __init__(self,root_dir : str = "./data",tr_tst : str = "train",
                 url : str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"):
        self.root_dir = root_dir
        self.tr_tst = tr_tst
        self._transform = None
        self.zip_file_path = os.path.join(self.root_dir,"tiny_imagenet_200.zip")
        self.extracted_folder_path = os.path.join(self.root_dir,"tiny_imagenet_200")
        if not(os.path.exists(self.root_dir)):
            os.mkdir(self.root_dir)
        if not(os.path.exists(self.extracted_folder_path)):
            if not(os.path.isfile(self.zip_file_path)):
                #download code here
                print("Downloading ZIP File")
                urllib.request.urlretrieve(url, self.zip_file_path)
                pass
            #extract zip file
            print(f"Extracting ZIP File to {root_dir}")
            with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            os.rename(os.path.join(self.root_dir,"tiny-imagenet-200"),self.extracted_folder_path)
        self.classwise = os.listdir(os.path.join(self.extracted_folder_path,"train"))
    def __len__(self):
        if self.tr_tst == "train":
            return 70000
        else:
            return 30000

    @property
    def transforms(self):
        return self._transform
    @transforms.setter
    def transforms(self,transforms):
        self._transform = transforms

    def __getitem__(self,index):
        if self.tr_tst == "train":
            divisor = 350
            index_offset = 0
        else:
            divisor = 150
            index_offset = 350
        
        class_index,image_index = divmod(index,divisor)
        # print(class_index,image_index)
        # print(self.classwise[class_index])
        # print(os.path.join(self.extracted_folder_path,self.classwise[class_index],"images"))
        class_image_folder_path = os.path.join(self.extracted_folder_path,"train",self.classwise[class_index],"images")
        # print(natsorted(os.listdir(class_image_folder_path))[image_index])
        # print("asfdsaf",class_image_folder_path)
        image_path = os.path.join(class_image_folder_path,natsorted(os.listdir(class_image_folder_path))[image_index+index_offset])
        # print(image_path)
        # print(class_index)
        # print(index_offset)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._transform:
            # Apply transformations
            image = self._transform(image=image)['image']
            return image,class_index
        else:
            return Image.open(image_path).convert('RGB'),class_index