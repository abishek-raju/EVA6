# -*- coding: utf-8 -*-

import io
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms





def get_misclassified_images(max_misclassified_images,test_loader,class_names,device):
    dataiter = iter(test_loader)
    # X_train, y_train = dataiter.next()
    
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    model.eval()
    with torch.no_grad():
        for data, target in dataiter:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if len(misclassified_images) < max_misclassified_images:
                misclassified_images = misclassified_images + list(data[output.argmax(dim = 1) != target])
                misclassified_labels = misclassified_labels + list(target[output.argmax(dim = 1) != target])
                misclassified_preds = misclassified_preds + list(output[output.argmax(dim = 1) != target].argmax(dim = 1))
            else:
                break
    return misclassified_images[:max_misclassified_images],misclassified_labels[:max_misclassified_images],misclassified_preds[:max_misclassified_images]

def image_grid(misclassified_images,misclassified_labels,misclassified_preds):  
    figure = plt.figure(figsize=(12,8))

    for i in range(25):    
        plt.subplot(5, 5, i + 1)
        plt.xlabel("Pred : "+str(misclassified_preds[i].item())+"    Truth : "+str(misclassified_labels[i].item()))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(misclassified_images[i].to("cpu").squeeze().permute(1,2,0))
        

    return figure



def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return  transforms.ToTensor()(img)

#writer.add_image("Misclassified Images", fig2img(image_grid()))
#writer.flush()