# Session6
## Contents
[1.Normalization Module](#norm_module)  
[2.Normalization Technique Working](#norm_excel)  
[3.Network with Group Normalization + L1](#EVAS6_iter_1)  
[4.Network with Layer Normalization + L2](#EVAS6_iter_2)  
[5.Network with L1 + L2 + BN](#EVAS6_iter_3)  
[6.Loss and Accuracy Curves ](#loss_accuracy)  

**1. Normalization Module:**<a name="norm_module"></a>  
Layer Normalization is a special case of Group Normalization. Where Layer Normalization has Group as 1.
```python
def normalization_technique(normalization,in_channels):
    if normalization == "GN":
        return nn.GroupNorm(2,in_channels)
    elif normalization == "LN":
        return nn.GroupNorm(1,in_channels)
    else:
        return nn.BatchNorm2d(in_channels)
```

**2. Normalization Technique Working:**<a name="norm_excel"></a>  
Excel shown [here](Batch_layer_group_norm.ods) shows how to perform the three Normalization techniques.  
**3. Network with Group Normalization + L1:**<a name="EVAS6_iter_1"></a>  
[EVAS6_iter_1](EVAS6_iter_1.ipynb)
Click [here](https://tensorboard.dev/experiment/w5w8rj1SR9mirUxefAxiEA/) for visualization of tensorboard.  
  
**25** misclassified images  
![](GN_l1.png)

**4. Network with Layer Normalization + L2:**<a name="EVAS6_iter_2"></a>  
[EVAS6_iter_2](EVAS6_iter_2.ipynb)  
Click [here](https://tensorboard.dev/experiment/kPv1c9zrRcaoKAEoyBLmPw/) for visualization of tensorboard.

**25** misclassified images  
![](LN_l2.png)

**5. Network with L1 + L2 + BN:**<a name="EVAS6_iter_3"></a>  
[EVAS6_iter_3](EVAS6_iter_3.ipynb)  
Click [here](https://tensorboard.dev/experiment/3e9S68TNSYuwfwmSL5iyXw/) for visualization of tensorboard.

**25** misclassified images  
![](BN_l1_l2.png)


**6.Loss and Accuracy Curves:**<a name="loss_accuracy"></a>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Group normalisation L1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Layer normalisation L2 L1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Batch normalisation L1 and L2
<p float="left">
  <img src="Accuracy_1.svg" width="300" />
  <img src="Accuracy_2.svg" width="300" /> 
  <img src="Accuracy_3.svg" width="300" />
</p>

<p float="left">
  <img src="Loss_1.svg" width="300" />
  <img src="Loss_2.svg" width="300" /> 
  <img src="Loss_3.svg" width="300" />
</p>




