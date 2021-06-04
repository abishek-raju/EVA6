# Session5
### Baseline
[Baseline notebook](EVAS5_baseline.ipynb)  
Target : Setting up the whole training pipeline.

Result : Params:6,379,786  
         Best Training Accuracy: 9.7%  
         Best Test Accuracy: 9.8%  

Analysis : Model is mot learning based on the data,fix the learning aspect.


### Iter1
[Iter1 notebook](EVAS5_iter1.ipynb)  
Target : Model skeleton which learns with less parameters.

Result : Params:27,006  
         Best Training ACC:9.8%  
         Best Test ACC:9.7%  
         
Analysis : Add Batch Normalisation after each convolution  layer.




### Iter2
[Iter2 notebook](EVAS5_iter2.ipynb)  
Target : ADD Batch Normalisation

Result : Params :5438  
         Best Train ACC:99.20%  
         Best Test ACC:99.0%  

Analysis : Accuracy stuck at 99.0% should add Dropout.


### Iter3
[Iter3 notebook](EVAS5_iter3.ipynb)  
Target : ADD LR Scheduler 

Result : Params : 8,828  
         Best Train ACC : 99.29%  
         Best Test ACC : 99.48%  

Analysis : Accuracy Target reached.
