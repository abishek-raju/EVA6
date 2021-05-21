# Pytorch 101

### Contents
[1.Data Representation](#datarepresentation)  
[2.Data Generation](#datageneration)  
[3.Combining Two Inputs](#combining)  
[4.Evaluating Results](#evaluation)  
[5.Final Results](#final_results)  
[6.loss Function](#loss)  
[7.Training on GPU](#training)  



**1.Data Representation:**<a name="datarepresentation"></a>
There are three parts by which the data is represented.
1.Black and white image of size 28x28
2.Labels "0-9" integer.
3.Randomn number between "0-9" integer.


**2.Data Generation:**<a name="datageneration"></a>
The below shown class is responsible to create the sample by combining the MNIST dataset and Randomn Number.
```python
class Mnist_Rand_Number(Dataset):
    def __init__(self, mnist_builtin):
        self.mnist_builtin = mnist_builtin

    def __getitem__(self, index):
        image,label = self.mnist_builtin[index]
        return image,label,random.randint(0,9)

    def __len__(self):
        return len(self.mnist_builtin)
```
**3.Combining Two Inputs:**<a name="combining"></a>
```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)    # 1x28x28
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)   # 8x26x26
        # maxpool                                                               # 16x24x24
        self.fc1 = nn.Linear(in_features=16 * 12 * 12, out_features=120)        # 16x12x12
        self.fc2 = nn.Linear(in_features=120, out_features=60)                  # 1x120
        self.out1 = nn.Linear(in_features=60, out_features=10)                  # 1x60
                                                                                # 1x10
        self.out2 = nn.Linear(in_features=60 + 10, out_features=19)             # 1x60
                                                                                # 1x19
    def forward(self,img,rnum):
        x = self.conv1(img)

        x = self.conv2(F.relu(x))
        x = F.max_pool2d(F.relu(x), kernel_size=2, stride=2)
        x = F.relu(x).flatten(start_dim=1, end_dim=-1)
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        y1 = self.out1(F.relu(x))
        y2 = self.out2(torch.cat([F.relu(x),torch.nn.functional.one_hot(torch.tensor(rnum),10)],dim = 1))
        return torch.cat([F.log_softmax(y1,dim = 1),F.log_softmax(y2,dim = 1)],dim = 1)
```
In the above code the line,
```python
y2 = self.out2(torch.cat([F.relu(x),torch.nn.functional.one_hot(torch.tensor(rnum),10)],dim = 1))
```
concatinates the output after the MNIST class is recognised to the randomn number by converting to a One Hot Vector.

**4.Evaluating Results:**<a name="evaluation"></a>
The Final Evaluation of the Model is based on the accuracy of MNIST class identifed correctly and the Sum of the MNIST class and the Randomn number generated.


**5.Final Results:**<a name="final_results"></a>
Number of Epochs = 20

Training MNIST Accuracy = 99.9
Test MNIST Accuracy = 99.04

Training Randomn Accuracy = 78.26
Test Randomn Accuracy = 77.37


**6.Loss Function:**<a name="loss"></a>
NLLLoss. This is the most common used loss function for classification tasks.


**7.Training on GPU:**<a name="training"></a>
Model is transferred to the GPU only once.
While each batch is fetched from the dataloader the batch is then moved to the Cuda GPU.
