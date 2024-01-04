import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
# import cv2
from torch.utils.data import DataLoader,Subset
import torch
from  sklearn.model_selection import train_test_split


device = torch.device("mps" if  torch.backends.mps.is_available() else 'cpu')
print(f"device : {device}")


transform = transforms.ToTensor()
train_data = CIFAR10("data",train=True,download=True,transform=transform)
test_data = CIFAR10("data",download=True,train=False,transform=transform)


# train_images = []
# train_labels = []

# for images,labels in train_data:
#     train_images.append(images)
#     train_labels.append(labels)

# train_images = torch.stack(train_images).permute(0,2,3,1)
# train_labels = torch.tensor(train_labels)

# test_images = []
# test_labels = []
# for images,labels in test_data:
#     test_images.append(images)
#     test_labels.append(labels)

# test_images = torch.stack(test_images).permute(0,2,3,1)
# test_labels = torch.tensor(test_labels)



# train_images = train_data.data # OG Images
# train_labels = train_data.targets

# test_images = test_data.data
# test_lables = test_data.targets


NAMES = np.array(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])

# 이게 왜 이렇게 되냐 짜증나네
# image, label = train_data[0]
# fig, axes = plt.subplots(ncols=2,nrows=1)
# axes[0].imshow(image.view(32,32,3))
# axes[1].imshow(image.permute(1,2,0))
# plt.show()


def show_images(images, labels, ncols=8):
    figure, axs = plt.subplots(nrows=1,ncols=ncols)
    for i in range(ncols):
        axs[i].imshow(images[i])
        label = labels[i].squeeze()
        axs[i].set_title(NAMES[int(label)])
    
    plt.show()



# show_images(train_images[:8],train_labels[:8],ncols=8)
# show_images(train_images[8:16],train_labels[8:16],ncols=8)

# print(train_labels.shape)

IMAGE_SIZE = 32

# class SimpleModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # (N - k + p) / stride + 1
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=3, stride=2)
#         self.linear = nn.Linear(in_features=1568, out_features=10,bias=True)



#     def forward(self,x):

#         output = self.conv1(x)
#         output = self.relu(output)
#         output = self.conv2(output)
#         output = self.relu(output)
#         output = output.view(output.size()[0],-1)
#         output = self.linear(output)
        

#         return output

class ConvNetCustom(nn.Module):
    def __init__(self):
        super(ConvNetCustom,self).__init__()
        self.hiddenLayer = nn.Sequential(
            # nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2,stride=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # stride = kernel_size (default)
            # MaxPool : 32 x 32 -> 16 x 16
          
            nn.Conv2d(32,64,3,1,1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # MaxPool : 16 x 16 -> 8 x 8

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # # MaxPool : 8 x 8 -> 4 x 4
        )

        self.fcl = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(128*4*4,out_features=10),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            # nn.Linear(300,10),
            # nn.Softmax(dim=1),

        )
    
    def forward(self,x):
        output = self.hiddenLayer(x)
        output = output.view(output.size()[0],-1)
        output = self.fcl(output)

        return output
    



### Train_Valid SPlit #1
train_indices = list(range(len(train_data)))
np.random.shuffle(train_indices)
valid_train_split = int(np.floor(0.85 * len(train_data)))
train_indices, valid_indices = train_indices[:valid_train_split],train_indices[valid_train_split:]

train_subset = Subset(train_data,train_indices)
valid_subset = Subset(train_data,valid_indices)

### Train Valid SPlit #2 -> using sklearn.model_selection.train_test_split
#train_image,valid_image,train_label,valid_label = train_test_split(train_data,)


train_dataLoader = DataLoader(dataset=train_subset,batch_size=64,shuffle=True)
valid_dataLoader = DataLoader(dataset=valid_subset,batch_size=64,shuffle=True)
test_dataLodaer = DataLoader(dataset=test_data,batch_size=64,shuffle=True)





def seeLossPerClass(outputs,lables):
    lables_count = len(np.unique(lables))
    labels_unique = np.zeros((lables_count,2))
    labels_unique[:,0] = np.unique(lables)
    for index,output in enumerate(outputs):
        cl_num = lables[index]
        CE = -output[cl_num] + np.log(sum(np.exp(output)))

        cl_index = np.where(labels_unique[:,0] == cl_num)
        labels_unique[cl_index,1] += CE 
   
    # print(cl_num,CE)
   
    # print(labels_unique)

    return labels_unique
    
    

def train(epoch):    
    model.train()
    
    for batch_index, (images,labels) in enumerate(train_dataLoader):
        prev = []
        for i in range(0,2):
            images = images.to(device = device)
            labels = labels.to(device = device)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output,labels)
            output_cpu = output.cpu().detach().numpy()
            lable_cpu = labels.cpu().detach().numpy()
            
            if i == 0:
                prev = seeLossPerClass(outputs=output_cpu,lables=lable_cpu)
            else:
                post = seeLossPerClass(outputs=output_cpu,lables=lable_cpu)
                prev[:,1] = prev[:,1] - post[:,1]
            loss.backward()
            optimizer.step()
        
        print("prev - post : ",prev)
        # if batch_index % 100 == 0:
        #     print('iter #{}'.format(batch_index))
        #     print('output: \n',output[0:10])
        
             
        break
       

def getAcc(dataloader):

    test_loss =0.0
    correct = 0

    for (images,labels) in (dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
    
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _,preds = outputs.max(1)
        correct += preds.eq(labels).sum()


    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            epoch,
            test_loss / len(dataloader.dataset),
            correct.float() / len(dataloader.dataset)
    ))

@torch.no_grad()
def eval_training(epoch):
    model.eval()
    getAcc(valid_dataLoader)


if __name__ == '__main__':

    model = ConvNetCustom()
    model.to(device=device)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.01);
    loss_function = nn.CrossEntropyLoss();
    

    for epoch in range(1,30):
        train(epoch)
        eval_training(epoch)
    
   

    getAcc(test_dataLodaer)
