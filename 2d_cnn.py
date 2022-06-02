from matplotlib import axis
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import argparse
import csv
import os.path
import torch.nn.parallel
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image

epoches = 50
batches = 200

import torch
import torchvision.transforms as transforms
import torchvision

from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split

# import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
train_folder = "data10000/"
test_folder = "test_data/"
train_set,test_set=train_test_split([file for file in os.listdir(train_folder) if file.endswith(".jpg")], test_size=0.2)
train_set=[file for file in os.listdir(train_folder) if file.endswith(".jpg")]
test_set=[file for file in os.listdir(test_folder) if file.endswith(".jpg")]

class ImageDataset(Dataset):
  def __init__(self,dataset,img_folder,transform):
    
    self.dataset=dataset
    self.transform=transform
    self.img_folder=img_folder
    
    self.image_names=dataset
    self.labels=[[float(dat.split('_')[2]), float(dat.split('_')[4])] for dat in dataset]

#The __len__ function returns the number of samples in our dataset.
  def __len__(self):
    return len(self.image_names)

  def __getitem__(self,index):
    image = np.array(Image.open(self.img_folder+self.image_names[index]))
    image=self.transform(image)
    
    targets = pd.DataFrame(self.labels[index]).values
    # return image, targets
    return image, torch.FloatTensor(targets).view([-1])

train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((61, 61)),
                transforms.Grayscale(),
                transforms.ToTensor()])

test_transform =transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((61, 61)),
                transforms.Grayscale(),
                transforms.ToTensor()])

train_dataset=ImageDataset(train_set,train_folder,train_transform)
test_dataset=ImageDataset(test_set,test_folder,test_transform)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batches,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=1,
    shuffle=True
)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.dout1 = nn.Dropout(0.2)
        self.dout2 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,256,5)
        self.conv4 = nn.Conv2d(256,512,5)
        
        self.fc1 = nn.Linear(25600*2, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fcf = nn.Linear(128, 2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dout1(x)
        x = F.relu(x)
        x = self.pool1(F.relu(self.conv3(x)))
        x = self.dout2(x)
        x = self.pool2(F.relu(self.conv4(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fcf(x)
        return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1 = nn.Conv2d(1,32,3)
#         self.pool = nn.MaxPool2d(2,2)
#         self.dout = nn.Dropout(0.2)
#         self.conv2 = nn.Conv2d(32,64,5)
#         self.conv3 = nn.Conv2d(64,128,5)
#         self.conv4 = nn.Conv2d(128,256,5)
        
#         self.fc1 = nn.Linear(25600, 1024)
#         self.fc2 = nn.Linear(1024, 128)
#         self.fcf = nn.Linear(128, 2)
    
#     def forward(self,x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = self.dout(x)
#         x = F.relu(x)
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.dout(x)
#         x = self.pool(F.relu(self.conv4(x)))
#         x = x.view(x.size(0),-1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fcf(x)
#         return x


net = Net()
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  net = nn.DataParallel(net)
 
if torch.cuda.is_available():
   net.cuda()
 
import torch.optim as optim
 
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr = 0.0002)
loss_grpah = []

for epoch in range(epoches):
    running_loss = 0.0
    total = 0
    error = [0, 0]
    for i, d in enumerate(train_dataloader, 0):
        inputs, labels = d
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        # print(inputs.shape, outputs.shape, labels.shape)
        # print(outputs)
        loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        pred_error = labels.detach().cpu().numpy() - outputs.detach().cpu().numpy()
        # print(np.abs(pred_error).mean(axis=0))

        error[0] += np.abs(pred_error).mean(axis=0)[0]
        error[1] += np.abs(pred_error).mean(axis=0)[1]

        # if i%10 == 0:
        print('epoch : {}, {}/{}'.format(epoch, (i+1) * batches, len(train_set)))
    print('[%d/%d] loss: %.6f kk_err : %.6f kb_err : %.6f' % (epoch + 1, epoches, running_loss/i, error[0]/(i+1), error[1]/(i+1)))
    loss_grpah.append([running_loss/(i+1), error[0]/(i+1), error[1]/(i+1)])
    running_loss = 0.0

print('Finished Training')
save_name = 'data{}_epoches{}_batches_{}_512model_fix'.format(len(train_set),epoches,batches)
torch.save(net, save_name+'_model.pt')

import matplotlib.pyplot as plt

plt.plot(np.arange(1, len(loss_grpah)+1), loss_grpah)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.legend(['loss', 'kk_error', 'kb_error'])
plt.savefig(save_name+"_loss.jpg")
# plt.show()
plt.clf()

import tqdm
import matplotlib.patches as mpatches

error = [0, 0]
error_graph = [[], []]
correct_graph = [[], []]
except_error = 0.01

with torch.no_grad():
    for i, d in enumerate(tqdm.tqdm(test_dataloader)): #test loader에서 데이터들을 하나씩 꺼냅니다.
        images, labels = d
        images, labels = images.to(device), labels.to(device)
        outputs = net(images) #image를 model에 넣으면 class당 probability를 출력합니다.
        pred_error = labels.detach().cpu().numpy() - outputs.detach().cpu().numpy()

        error[0] += np.abs(pred_error.mean(axis=0)[0])
        error[1] += np.abs(pred_error.mean(axis=0)[1])
        # error[2] += np.abs(pred_error.mean(axis=0)[2])
        if (abs(pred_error[0][0]) <= except_error and abs(pred_error[0][1]) <= except_error):
            correct_graph[0].append(pred_error[0][0])
            correct_graph[1].append(pred_error[0][1])
            # correct_graph[2].append(pred_error[0][2])
        else:
            error_graph[0].append(pred_error[0][0])
            error_graph[1].append(pred_error[0][1])
            # error_graph[2].append(pred_error[0][2])

        # print(pred_error)
        # print(error[0]/total, error[1]/total)
        # print("test {}/{} finished!! {} {}".format(i, len(test_set), len(correct_graph[0]),len(test_set)))
        
    print('Accuracy : %f %%' % (100*len(correct_graph[0])/len(test_set)))

    plt.title('Accuracy : %f %%' % (100*len(correct_graph[0])/len(test_set)))
    plt.scatter(correct_graph[0]+error_graph[0], correct_graph[1]+error_graph[1], s=2)
    rect=mpatches.Rectangle((-except_error,-except_error),except_error*2,except_error*2, 
                            #fill=False,
                            alpha=0.2,
                        facecolor="red")
    plt.gca().add_patch(rect)

    plt.xlabel('kk')
    plt.ylabel('kb')
    plt.grid()
    plt.savefig(save_name+"_acc.jpg")
    # plt.show()