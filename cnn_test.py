from sympy import Ne
import torch
from torch import nn
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

img_folder = "test_data/"
test_set=[file for file in os.listdir(img_folder) if file.endswith(".jpg")]

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

test_dataset=ImageDataset(test_set,img_folder,test_transform)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=1,
    shuffle=True
)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.dout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.conv4 = nn.Conv2d(128,256,5)
        
        self.fc1 = nn.Linear(25600, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fcf = nn.Linear(128, 2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dout(x)
        x = F.relu(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fcf(x)
        return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1 = nn.Conv2d(1,32,3)
#         self.pool1 = nn.MaxPool2d(2,2)
#         self.pool2 = nn.MaxPool2d(2,2)
#         self.dout1 = nn.Dropout(0.2)
#         self.dout2 = nn.Dropout(0.2)
#         self.conv2 = nn.Conv2d(32,64,5)
#         self.conv3 = nn.Conv2d(64,256,5)
#         self.conv4 = nn.Conv2d(256,512,5)
        
#         self.fc1 = nn.Linear(25600*2, 1024)
#         self.fc2 = nn.Linear(1024, 128)
#         self.fcf = nn.Linear(128, 2)
    
#     def forward(self,x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = self.dout1(x)
#         x = F.relu(x)
#         x = self.pool1(F.relu(self.conv3(x)))
#         x = self.dout2(x)
#         x = self.pool2(F.relu(self.conv4(x)))
#         x = x.view(x.size(0),-1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fcf(x)
#         return x

model = Net()
model = torch.load("data10000_epoches50_batches_200_512model_model.pt", map_location=device)
# model.eval()

model.to(device)
import tqdm

error = [0, 0]
error_graph = [[], []]
correct_graph = [[], []]
except_error = 0.01
with torch.no_grad():
    for i, d in enumerate(tqdm.tqdm(test_dataloader)): #test loader에서 데이터들을 하나씩 꺼냅니다.
        images, labels = d
        images, labels = images.to(device), labels.to(device)
        outputs = model(images) #image를 model에 넣으면 class당 probability를 출력합니다.
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
    plt.show()