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

# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1 = nn.Conv2d(1,32,3)
#         self.pool = nn.MaxPool2d(2,2)
#         self.dout = nn.Dropout(0.2)
#         self.conv2 = nn.Conv2d(32,64,5)
#         self.conv3 = nn.Conv2d(64,256,5)
#         self.conv4 = nn.Conv2d(256,512,5,padding=0)
        
#         self.fc1 = nn.Linear(51200, 1024)
#         self.fc2 = nn.Linear(1024, 128)
#         self.fc3 = nn.Linear(128, 2)
    
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.dout(x)
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
#         x = self.fc3(x)
#         return x
# model = Net()
# model = torch.load("model_epoches50_batches_50.pt", map_location=device)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1 = nn.Conv2d(1,32,3,padding=3)
#         self.pool = nn.MaxPool2d(2,2)
#         self.dout = nn.Dropout(0.2)
#         self.conv2 = nn.Conv2d(32,64,5,padding=5)
#         self.conv3 = nn.Conv2d(64,256,5)
#         self.conv4 = nn.Conv2d(256,512,5,padding=0)
        
#         self.fc1 = nn.Linear(100352, 2048)
#         self.fc2 = nn.Linear(2048, 512)
#         self.fc3 = nn.Linear(512, 128)
#         self.fc4 = nn.Linear(128, 2)
    
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.dout(x)
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
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
 
# model = Net()
# model = torch.load("model_epoches50_batches_50.pt", map_location=device)
# model.to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=3)
        self.pool = nn.MaxPool2d(2,2)
        self.dout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32,64,5,padding=5)
        self.conv3 = nn.Conv2d(64,256,5)
        self.conv4 = nn.Conv2d(256,512,5,padding=0)
        self.conv5 = nn.Conv2d(512,1024,5,padding=0)
        
        self.fc1 = nn.Linear(25600, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dout(x)
        x = F.relu(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=3)
        self.pool = nn.MaxPool2d(2,2)
        self.dout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32,64,5,padding=0)
        self.conv3 = nn.Conv2d(64,256,5)
        self.conv4 = nn.Conv2d(256,512,5,padding=0)
        self.conv5 = nn.Conv2d(512,2048,5,padding=0)
        
        self.fc1 = nn.Linear(32768, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dout(x)
        x = F.relu(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = Net()
model = torch.load("data12000_epoches30_batches_1000_model.pt", map_location=device)
model.to(device)



# class BasicBlock(nn.Module):
#     """Basic Block for resnet 18 and resnet 34
#     """

#     #BasicBlock and BottleNeck block
#     #have different output size
#     #we use class attribute expansion
#     #to distinct
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         #residual function
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#         )

#         #shortcut
#         self.shortcut = nn.Sequential()

#         #the shortcut output dimension is not the same with residual function
#         #use 1*1 convolution to match the dimension
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# class BottleNeck(nn.Module):
#     """Residual block for resnet over 50 layers
#     """
#     expansion = 4
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )

#         self.shortcut = nn.Sequential()

#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion)
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# class ResNet(nn.Module):

#     def __init__(self, block, num_block, num_classes=2):
#         super().__init__()

#         self.in_channels = 64

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         #we use a different inputsize than the original paper
#         #so conv2_x's stride is 1
#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         """make resnet layers(by layer i didnt mean this 'layer' was the
#         same as a neuron netowork layer, ex. conv layer), one layer may
#         contain more than one residual block
#         Args:
#             block: block type, basic block or bottle neck block
#             out_channels: output depth channel number of this layer
#             num_blocks: how many blocks per layer
#             stride: the stride of the first block of this layer
#         Return:
#             return a resnet layer
#         """

#         # we have num_block blocks per layer, the first block
#         # could be 1 or 2, other blocks would always be 1
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         output = self.conv1(x)
#         output = self.conv2_x(output)
#         output = self.conv3_x(output)
#         output = self.conv4_x(output)
#         output = self.conv5_x(output)
#         output = self.avg_pool(output)
#         output = output.view(output.size(0), -1)
#         output = self.fc(output)

#         return output

# def resnet50():
#     """ return a ResNet 50 object
#     """
#     return ResNet(BottleNeck, [1, 4, 6, 3])

# net = resnet50()
# model = torch.load("resnet50_epoches50_batches_200_model.pt", map_location=device)


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