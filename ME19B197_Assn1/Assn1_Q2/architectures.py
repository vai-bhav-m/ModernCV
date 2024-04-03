# Author: Vaibhav Mahapatra (ME19B197)

import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size, with_bn = False):
        super(MLPClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 250)
        self.bn2 = nn.BatchNorm1d(250)
        self.fc3 = nn.Linear(250, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, output_size)
        self.with_bn = with_bn

    def forward(self, x):
        x = self.flatten(x)

        if self.with_bn:
          x = F.relu(self.bn1(self.fc1(x)))
          x = F.relu(self.bn2(self.fc2(x)))
          x = F.relu(self.bn3(self.fc3(x)))

        else:
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = F.relu(self.fc3(x))

        res = self.out(x)
        return res

class VGG11Classifier(nn.Module):
    def __init__(self, n_hidden, n_units, num_classes=10):
        super(VGG11Classifier, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # 16 x 16 x 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # 8 x 8 x 128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # 4 x 4 x 256
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # 2 x 2 x 512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # 1 x 1 x 512
        )
        
        fcn = [nn.Flatten(),
               nn.Linear(512, n_units),
               nn.ReLU(inplace=True)]
        
        for _ in range(n_hidden):
            fcn.extend([nn.Linear(n_units, n_units), nn.ReLU(inplace=True)])

        fcn.extend([
            nn.Linear(n_units, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes)])

        self.ffn = nn.Sequential(*fcn)
        

    def forward(self, x):
        x = self.convnet(x)
        res = self.ffn(x)
        return res

class VGG11Classifier_BN(nn.Module):
    def __init__(self, n_hidden, n_units, num_classes=10):
        super(VGG11Classifier_BN, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # 16 x 16 x 64
           
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # 8 x 8 x 128
           
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # 4 x 4 x 256
           
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # 2 x 2 x 512
           
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # 1 x 1 x 512
        )
        
        fcn = [nn.Flatten(),
               nn.Linear(512, n_units),
               nn.BatchNorm1d(n_units),
               nn.ReLU(inplace=True)]
        
        for _ in range(n_hidden):
            fcn.extend([nn.Linear(n_units, n_units), nn.BatchNorm1d(n_units), nn.ReLU(inplace=True)])

        fcn.extend([
            nn.Linear(n_units, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes)])

        self.ffn = nn.Sequential(*fcn)
        

    def forward(self, x):
        x = self.convnet(x)
        res = self.ffn(x)
        return res
    
