# -*- encoding: utf-8 -*-
import torch.nn as nn

from Activation.kafnets import KAF, KAF2D

class SiameseNetwork(nn.Module):
    def __init__(self,flag_kaf=False):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(2, stride=2))
        if(flag_kaf):
            self.fc1 = nn.Sequential(
                nn.Linear(50 * 4 * 4, 500),
                #nn.ReLU(inplace=True),
                KAF(500),
                #KAF2D(500),
                #nn.Linear(250, 10),
                nn.Linear(500,10),
                nn.Linear(10, 2))
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(50 * 4 * 4, 500),
                nn.ReLU(inplace=True),
                #KAF(500),
                #KAF2D(500),
                #nn.Linear(250, 10),
                nn.Linear(500,10),
                nn.Linear(10, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
