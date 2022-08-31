#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
from torch._C import qscheme
import torch.nn as nn
import torch
from .GDN import GDN
#from .RDN import RDB
from .resnet import ResBlock
from .RDN import rdn
import numpy as np
import pandas as pd

kernel_size = 3
class Analysis_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Analysis_net, self).__init__()

        self.conv1 = nn.Conv2d(3, out_channel_N, kernel_size, stride=2, padding=int((kernel_size-1)/2))
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (1 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = torch.nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, kernel_size, stride=2, padding=int((kernel_size-1)/2))
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = torch.nn.ReLU(inplace = True)  
        
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, kernel_size, stride=2, padding=int((kernel_size-1)/2))
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = torch.nn.ReLU(inplace = True)

        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, kernel_size, stride=2, padding=int((kernel_size-1)/2))
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)


    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        # --------------------------------------------------------------------------- 
        # a = x.cpu().detach().numpy().reshape(32,128,128)
        # print(a.shape)
        # for i in range(32):
        #     file1 = './layer/layer1_k'+str(i)+'.csv'
        #     b = np.array(a[i,:,:]).reshape(128,128)
        #     pd.DataFrame(b).to_csv(file1) 
        # --------------------------------------------------------------------------- 

        x = self.gdn2(self.conv2(x)) 
        # --------------------------------------------------------------------------- 
        # a = x.cpu().detach().numpy().reshape(32,64,64)
        # print(a.shape)
        # for i in range(32):
        #     file1 = './layer/layer2_k'+str(i)+'.csv'
        #     b = np.array(a[i,:,:]).reshape(64,64)
        #     pd.DataFrame(b).to_csv(file1)
        # --------------------------------------------------------------------------- 

        x = self.gdn3(self.conv3(x))
        # --------------------------------------------------------------------------- 
        # a = x.cpu().detach().numpy().reshape(32,32,32)
        # print(a.shape)
        # for i in range(32):
        #     file1 = './layer/layer3_k'+str(i)+'.csv'
        #     b = np.array(a[i,:,:]).reshape(32,32)
        #     pd.DataFrame(b).to_csv(file1)
        # # --------------------------------------------------------------------------- 

        return self.conv4(x)


class Load_Analysis_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Load_Analysis_net, self).__init__()

        self.conv1 = nn.Conv2d(3, out_channel_N, kernel_size, stride=2, padding=int((kernel_size-1)/2))
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (1 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = torch.nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, kernel_size, stride=2, padding=int((kernel_size-1)/2))
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = torch.nn.ReLU(inplace = True)  
        
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, kernel_size, stride=2, padding=int((kernel_size-1)/2))
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = torch.nn.ReLU(inplace = True)

        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, kernel_size, stride=2, padding=int((kernel_size-1)/2))
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)


    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        # --------------------------------------------------------------------------- 
        a = x.cpu().detach().numpy().reshape(x.shape[1],128,128)
        print(a.shape)
        for i in range(x.shape[1]):
            file1 = './layer/layer1_k'+str(i)+'.csv'
            b = np.array(a[i,:,:]).reshape(128,128)
            pd.DataFrame(b).to_csv(file1) 
        # --------------------------------------------------------------------------- 

        x = self.gdn2(self.conv2(x)) 
        # --------------------------------------------------------------------------- 
        a = x.cpu().detach().numpy().reshape(x.shape[1],64,64)
        print(a.shape)
        for i in range(x.shape[1]):
            file1 = './layer/layer2_k'+str(i)+'.csv'
            b = np.array(a[i,:,:]).reshape(64,64)
            pd.DataFrame(b).to_csv(file1)
        # --------------------------------------------------------------------------- 

        x = self.gdn3(self.conv3(x))
        # --------------------------------------------------------------------------- 
        a = x.cpu().detach().numpy().reshape(x.shape[1],32,32)
        print(a.shape)
        for i in range(x.shape[1]):
            file1 = './layer/layer3_k'+str(i)+'.csv'
            b = np.array(a[i,:,:]).reshape(32,32)
            pd.DataFrame(b).to_csv(file1)
        # # --------------------------------------------------------------------------- 

        return self.conv4(x)

def build_model():
        input_image = torch.zeros([4, 3, 256, 256])

        analysis_net = Analysis_net()
        feature = analysis_net(input_image)

        print(feature.size())


if __name__ == '__main__':
    build_model()
