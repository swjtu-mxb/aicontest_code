#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 
# * Create time   : 2018-08-22 09:54
# * Last modified : 2018-08-22 09:54
# * Filename      : RDN.py
# * Description   : this part for us is realize the RDN model from the paper
# * all detail you can see from the paper "Residual Dense Network for Image SR"
# **********************************************************
#from .BasicModule import basic
import torch.nn as nn 
import torch
import time

'''class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        #x = self.upscale(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    RDN(None)'''


class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(inchanels, growth_rate, kernel_size=kernel_size,padding = 1,stride= 1)
        self.relu = nn.ReLU()
    def forward(self,x):
        output = self.relu(self.conv(x))
        return torch.cat((x,output),1)

class RDB(nn.Module):
    def __init__(self,G0,C,G,kernel_size = 3):
        super(RDB,self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G,G))
        self.conv = nn.Sequential(*convs)
        #local_feature_fusion
        self.LFF = nn.Conv2d(G0+C*G,G0, kernel_size = 1, padding = 0,stride = 1)
    def forward(self,x):
        out = self.conv(x)
        lff = self.LFF(out)
        #local residual learning
        return lff + x

class rdn(nn.Module):
    def __init__(self, D = 2, C = 3, G = 32, G0 = 64, kernel_size = 3, input_channels = 320, out_channels = 320):
        
        #opts: the system para
       
        super(rdn,self).__init__()
        
        #D: RDB number 20
        #C: the number of conv layer in RDB 6
        #G: the growth rate 32
        #G0:local and global feature fusion layers 64filter
        
        self.D = D
        self.C = C
        self.G = G
        self.G0 = G0
        print("D:{},C:{},G:{},G0:{}".format(self.D,self.C,self.G,self.G0))
        kernel_size = kernel_size
        input_channels = input_channels
        #shallow feature extraction 
        self.SFE1 = nn.Conv2d(input_channels, self.G0, kernel_size=kernel_size, padding = 1, stride=1)
        self.SFE2 = nn.Conv2d(self.G0, self.G0, kernel_size=kernel_size, padding = 1, stride =1)
        #RDB for paper we have D RDB block
        self.RDBS = nn.ModuleList()
        for d in range(self.D):
            self.RDBS.append(RDB(self.G0,self.C,self.G,kernel_size))
        #Global feature fusion
        self.GFF = nn.Sequential(
               nn.Conv2d(self.D*self.G0, self.G0, kernel_size = 1, padding = 0 ,stride= 1),
               nn.Conv2d(self.G0, self.G0, kernel_size, padding = 1, stride = 1),
        )
        #upsample net 
        '''self.up_net = nn.Sequential(
                nn.Conv2d(self.G0, self.G*4, kernel_size = kernel_size, padding = 1, stride = 1),
                nn.PixelShuffle(2),
                nn.Conv2d(self.G, self.G*4, kernel_size = kernel_size, padding = 1, stride = 1),
                nn.PixelShuffle(2),
                nn.Conv2d(self.G, out_channels, kernel_size=kernel_size, padding = 1, stride = 1)
        )'''

        self.up_net = nn.Conv2d(self.G0, out_channels, kernel_size = kernel_size, padding = 1, stride = 1)
        #init
        for para in self.modules():
            if isinstance(para,nn.Conv2d):
                nn.init.kaiming_normal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()

    def forward(self,x):
        #f-1
        f__1 = self.SFE1(x)
        out  = self.SFE2(f__1)
        RDB_outs = []
        for i in range(self.D):
            out = self.RDBS[i](out)
            RDB_outs.append(out)
        out = torch.cat(RDB_outs,1)
        out = self.GFF(out)
        out = f__1+out
        return self.up_net(out)
        #return out
