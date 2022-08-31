import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *
from torch.distributions.uniform import Uniform
import pandas as pd



class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b = np.random.uniform(-1, 1)
        #b = 0
        uniform_distribution = Uniform(-0.5*torch.ones(x.size())
                                       * (2**b), 0.5*torch.ones(x.size())*(2**b)).sample().cuda()
        return torch.round(x+uniform_distribution)-uniform_distribution

    @staticmethod
    def backward(ctx, g):

        return g

def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.gaussin_entropy_func = Distribution_for_entropy()
        self.p = P_Model(out_channel_M)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)

        
        feature = self.Encoder(input_image)
        # ---------------------------------------------------------------------------
        # print(feature.shape)
        # a = feature.cpu().detach().numpy().reshape(64,16,16)
        # print(a.shape)
        
        # for i in range(64):
        #     file1 = './data/output_k'+str(i)+'.csv'
        #     b = np.array(a[i,:,:]).reshape(16,16)
        #     pd.DataFrame(b).to_csv(file1)
        # ---------------------------------------------------------------------------
        batch_size = feature.size()[0]

        z = self.priorEncoder(feature)
        
        if self.training:
            compressed_z = UniverseQuant.apply(z)
        else:
            compressed_z = torch.round(z)     

        feature_renorm = feature
   
        if self.training:
            compressed_feature_renorm = UniverseQuant.apply(feature_renorm)
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        hyper_dec =  self.p(self.priorDecoder(compressed_z))
        xp1 = self.gaussin_entropy_func(compressed_feature_renorm, hyper_dec)
        
        train_bpp = torch.sum(torch.log(xp1)) / (-np.log(2))

        recon_image = self.Decoder(compressed_feature_renorm)
        
        clipped_recon_image = recon_image.clamp(0., 1.)
        
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob


        total_bits_feature = train_bpp
       
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp

    def train_only_decoder(self, x):
        with torch.no_grad():
            y = self.Encoder(x)
        recon_image = self.Decoder(torch.round(y))    
        clipped_recon_image = recon_image.clamp(0., 1.)
        mse_loss = torch.mean((recon_image - x).pow(2))
       
        return clipped_recon_image, mse_loss


class LoadImageCompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(LoadImageCompressor, self).__init__()
        self.Encoder = Load_Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.gaussin_entropy_func = Distribution_for_entropy()
        self.p = P_Model(out_channel_M)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)

        
        feature = self.Encoder(input_image)
        # ---------------------------------------------------------------------------
        print(feature.shape)
        a = feature.cpu().detach().numpy().reshape(feature.shape[1],16,16)
        print(a.shape)
        
        for i in range(feature.shape[1]):
            file1 = './data/output_k'+str(i)+'.csv'
            b = np.array(a[i,:,:]).reshape(16,16)
            pd.DataFrame(b).to_csv(file1)
        # ---------------------------------------------------------------------------
        batch_size = feature.size()[0]

        z = self.priorEncoder(feature)
        
        if self.training:
            compressed_z = UniverseQuant.apply(z)
        else:
            compressed_z = torch.round(z)     

        feature_renorm = feature
   
        if self.training:
            compressed_feature_renorm = UniverseQuant.apply(feature_renorm)
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        hyper_dec =  self.p(self.priorDecoder(compressed_z))
        xp1 = self.gaussin_entropy_func(compressed_feature_renorm, hyper_dec)
        
        train_bpp = torch.sum(torch.log(xp1)) / (-np.log(2))

        recon_image = self.Decoder(compressed_feature_renorm)
        
        clipped_recon_image = recon_image.clamp(0., 1.)
        
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob


        total_bits_feature = train_bpp
       
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp

    def train_only_decoder(self, x):
        with torch.no_grad():
            y = self.Encoder(x)
        recon_image = self.Decoder(torch.round(y))    
        clipped_recon_image = recon_image.clamp(0., 1.)
        mse_loss = torch.mean((recon_image - x).pow(2))
       
        return clipped_recon_image, mse_loss


class Comp_decompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Comp_decompressor, self).__init__()
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_image):
        
        feature = self.Encoder(input_image)
        compressed_feature_renorm = torch.floor(feature)
        
        recon_image = self.Decoder(compressed_feature_renorm)
        
        clipped_recon_image = recon_image.clamp(0., 1.)
        
        return clipped_recon_image



class Compressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Compressor, self).__init__()
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.gaussin_entropy_func = Distribution_for_entropy()
        self.p = P_Model(out_channel_M)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_image):
        
        feature = self.Encoder(input_image)
        
        z = self.priorEncoder(feature)
        if self.training:
            compressed_z = UniverseQuant.apply(z)
        else:
            compressed_z = torch.round(z)         

        feature_renorm = feature
   
        if self.training:
            compressed_feature_renorm = UniverseQuant.apply(feature_renorm)
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
           
        hyper_dec =  self.p(self.priorDecoder(compressed_z))

        c = hyper_dec.size()[1]
        mean = hyper_dec[:, :c//2, :, :]
        scale = hyper_dec[:, c//2:, :, :]
        scale[scale == 0] = 1e-9

       
        return compressed_feature_renorm, mean, scale

class DeCompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(DeCompressor, self).__init__()
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, cx):
        
        recon_image = self.Decoder(cx)
        
        clipped_recon_image = recon_image.clamp(0., 1.)
       
        return clipped_recon_image