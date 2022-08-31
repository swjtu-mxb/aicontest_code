import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn
import math
from model import *

class AutoEncoder(nn.Layer):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2D(in_channels=128, out_channels=192, kernel_size=5, stride=2, padding=2))


        self.decoder = nn.Sequential(
            nn.Conv2DTranspose(192, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2DTranspose(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2DTranspose(128, 3, kernel_size=5, stride=2, padding=2, output_padding=1))

        self.bitEstimator = BitEstimator(192)
        
    
    def forward(self, input_image):
        a = input_image.shape[0]
        input_image = input_image.reshape([-1,3,256,256])
        encoded = self.encoder(input_image)

        feature_renorm = encoded
        if self.training:
            quant_noise_feature = paddle.zeros([input_image.shape[0], 192, input_image.shape[2] // 8, input_image.shape[3] // 8])
            quant_noise_feature = paddle.uniform(paddle.zeros_like(quant_noise_feature).shape, min=-0.5, max=0.5)
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = paddle.round(feature_renorm)

        decoded = self.decoder(compressed_feature_renorm)

        clipped_recon_image = decoded.clip(0., 1.)
        # distortion
        loss_func = nn.MSELoss()

        mse_loss = loss_func(clipped_recon_image, input_image)

        def iclr18_estimate_bits_z(z):
            prob_a = self.bitEstimator(z + 0.5)
            prob_b = self.bitEstimator(z - 0.5)
            prob = prob_a - prob_b
            # prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
            total_bits = paddle.sum(paddle.clip(-1.0 * paddle.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = iclr18_estimate_bits_z(compressed_feature_renorm)
        
        bpp_feature = total_bits_feature / (input_image.shape[0] * input_image.shape[2] * input_image.shape[3])

        return mse_loss, bpp_feature, encoded, decoded


def build_model():
    input_image = paddle.zeros([32,1,256,256])
    model = AutoEncoder()
    mse_loss, bpp_feature, encoded, decoded = model(input_image)

if __name__ == '__main__':
    build_model()