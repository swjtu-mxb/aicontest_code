import argparse
import cv2
from glob import glob
from itertools import product
import numpy as np
import os
from tqdm import tqdm
from scipy.special import erf

import torch
import torch.nn as nn
from datasets import Datasets, TestKodakDataset

from model import *
import arithmatic as ac
from PIL import Image
import torchvision
from torchvision import transforms
import pandas as pd

# MAX_N = 65536
# TINY = 1e-10
MAX_N = 2**16
TINY = 2**(-10)

def read_frequencies(bitin):
	def read_int(n):
		result = 0
		for _ in range(n):
			result = (result << 1) | bitin.read_no_eof()  # Big endian
		return result
	freqs = [read_int(32) for _ in range(512)]
	freqs.append(1)  # EOF symbol
	return ac.SimpleFrequencyTable(freqs)


def decompress(args):
    os.makedirs("outputs/recon/", exist_ok=True)
    
    if os.path.isdir(args.image_path):
        pathes = glob(os.path.join(args.image_path, '*'))
    else:
        pathes = [args.iamge_path]

    out_channel_N = 32
    out_channel_M = 64
    
    decompmodel = DeCompressor(out_channel_N, out_channel_M)
    load_model(decompmodel, args.model_path)
    decompnet = decompmodel.cuda()


    with torch.no_grad():
        decompnet.eval()
        for path in pathes:
            fileobj = open(path, mode='rb')
        bitin = ac.BitInputStream(fileobj)
        freqs = read_frequencies(bitin)

        dec = ac.ArithmeticDecoder(bitin)
        
        arr = np.zeros((1,out_channel_M,16,16))

        symbol_list = [] 
    
        for ch_idx in range(out_channel_M):
            for h_idx in range(16):
                 for w_idx in range(16):
                    symbols = dec.read(freqs)
                    symbol_list.append(symbols)
                    arr[0, ch_idx, h_idx, w_idx] = symbols-255

           
        y_hat = torch.from_numpy(arr).to("cuda")
        y_hat= y_hat.to(torch.float32)
        fake_images = decompnet(y_hat)
        fakepath = "outputs/recon/{}.png".format(os.path.basename(path).split('.')[0])
        
        torchvision.utils.save_image(fake_images, fakepath, normalize=False)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./checkpoints/ex_k3_32_64_16384/iter_250128.pth.tar')
    parser.add_argument('--image_path', default='./image/')

    args = parser.parse_args()
    decompress(args)