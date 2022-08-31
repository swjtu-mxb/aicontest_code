from importlib import import_module
from multiprocessing import Pool
import argparse
from unittest import result
import arithmatic as ac
import torch
import torchvision
from model import *

from torch.utils.data.dataset import Dataset
import math
import sys

MAX_N = 2**16
TINY = 2**(-10)

out_channel_N = 32
out_channel_M = 64

ctx = torch.multiprocessing.get_context("spawn")
decompmodel = DeCompressor(out_channel_N, out_channel_M)
model_path = 'datas/ckpt/iter_220600.pth.tar'
load_model(decompmodel, model_path)
decompnet = decompmodel.cuda()
decompnet.eval()


def read_frequencies(bitin):
	def read_int(n):
		result = 0
		for _ in range(n):
			result = (result << 1) | bitin.read_no_eof()  # Big endian
		return result
	freqs = [read_int(32) for _ in range(512)]
	freqs.append(1)  # EOF symbol
	return ac.SimpleFrequencyTable(freqs)

def arithmeticDecompress(cmp_str):
    bitin = ac.BitInputStream(cmp_str)
    freqs = read_frequencies(bitin)
    dec = ac.ArithmeticDecoder(bitin)

    arr = np.zeros((out_channel_M,16,16))
    symbol_list = [] 
    
    for ch_idx in range(out_channel_M):
        for h_idx in range(16):
             for w_idx in range(16):
                symbols = dec.read(freqs)
                symbol_list.append(symbols)
                arr[ch_idx, h_idx, w_idx] = symbols-255
    
    # y_hat = torch.from_numpy(arr).to("cuda")
    return arr

def decompress(batch_y):                       
    y_hat = torch.from_numpy(batch_y).to("cuda")
    y_hat= y_hat.to(torch.float32)
    fake_images = decompnet(y_hat)
    # fakepath = "recon.png"        
    # torchvision.utils.save_image(fake_images, fakepath, normalize=False)
    # [16, 3, 256, 256]
    sq = (int)(math.sqrt(fake_images.shape[0]))
    subChannel = fake_images.shape[1:]
    # 0   1  2  3    4
    # [4, 4, 3, 256, 256]
    # 2, 0, 3, 1, 4
    subImgs = fake_images.reshape([sq, sq, subChannel[0], subChannel[1], subChannel[2]])
    # [3, 4, 256, 4, 256]
    subImgs = subImgs.permute(2, 0, 3, 1, 4)
    reWidth = subImgs.shape[1] * subImgs.shape[2]
    # [3, 4 * 256, 4 *256]
    ret = subImgs.reshape([3, reWidth, reWidth])
    ret = ret.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # [1024, 1024, 3]
    return ret
        

# transmit width and height
if __name__ == "__main__":

    # ctx = torch.multiprocessing.get_context("spawn")

    pool = Pool(processes=16)
    bin1to16 = []
    for i in range(16):
        path = '/home/cz-dl/cx/cccs/tmp/test_7_' + str(i) + '.bin'
        bin1to16.append(path)
    arr_batch = pool.map(arithmeticDecompress, bin1to16)
    arr = np.array(arr_batch)
    fake_images = decompress(arr)
    fakepath = './result.png'
    torchvision.utils.save_image(fake_images, fakepath, normalize=False)
    print(fake_images.shape)
    

