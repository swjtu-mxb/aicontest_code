import numpy as np
import os
import torch
from models import *

from model import *
import pandas as pd

model = LoadImageCompressor(out_channel_N=32,out_channel_M=64)
kernel_size = 3

os.makedirs('weight', exist_ok=True)
os.makedirs('bias', exist_ok=True)
os.makedirs('layer', exist_ok=True)
os.makedirs('data', exist_ok=True)

with open('./checkpoints/LEVIR_k3_32_64_16384/iter_220600.pth.tar', 'rb') as f:
   pretrained_dict = torch.load(f)
   model_dict = model.state_dict()   
   pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
   model_dict.update(pretrained_dict)
   model.load_state_dict(model_dict)

for name, parameters in model.state_dict().items():
  if not "prior" in name:
    if "bias" in name:
      if "Encoder" in name:
        a = parameters.detach().numpy()
        print(name, a.shape)
  
        if "fc" in name:
          file1 = './fc/' + name + '.csv'
          pd.DataFrame(a).to_csv(file1)
        else:
          file1 = './bias/' + name + '.csv'
          pd.DataFrame(a).to_csv(file1)

layer_list = []

for name, parameters in model.state_dict().items():
  if not "prior" in name:
    if "weight" in name:
      if "Encoder" in name:
        a = parameters.detach().numpy()
        print(name, a.shape)

        if "fc" in name:
          file1 = './weight/' + name + '.csv'
          pd.DataFrame(a).to_csv(file1)
        else:
          for i in range(a.shape[0]):
            for j in range(a.shape[1]):
              file1 = './weight/' + name +'_' + str(i) + '_' + str(j) + '.csv'
              b = np.array(a[i,j,:,:]).reshape(kernel_size,kernel_size)
              pd.DataFrame(b).to_csv(file1)


# dit = {'name':layer_list, 'min':min_list, 'max':max_list,}
# df = pd.DataFrame(dit)
# df.to_csv('./result_bias.csv',columns=['name','min','max'],index=False,sep=',')


