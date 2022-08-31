# from https://www.codeproject.com/Articles/5061271/PixelCNN-in-Autoregressive-Models
from torch import nn


class MaskedConv2d(nn.Conv2d):
	
	def __init__(self, mask_type, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert mask_type in ('A', 'B')
		self.register_buffer('mask', self.weight.data.clone())
		_, _, kH, kW = self.weight.size()
		self.mask.fill_(1)
		self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
		self.mask[:, :, kH // 2 + 1:] = 0

	def forward(self, x):
		self.weight.data *= self.mask
		return super(MaskedConv2d, self).forward(x)


class ContextPrediction(nn.Module):
	def __init__(self, dim_in):
		super(ContextPrediction, self).__init__()
		self.masked = MaskedConv2d("A", in_channels=dim_in, out_channels=384, kernel_size=5, stride=1, padding=2)
	
	def forward(self, x):
		return self.masked(x)


class EntropyParameters(nn.Module):
	def __init__(self, dim_in):
		super(EntropyParameters, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=640, kernel_size=1, stride=1)
		self.conv2 = nn.Conv2d(in_channels=640, out_channels=512, kernel_size=1, stride=1)
		self.conv3 = nn.Conv2d(in_channels=512, out_channels=384, kernel_size=1, stride=1)
	
	def forward(self, x):
		x = self.conv1(x)
		x = nn.LeakyReLU()(x)
		x = self.conv2(x)
		x = nn.LeakyReLU()(x)
		x = self.conv3(x)
		return x