
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34

from . import DBHead
import einops

class ImageMultiheadSelfAttention(nn.Module) :
	def __init__(self, planes):
		super(ImageMultiheadSelfAttention, self).__init__()
		self.attn = nn.MultiheadAttention(planes, 8)
	def forward(self, x) :
		res = x
		n, c, h, w = x.shape
		x = einops.rearrange(x, 'n c h w -> (h w) n c')
		x = self.attn(x, x, x)[0]
		x = einops.rearrange(x, '(h w) n c -> n c h w', n = n, c = c, h = h, w = w)
		return res + x

class double_conv(nn.Module):
	def __init__(self, in_ch, mid_ch, out_ch, stride = 1, planes = 256):
		super(double_conv, self).__init__()
		self.planes = planes
		# down = None
		# if stride > 1 :
		# 	down = nn.Sequential(
		# 		nn.AvgPool2d(2, 2),
		# 		nn.Conv2d(in_ch + mid_ch, self.planes * Bottleneck.expansion, kernel_size=1, stride=1, bias=False),nn.BatchNorm2d(self.planes * Bottleneck.expansion)
		# 		)
		self.down = None
		if stride > 1 :
			self.down = nn.AvgPool2d(2,stride=2)
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, stride = 1, bias=False),
			nn.BatchNorm2d(mid_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride = 1, bias=False),
			nn.BatchNorm2d(mid_ch),
			nn.ReLU(inplace=True),
			#Bottleneck(mid_ch, self.planes, stride, down, 2, 1, avd = True, norm_layer = nn.BatchNorm2d),
			nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride = 1, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		if self.down is not None :
			x = self.down(x)
		x = self.conv(x)
		return x

class double_conv_up(nn.Module):
	def __init__(self, in_ch, mid_ch, out_ch, planes = 256):
		super(double_conv_up, self).__init__()
		self.planes = planes
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, stride = 1, bias=False),
			nn.BatchNorm2d(mid_ch),
			nn.ReLU(inplace=True),
			#Bottleneck(mid_ch, self.planes, stride, down, 2, 1, avd = True, norm_layer = nn.BatchNorm2d),
			nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride = 1, padding=1, bias=False),
			nn.BatchNorm2d(mid_ch),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride = 2, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		x = self.conv(x)
		return x

class TextDetection(nn.Module) :
	def __init__(self, pretrained=None) :
		super(TextDetection, self).__init__()
		self.backbone = resnet34(pretrained=True if pretrained else False)

		self.conv_db = DBHead.DBHead(64, 0)

		self.conv_mask = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1),
			nn.Sigmoid()
		)

		self.down_conv1 = double_conv(0, 512, 512, 2)
		self.down_conv2 = double_conv(0, 512, 512, 2)
		self.down_conv3 = double_conv(0, 512, 512, 2)

		self.upconv1 = double_conv_up(0, 512, 256)
		self.upconv2 = double_conv_up(256, 512, 256)
		self.upconv3 = double_conv_up(256, 512, 256)
		self.upconv4 = double_conv_up(256, 512, 256, planes = 128)
		self.upconv5 = double_conv_up(256, 256, 128, planes = 64)
		self.upconv6 = double_conv_up(128, 128, 64, planes = 32)
		self.upconv7 = double_conv_up(64, 64, 64, planes = 16)

	def forward(self, x) :
		x = self.backbone.conv1(x)
		x = self.backbone.bn1(x)
		x = self.backbone.relu(x)
		x = self.backbone.maxpool(x) # 64@384

		h4 = self.backbone.layer1(x) # 64@384
		h8 = self.backbone.layer2(h4) # 128@192
		h16 = self.backbone.layer3(h8) # 256@96
		h32 = self.backbone.layer4(h16) # 512@48
		h64 = self.down_conv1(h32) # 512@24
		h128 = self.down_conv2(h64) # 512@12
		h256 = self.down_conv3(h128) # 512@6

		up256 = self.upconv1(h256) # 128@12
		up128 = self.upconv2(torch.cat([up256, h128], dim = 1)) # 64@24
		up64 = self.upconv3(torch.cat([up128, h64], dim = 1)) # 128@48
		up32 = self.upconv4(torch.cat([up64, h32], dim = 1)) # 64@96
		up16 = self.upconv5(torch.cat([up32, h16], dim = 1)) # 128@192
		up8 = self.upconv6(torch.cat([up16, h8], dim = 1)) # 64@384
		up4 = self.upconv7(torch.cat([up8, h4], dim = 1)) # 64@768

		return self.conv_db(up8), self.conv_mask(up4)

if __name__ == '__main__' :
	net = TextDetection().cuda()
	img = torch.randn(2, 3, 1536, 1536).cuda()
	db, seg = net(img)
	target = torch.randn(2, 3, 1536, 1536).cuda()
	F.l1_loss(db, target).backward()
	print(db.shape)
	print(seg.shape)
	breakpoint()
