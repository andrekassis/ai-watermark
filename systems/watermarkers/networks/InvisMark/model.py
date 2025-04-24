import logging

from .configs import ModelConfig
import torch
from torch import nn, Tensor
import torchvision
from torch.nn import functional as thf
import torchvision.transforms as transforms

from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = thf.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ImageViewLayer(nn.Module):
    def __init__(self, hidden_dim=16, channel=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel = channel 

    def forward(self, x):
        return x.view(-1, self.channel, self.hidden_dim, self.hidden_dim)


class ImageRepeatLayer(nn.Module):
    def __init__(self, num_repeats):
        super().__init__()
        self.num_repeats = num_repeats

    def forward(self, x):
        return x.repeat(1, 1, self.num_repeats, self.num_repeats)


class Watermark2Image(nn.Module):
    def __init__(self, watermark_len, resolution=256, hidden_dim=16, num_repeats=2, channel=3):
        super().__init__()
        assert resolution % hidden_dim == 0, "Resolution should be divisible by hidden_dim"
        pad_length = resolution // 4
        self.transform = nn.Sequential(
            nn.Linear(watermark_len, hidden_dim*hidden_dim*channel),
            ImageViewLayer(hidden_dim),
            nn.Upsample(scale_factor=(resolution//hidden_dim//num_repeats//2, resolution//hidden_dim//num_repeats//2)),
            ImageRepeatLayer(num_repeats),
            transforms.Pad(pad_length),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.transform(x)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activ='relu', norm=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activ == 'silu':
            self.activ = nn.SiLU(inplace=True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'leaky_relu':
            self.activ =  nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activ = None

        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        else:
            self.norm = None
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activ:
            x = self.activ(x)
        return x


class DecBlock(nn.Module):
    def __init__(self, in_channels, skip_channels='default', out_channels='default', activ='relu', norm=None):
        super().__init__()
        if skip_channels == 'default':
            skip_channels = in_channels//2
        if out_channels == 'default':
            out_channels = in_channels//2
        self.up = nn.Upsample(scale_factor=(2,2))
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv1 = Conv2d(in_channels, out_channels, 2, 1, 0, activ=activ, norm=norm)
        self.conv2 = Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1, activ=activ, norm=norm)
    
    def forward(self, x, skip):
        x = self.conv1(self.pad(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ModelConfig()
        self.watermark2image = Watermark2Image(self.config.num_encoded_bits, self.config.image_shape[0], 
                                                        self.config.watermark_hidden_dim, num_repeats=1) 
        # input_channel: 3 from image + 3 from watermark
        self.pre = Conv2d(6, self.config.num_initial_channels, 3, 1, 1)
        self.enc = nn.ModuleList()
        input_channel = self.config.num_initial_channels
        for _ in range(self.config.num_down_levels):
            self.enc.append(Conv2d(input_channel, input_channel*2, 3, 2, 1))
            input_channel *= 2
        
        self.dec = nn.ModuleList()
        for i in range(self.config.num_down_levels):
            skip_width = input_channel // 2 if i < self.config.num_down_levels - 1 else input_channel // 2 + 6 # 3 image channel + 3 watermark channel
            self.dec.append(DecBlock(input_channel, skip_width, activ='relu', norm='none'))
            input_channel //= 2 

        self.post = nn.Sequential(
            Conv2d(input_channel, input_channel, 3, 1, 1, activ='None'),
            Conv2d(input_channel, input_channel//2, 1, 1, 0, activ='silu'),
            Conv2d(input_channel//2, 3, 1, 1, 0, activ='tanh')
        )

    def forward(self, image: torch.Tensor, watermark=None):
        if watermark == None:
            logger.info("Watermark is not provided. Use zero bits as test watermark.")
            watermark = torch.zeros(image.shape[0], self.config.num_encoded_bits, device = image.device)
        watermark = self.watermark2image(watermark)
        inputs = torch.cat((image, watermark), dim=1)

        enc = []
        x = self.pre(inputs)
        for layer in self.enc:
            enc.append(x)
            x = layer(x)
        
        enc = enc[::-1]
        for i, (layer, skip) in enumerate(zip(self.dec, enc)):
            if i < self.config.num_down_levels - 1:
                x = layer(x, skip)
            else:
                x = layer(x, torch.cat([skip, inputs], dim=1))
        return self.post(x)




class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ModelConfig()

        self.extractor = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
        n_inputs = None
        for name, child in self.extractor.named_children():
            if name == 'classifier':
                for sub_name, sub_child in child.named_children():
                    if sub_name == '2':
                        n_inputs = sub_child.in_features
    
        self.extractor.classifier = nn.Sequential(
                    LayerNorm2d(n_inputs, eps=1e-6),
                    nn.Flatten(1),
                    nn.Linear(in_features=n_inputs, out_features=self.config.num_encoded_bits),
                )

        self.main = nn.Sequential(
            self.extractor,
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor):
        if image.shape[-1] != self.config.image_shape[-1] or image.shape[-2] != self.config.image_shape[-2]:
            logger.debug(f"Image shape should be {self.config.image_shape} but got {image.shape}")
            image = transforms.Resize(self.config.image_shape)(image)
        return self.main(image)
