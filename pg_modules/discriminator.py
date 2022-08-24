import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import torch.nn.init as init
from pg_modules.projector import F_RandomProj
from pg_modules.blocks import DownBlock, DownBlockPatch, conv2d
from stylegan2.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2
        #self.last_layer = conv2d(nfc[end_sz], 1, 4, 2, 0, bias=False) # best
        self.last_layer = conv2d(nfc[end_sz]+1, 1, 4, 1, 0, bias=False)
        
        #layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False)) # for 192 64
        #layers.append(conv2d(nfc[end_sz], 1, 4, 2, 0, bias=False))
        self.main = nn.Sequential(*layers)
        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x, c, alpha=0):
        out = self.main(x)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.last_layer(out)
        out = out.view(batch, -1)
        return out

class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=4,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        separable=True,
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs] 
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDisc #SingleDiscCond if cond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch)],

        self.mini_discs = nn.ModuleDict(mini_discs)
        
    def forward(self, features, c, alpha=-1):
        all_logits = []
        for k, disc in self.mini_discs.items():
            tmp = disc(features[k], c, alpha)
            tmp = tmp.view(features[k].size(0), -1)
            all_logits.append(tmp)
        all_logits = torch.cat(all_logits, dim=1) 
        
        return all_logits

#ok
class SingleProjectedDiscriminator(nn.Module):
    def __init__(
        self,
        model_type="vgg",
        model_path=None,
        loops_type=None,
        backbone_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.model_type = model_type
        self.feature_network = F_RandomProj(backbone=model_type, model_path=model_path, loops_type=loops_type)
        self.discriminator = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            **backbone_kwargs,
        )
    def train(self, mode=True):
        self.feature_network = self.feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)
    
    def forward(self, x):
        features = {}
        features_0_map = {}
        features_1_map = {}
        front = x[:, :, 0:96, :]
        back = x[:, :, 96:192, :]
        d = 3 
        if self.model_type != "vgg":
            front = front.permute(0, 1, 3, 2)
            back = back.permute(0, 1, 3, 2)
            d = 2
        features_0_map = self.feature_network(front) 
        features_1_map = self.feature_network(back) 

        features['0'] = torch.cat([features_0_map['0'], features_1_map['0']], dim=d)
        features['1'] = torch.cat([features_0_map['1'], features_1_map['1']], dim=d)
        features['2'] = torch.cat([features_0_map['2'], features_1_map['2']], dim=d)
        features['3'] = torch.cat([features_0_map['3'], features_1_map['3']], dim=d)
        
        logits = self.discriminator(features, None)

        return logits

#ok
class FusionProjectedDiscriminator(nn.Module):
    def __init__(
        self,
        model_type="vgg",
        additional_model_type=None,
        model_path=None,
        loops_type=None,
        backbone_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.model_type = model_type
        self.additional_model_type = additional_model_type
        assert self.model_type == "vgg"
        self.feature_network_general = F_RandomProj(backbone=model_type, model_path=None, loops_type=None)
        self.discriminator_general = MultiScaleD(
            channels=self.feature_network_general.CHANNELS,
            resolutions=self.feature_network_general.RESOLUTIONS,
            **backbone_kwargs,
        )

        self.feature_network_domain = F_RandomProj(backbone=additional_model_type, model_path=model_path, loops_type=loops_type)
        self.discriminator_domain = MultiScaleD(
            channels=self.feature_network_domain.CHANNELS,
            resolutions=self.feature_network_domain.RESOLUTIONS,
            **backbone_kwargs,
        )

    def train(self, mode=True):
        self.feature_network_general = self.feature_network_general.train(False)
        self.discriminator_general = self.discriminator_general.train(mode)
        self.feature_network_domain = self.feature_network_domain.train(False)
        self.discriminator_domain = self.discriminator_domain.train(mode)
        return self

    def eval(self):
        return self.train(False)
    
    def forward(self, x):
        
        general_features = {}
        general_features_0_map = {}
        general_features_1_map = {}
        front = x[:, :, 0:96, :]
        back = x[:, :, 96:192, :]
        d = 3 
        general_features_0_map = self.feature_network_general(front) 
        general_features_1_map = self.feature_network_general(back) 
        general_features['0'] = torch.cat([general_features_0_map['0'], general_features_1_map['0']], dim=d)
        general_features['1'] = torch.cat([general_features_0_map['1'], general_features_1_map['1']], dim=d)
        general_features['2'] = torch.cat([general_features_0_map['2'], general_features_1_map['2']], dim=d)
        general_features['3'] = torch.cat([general_features_0_map['3'], general_features_1_map['3']], dim=d)


        domain_features = {}
        domain_features_0_map = {}
        domain_features_1_map = {}
        d = 2
        domain_features_0_map = self.feature_network_domain(front.permute(0, 1, 3, 2)) 
        domain_features_1_map = self.feature_network_domain(back.permute(0, 1, 3, 2)) 
        domain_features['0'] = torch.cat([domain_features_0_map['0'], domain_features_1_map['0']], dim=d)
        domain_features['1'] = torch.cat([domain_features_0_map['1'], domain_features_1_map['1']], dim=d)
        domain_features['2'] = torch.cat([domain_features_0_map['2'], domain_features_1_map['2']], dim=d)
        domain_features['3'] = torch.cat([domain_features_0_map['3'], domain_features_1_map['3']], dim=d)
        
        logits = self.discriminator_general(general_features, None)
        logits += self.discriminator_domain(domain_features, None)

        return logits
