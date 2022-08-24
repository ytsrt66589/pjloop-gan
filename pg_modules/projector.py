import sys
sys.path.append('.')
sys.path.append('..')  
import torch
import torch.nn as nn
from feature_networks.pretrained_builder import _make_pretrained
from pg_modules.blocks import FeatureFusionBlock


def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    # shapes 
    out_channels = [cout, cout*2, cout*4, cout*8] if expand else [cout]*4

    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)
    
    scratch.CHANNELS = out_channels
    return scratch

def _make_scratch_csm(scratch, in_channels, cout, expand):
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], nn.ReLU(False), expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], nn.ReLU(False))
    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout*2, cout*4] if expand else [cout]*4
    return scratch


def _make_projector(res, backbone, model_path, cout, proj_type, loops_type, expand=False):
    assert proj_type in [0, 1, 2], "Invalid projection type"
    
    ### Build pretrained feature network 
    pretrained = _make_pretrained(backbone, model_path, loops_type)
    
    # Following Projected GAN 
    res = 128
    pretrained.RESOLUTIONS = [res//2, res//4, res//8, res//16]
    if proj_type == 0: return pretrained, None

    ### Build CCM 
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)
    
    pretrained.CHANNELS = scratch.CHANNELS 
    if proj_type == 1: return pretrained, scratch

    ### Build CSM
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)
    # CSM upsamples x2 so the feature map resolution doubles
    pretrained.RESOLUTIONS = [res*2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch


class F_Identity(nn.Module):
    def forward(self, x):
        return x

class F_RandomProj(nn.Module):
    def __init__(
        self,
        backbone,
        loops_type=None,
        model_path=None,
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        **kwargs,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.backbone = backbone
        self.loops_type = loops_type
        self.cout = cout
        self.expand = expand


        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(res=im_res, backbone=self.backbone, model_path=model_path, cout=self.cout,
                                                        proj_type=self.proj_type, loops_type=self.loops_type, expand=self.expand)
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x):
        
        # predict feature maps
        out0 = self.pretrained.layer0(x) 
        out1 = self.pretrained.layer1(out0) 
        out2 = self.pretrained.layer2(out1) 
        out3 = self.pretrained.layer3(out2) 

        # start enumerating at the lowest layer (this is where we put the first discriminator)
        out = {
            '0': out0,
            '1': out1,
            '2': out2,
            '3': out3,
        }

        if self.proj_type == 0: return out

        out0_channel_mixed = self.scratch.layer0_ccm(out['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(out['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(out['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(out['3'])

        out = {
            '0': out0_channel_mixed,
            '1': out1_channel_mixed,
            '2': out2_channel_mixed,
            '3': out3_channel_mixed,
        }

        if self.proj_type == 1: return out

        # from bottom to top
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            '0': out0_scale_mixed,
            '1': out1_scale_mixed,
            '2': out2_scale_mixed,
            '3': out3_scale_mixed,
        }

        return out
