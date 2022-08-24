import numpy as np
import torch
import torch.nn as nn
import timm
from torch.autograd import Function
from feature_networks.feature import ShortCunk_CNN_Loop_Genre_Classifier, ShortCunk_CNN_AutoTagging_Classifier
from .torchvggish.torchvggish.vggish import VGGish

def _make_ShortCunk_CNN_Loop_Genre_Classifier(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.spec_bn, model.layer1)
    pretrained.layer1 = nn.Sequential(model.layer2)
    pretrained.layer2 = nn.Sequential(model.layer3)
    pretrained.layer3 = nn.Sequential(model.layer4)
    return pretrained

def _make_Shortchunk_CNN_Pretrained_on_MTAT(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.spec_bn, model.layer1)
    pretrained.layer1 = nn.Sequential(model.layer2)
    pretrained.layer2 = nn.Sequential(model.layer3)
    pretrained.layer3 = nn.Sequential(model.layer4)
    return pretrained

def _make_VGG(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.features[0], model.features[1], model.features[2])
    pretrained.layer1 = nn.Sequential(model.features[3], model.features[4], model.features[5])
    pretrained.layer2 = nn.Sequential(model.features[6], model.features[7], model.features[8], model.features[9], model.features[10])
    pretrained.layer3 = nn.Sequential(model.features[11], model.features[12], model.features[13], model.features[14], model.features[15])
    return pretrained

def calc_dims(pretrained, model_type=None):
    dims = []
    inp_res = 96
    if model_type == 'vgg':
        tmp = torch.zeros(1, 1, 96, 64).cuda() 
    else:
        tmp = torch.zeros(1, 1, 64, 96).cuda()
    tmp = pretrained.layer0(tmp)
    dims.append(tmp.shape[1:4])
    tmp = pretrained.layer1(tmp)
    dims.append(tmp.shape[1:4])
    tmp = pretrained.layer2(tmp)
    dims.append(tmp.shape[1:4])
    tmp = pretrained.layer3(tmp)
    dims.append(tmp.shape[1:4])

    # split to channels and resolution multiplier
    dims = np.array(dims)

    channels = dims[:, 0]
    res_mult = dims[:, 1] / inp_res
    return channels, res_mult

def _make_pretrained(backbone, model_path=None, loops_type=None, verbose=None):
    if backbone == "loops_genre": # ok
        model = ShortCunk_CNN_Loop_Genre_Classifier(loops_type=loops_type).cuda()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        pretrained = _make_ShortCunk_CNN_Loop_Genre_Classifier(model).cuda()
    elif backbone == "autotagging": # ok
        model = ShortCunk_CNN_AutoTagging_Classifier()
        checkpoint = torch.load(model_path)
        if 'spec.mel_scale.fb' in checkpoint.keys():
            model.spec.mel_scale.fb = checkpoint['spec.mel_scale.fb']
        model.load_state_dict(checkpoint)
        pretrained = _make_Shortchunk_CNN_Pretrained_on_MTAT(model).cuda()
    elif backbone == "vgg": # ok
        urls = {
            'vggish': "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"
        }
        model = VGGish(urls, preprocess=False, postprocess=False)
        pretrained = _make_VGG(model)
    
    pretrained.CHANNELS, pretrained.RES_MULT = calc_dims(pretrained, model_type=backbone)
    if verbose:
        print(f"Succesfully loaded:    {backbone}")
        print(f"Channels:              {pretrained.CHANNELS}")
        print(f"Resolution Multiplier: {pretrained.RES_MULT}")
    return pretrained

