import torch
from torch.nn import functional as F
from torch import nn, autograd, optim

def g_loss(fake_pred):
    loss = (-fake_pred).mean()
    return loss

def d_loss(real_pred, fake_pred):
    loss_Dgen = (F.relu(torch.ones_like(fake_pred) + fake_pred)).mean()
    loss_Dreal = (F.relu(torch.ones_like(real_pred) - real_pred)).mean()
    return loss_Dreal+loss_Dgen
