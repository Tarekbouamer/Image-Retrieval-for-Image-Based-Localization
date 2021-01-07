import torch
import torch.nn as nn

import cirtorch.layers.functional as LF


def contrastive_loss(x, label=None, margin=0.7, eps=1e-6):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    y = 0.5*lbl*torch.pow(D,2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y


def triplet_loss(x, label=None, label_msk=None, margin=0.1):

    N = len(torch.unique(label_msk))
    dim = x.size(0)

    x = x.permute(1, 0).unsqueeze(0).view(N, -1, dim)
    label = label.unsqueeze(0).view(N, -1)

    ns = torch.sum(label.data == 0, dim=1)

    xa = x[label.data == -1, :].unsqueeze(1).repeat(1, ns[0], 1)
    xp = x[label.data == 1, :].unsqueeze(1).repeat(1, ns[0], 1)

    xn = x[label.data == 0, :].unsqueeze(1).view(N, ns[0], -1)

    dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=2)
    dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=2)

    loss = torch.sum(torch.clamp(dist_pos - dist_neg + margin, min=0.0))

    return loss