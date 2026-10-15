import torch
from torch import nn
from torch.nn import functional as F

class SpectrogramCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        # Retain coarse frequency location; full global pooling loses pitch cues.
        self.features=nn.Sequential(nn.Conv2d(1,8,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((8,1)))
        self.head=nn.Linear(128,classes)
    def forward(self,x): return self.head(self.features(x).flatten(1))

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.):
        super().__init__()
        if gamma < 0: raise ValueError("gamma must be nonnegative")
        alpha=torch.as_tensor(alpha,dtype=torch.float32)
        if not bool(((alpha>0)&(alpha<1)).all()): raise ValueError("alpha must be in (0,1)")
        self.register_buffer("alpha",alpha);self.gamma=gamma
    def forward(self,logits,targets):
        bce=F.binary_cross_entropy_with_logits(logits,targets,reduction="none")
        p=torch.sigmoid(logits);pt=p*targets+(1-p)*(1-targets)
        weight=self.alpha*targets+(1-self.alpha)*(1-targets)
        return (weight*(1-pt).pow(self.gamma)*bce).mean()
