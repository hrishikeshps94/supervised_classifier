import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50,resnet18
from torchvision import models
from collections import OrderedDict


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        dim_mlp = resnet18().fc.in_features
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(dim_mlp, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Simclr_ResNet(nn.Module):

    def __init__(self, base_model,feature_dim=128,feat_extract=False):
        super(Simclr_ResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=True),
                            "resnet50": models.resnet50(pretrained=True)}
        self.backbone = self.set_parameter_requires_grad(self._get_basemodel(base_model),feat_extract)
        dim_mlp = self.backbone.fc.in_features
        # Taking only the required layers for feature extraction
        feat_extract = OrderedDict(self.backbone.named_children())
        del feat_extract['fc']
        self.backbone = nn.Sequential(feat_extract)
        

        #add mlp projection head
        self.head = nn.Sequential(nn.Linear(dim_mlp, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def set_parameter_requires_grad(self,model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
        return model

    
    def forward(self, x):
        feat = self.backbone(x)
        feat = torch.flatten(feat, start_dim=1)
        out = self.head(feat)
        return F.normalize(feat, dim=-1), F.normalize(out, dim=-1)