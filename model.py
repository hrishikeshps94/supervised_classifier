from operator import mod
from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models
from functools import partial
import numpy as np
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from collections import OrderedDict

class ResNet(nn.Module):

    def __init__(self, base_model,out_dim=3,feat_extract=True):
        super(ResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights=models.ResNet18_Weights),
                            "resnet50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
                            'effnetv2m':models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights),
                            'effnetv2s':models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights),
                            'regnet':models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2)}
        self.backbone = self.set_parameter_requires_grad(self._get_basemodel(base_model),feat_extract)
        if base_model=='effnetv2m' or base_model=='effnetv2s' :
            dim_mlp = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(nn.Linear(dim_mlp,dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))
        elif base_model=='resnet18' or base_model=='resnet50' or base_model=='regnet' :
            dim_mlp = self.backbone.fc.in_features        
            #add mlp projection head
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp,dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))
            # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp))

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def set_parameter_requires_grad(self,model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
        return model

    
    def forward(self, x):
        return self.backbone(x)

    
    def forward(self, x):
        return self.backbone(x)

class ResNet_Simclr(nn.Module):

    def __init__(self, base_model,model_data,out_dim=3,feat_extract=True):
        super(ResNet_Simclr, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=True),
                            "resnet50": models.resnet50(pretrained=True)}
        self.backbone = self.set_parameter_requires_grad(self._get_basemodel(base_model),feat_extract)
        dim_mlp = self.backbone.fc.in_features
        feat_extract = OrderedDict(self.backbone.named_children())
        del feat_extract['fc']
        self.backbone = nn.Sequential(feat_extract)
        model_data = {'.'.join(k.split('.')[1:]):v for k,v in model_data.items() if ( k.startswith('backbone') and not k.endswith('total_ops') and not k.endswith('total_params'))}    
        #add mlp projection head
        self.head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))
        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp))

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
        out = self.head(torch.flatten(feat,start_dim=1))
        return out


class ResNetFusion(nn.Module):

    def __init__(self, base_model,out_dim=3,feat_extract=True):
        super(ResNetFusion, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=True),
                            "resnet50": models.resnet50(pretrained=True)}
        self.backbone = self.set_parameter_requires_grad(self._get_basemodel(base_model),feat_extract)
        dim_mlp = self.backbone.fc.in_features        
        #add mlp projection head
        feat_extract = nn.ModuleList(self.backbone.children())
        self.in_conv = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(7,7),padding=(3,3))
        self.layer_0 = nn.Sequential(*feat_extract[1:4])
        self.layer_1 = feat_extract[4]
        self.layer_2 = feat_extract[5]
        self.layer_3 = feat_extract[6]
        self.layer_4 = feat_extract[7]
        self.out_feat = feat_extract[8]
        self.fusion_head = nn.Sequential(nn.Linear(dim_mlp,dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottel_neck = nn.Conv2d(960,512,1,padding='same')
        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp))

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def set_parameter_requires_grad(self,model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
        return model

    
    def forward(self, x):
        b,c,h,w = x.shape
        x = self.in_conv(x)
        feat_0 = self.layer_0(x)
        feat_1 = self.layer_1(feat_0)
        feat_2 = self.layer_2(feat_1)
        feat_3 = self.layer_3(feat_2)
        feat_4 = self.layer_4(feat_3)
        out_feat = self.out_feat(feat_4)
        fuse_feat = torch.cat([self.pool(self.pool(self.pool(feat_1))),self.pool(self.pool(feat_2)),self.pool(feat_3),feat_4],dim=1)

        fuse_feat = self.out_feat(self.bottel_neck(fuse_feat))
        # feat = torch.cat([fuse_feat,out_feat],dim=1)
        feat = fuse_feat+out_feat
        out = self.fusion_head(feat.view(b,-1))
        return out


class ResNetFuser(nn.Module):

    def __init__(self, base_model,out_dim=3,feat_extract=True):
        super(ResNetFuser, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=True),
                            "resnet50": models.resnet50(pretrained=True)}
        self.backbone = self.set_parameter_requires_grad(self._get_basemodel(base_model),feat_extract)
        dim_mlp = self.backbone.fc.in_features        
        #add mlp projection head
        self.backbone.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(7,7),padding=(3,3))
        self.backbone.fc = nn.Identity()
        self.mixer = nn.Sequential(nn.Linear(dim_mlp,dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def set_parameter_requires_grad(self,model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
        return model

    
    def forward(self, nr,seg):
        nr = self.backbone(nr)
        seg = self.backbone(seg)
        feat = nr+seg
        return self.mixer(feat)
