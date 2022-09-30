from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

class ResNet(nn.Module):

    def __init__(self, base_model,out_dim=3,feat_extract=True):
        super(ResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=True),
                            "resnet50": models.resnet50(pretrained=True)}
        self.backbone = self.set_parameter_requires_grad(self._get_basemodel(base_model),feat_extract)
        dim_mlp = self.backbone.fc.in_features        
        #add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))
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