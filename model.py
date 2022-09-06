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






class MILVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        # self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        # self.num_classes=3
        self.size2 = int(np.sqrt(num_patches))
        self.dim = self.embed_dim
        self.L = 256 #self.dim//3
        self.D = 128 #self.dim//5
        self.K = 1 #self.num_classes*1
        # self.head = nn.Linear(in_features=384,out_features=self.num_classes)
        self.MIL_Prep = torch.nn.Sequential(
                torch.nn.Linear(self.dim, self.L),
                # torch.nn.BatchNorm1d(num_patches),
                torch.nn.LayerNorm(self.L),
                torch.nn.ReLU(inplace=True),
                # nn.Dropout(0.1)
                )
        self.MIL_attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            # nn.Tanh(),
            # nn.BatchNorm1d(num_patches),
            torch.nn.LayerNorm(self.D),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.D, self.K)

            # nn.Linear(self.L, self.K)
        )

        self.MIL_classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.num_classes),
        )


        trunc_normal_(self.pos_embed, std=.02)
        # self.head_dist.apply(self._init_weights)
        self.MIL_Prep[0].apply(self._init_weights)
        self.MIL_Prep[1].apply(self._init_weights)
        self.MIL_attention[0].apply(self._init_weights)
        self.MIL_attention[1].apply(self._init_weights)
        self.MIL_attention[4].apply(self._init_weights)
        self.MIL_classifier[0].apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def forward(self, x):
        x, x_patches = self.forward_features(x)
        vt_out = self.head(x)

        """MIL operations for the """
        H = self.MIL_Prep(x_patches)  #B*N*D -->  B*N*L

        A = self.MIL_attention(H)  # B*N*K
        # A = torch.transpose(A, 1, 0)  # KxN
        A = A.permute((0, 2, 1))  #B*K*N
        A = nn.functional.softmax(A, dim=2)  # softmax over N
        M = torch.bmm(A, H)  # B*K*N X B*N*L --> B*K*L
        M = M.view(-1, M.size(1)*M.size(2))

        mil_out = self.MIL_classifier(M)

        # return vt_out, mil_out, x_patches
        if self.training:
            return vt_out, mil_out
        else:
            # during inference, return the average of both classifier predictions
            return (vt_out+ mil_out) / 2


###################
class MILVisionTransformer_Distil(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.size2 = int(np.sqrt(num_patches))
        self.dim = self.embed_dim
        self.L = 256 #self.dim//3
        self.D = 128 #self.dim//5
        self.K = 1 #self.num_classes*1
        self.MIL_Prep = torch.nn.Sequential(
                torch.nn.Linear(self.dim, self.L),
                # torch.nn.BatchNorm1d(num_patches),
                torch.nn.LayerNorm(self.L),
                torch.nn.ReLU(inplace=True),
                # nn.Dropout(0.1)
                )
        self.MIL_attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            # nn.Tanh(),
            # nn.BatchNorm1d(num_patches),
            torch.nn.LayerNorm(self.D),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.D, self.K)

            # nn.Linear(self.L, self.K)
        )

        self.MIL_classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.num_classes),
            # nn.Sigmoid()
        )

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.MIL_Prep[0].apply(self._init_weights)
        self.MIL_Prep[1].apply(self._init_weights)
        self.MIL_attention[0].apply(self._init_weights)
        self.MIL_attention[1].apply(self._init_weights)
        self.MIL_attention[4].apply(self._init_weights)
        self.MIL_classifier[0].apply(self._init_weights)


    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1], x[:, 2:]

    def forward(self, x):
        x,  x_dist,  x_patches = self.forward_features(x)
        vt_out = self.head(x)
        dist_out = self.head_dist(x_dist)

        """MIL operations for the """
        """MIL operations for the """
        H = self.MIL_Prep(x_patches)  #B*N*D -->  B*N*L

        A = self.MIL_attention(H)  # B*N*K
        # A = torch.transpose(A, 1, 0)  # KxN
        A = A.permute((0, 2, 1))  #B*K*N
        A = nn.functional.softmax(A, dim=2)  # softmax over N
        M = torch.bmm(A, H)  # B*K*N X B*N*L --> B*K*L
        M = M.view(-1, M.size(1)*M.size(2))

        mil_out = self.MIL_classifier(M)

        # return vt_out, mil_out, x_patches
        if self.training:
            return (vt_out, dist_out), mil_out
        else:
            # during inference, return the average of both classifier predictions
            return (vt_out + dist_out + mil_out) / 3
#######################


###############################


@register_model
def MIL_VT_small_patch16_384(pretrained=False, **kwargs):
    model = MILVisionTransformer(
        img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def MIL_VT_small_patch16_512(pretrained=False, **kwargs):
    model = MILVisionTransformer(
        img_size=512, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
