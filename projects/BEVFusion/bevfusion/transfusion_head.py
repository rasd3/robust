# modify from https://github.com/mit-han-lab/bevfusion
import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.task_modules import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from torch import nn

from mmdet3d.models import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.models.layers import nms_bev
from mmdet3d.registry import MODELS
from mmdet3d.structures import xywhr2xyxyr
from .encoder_utils import LocalContextAttentionBlock_BEV, ConvBNReLU
from timm.models.layers import trunc_normal_
import math
from .deformable_transformer import build_deforamble_transformer
from .deformable_utils.position_encoding import PositionEmbeddingSine
from .utils import NestedTensor

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y
    
@MODELS.register_module()
class DeformableTransformer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.mask_freq = kwargs.pop("mask_freq")
        self.mask_ratio = kwargs.pop("mask_ratio")
        if kwargs.get('residual', False):
            self.residual = kwargs.pop("residual")
        else:
            self.residual = False
        if kwargs.get('loss_weight', False):
            self.loss_weight = kwargs.pop("loss_weight")
        else:
            self.loss_weight = 1
        self.model = build_deforamble_transformer(**kwargs)
        feat_channels = 256
        target_channels = 80
        in_channels = [80, 256]
        self.conv = nn.Sequential(nn.Conv2d(
                sum(in_channels), feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(True))
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats= feat_channels // 2, normalize=True)
        self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(feat_channels, feat_channels, kernel_size=1),
                    nn.GroupNorm(32, feat_channels),
                )])        
        self.num_cross_attention_layers = kwargs.get('num_cross_attention_layers', False)
        if self.num_cross_attention_layers:
            self.target_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(target_channels, feat_channels, kernel_size=1),
                        nn.GroupNorm(32, feat_channels),
                    )]) 
        self.pts_mask_tokens = nn.Parameter(torch.zeros(1, 1, feat_channels))
        
        self.pred = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        if self.residual == 'concat':
            self.P_integration = ConvBNReLU(2 * feat_channels, feat_channels, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.initialize_weights()
        
    def initialize_weights(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        if self.num_cross_attention_layers:
            for _proj in self.target_proj:
                nn.init.xavier_uniform_(_proj[0].weight, gain=1)
                nn.init.constant_(_proj[0].bias, 0)
        torch.nn.init.normal_(self.pts_mask_tokens, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def pts_masking(self, x):
        # code from https://github.com/facebookresearch/mae/blob/main/models_mae.py
        _x = x.clone().detach()
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)    
        pts_mask_tokens = self.pts_mask_tokens.repeat(N, ids_restore.shape[1] + 1 - x_masked.shape[1], 1)
        x_ = torch.cat([x_masked, pts_mask_tokens],dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
        return _x, x, mask

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # image feature, points feature
        if self.residual:
            residual = inputs[1]
        
        prob = np.random.uniform()
        mask_pts = prob < self.mask_freq
        
        if mask_pts and inputs[1].requires_grad:
            bs, c, h, w = inputs[1].shape
            pts_feat = inputs[1].flatten(2).transpose(1, 2)
            pts_target, pts_feat, pts_mask = self.pts_masking(pts_feat)
            src = pts_feat.view(bs,h,w,c).permute(0,3,1,2) # bs, c, h, w
        else:
            src = inputs[1]
        s_proj = self.input_proj[0](src)
        target = inputs[0]
        if self.num_cross_attention_layers:
            t_proj = self.target_proj[0](target)
        else:
            t_proj = target
        masks = torch.zeros(
                (s_proj.shape[0], s_proj.shape[2], s_proj.shape[3]),
                dtype=torch.bool,
                device=s_proj.device,
            )
        pos_embeds = self.position_embedding(NestedTensor(s_proj, masks)).to(
                s_proj.dtype)
        inputs[1] = self.model([s_proj], [masks], [pos_embeds], [t_proj], query_embed=None)
        inputs[1] = self.pred(inputs[1])
        if self.residual == 'sum':
            inputs[1] += residual
        if self.residual == 'concat':
            inputs[1] = self.P_integration(torch.cat((inputs[1], residual), dim=1))
        if mask_pts and inputs[1].requires_grad:
            pts_feat = inputs[1].flatten(2).transpose(1, 2)
            loss = (pts_feat - pts_target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = (loss * pts_mask).sum() / pts_mask.sum()  # mean loss on removed patches
            loss = self.loss_weight * loss
            return self.conv(torch.cat(inputs, dim=1)), loss
        
        return self.conv(torch.cat(inputs, dim=1)), False

@MODELS.register_module()
class ModalitySpecificDecoderMask(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mask_ratio=0.5, mask_pts=False, mask_img=False, num_layers=1, kernel_size=9, bn_momentum=0.1, bias='auto', pos_emb=False):
        super(ModalitySpecificDecoderMask, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_hidden_channel = 256
        self.camera_hidden_channel = 80
        self.conv = nn.Sequential(nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        # self.pts_mask_token = nn.Parameter(torch.zeros(1, self.lidar_hidden_channel, 1, 1))
        # trunc_normal_(self.pts_mask_token, mean=0., std=.02)
        self.mask_ratio = mask_ratio
        self.mask_pts = mask_pts
        self.mask_img = mask_img
        in_channels_img = in_channels[0]
        in_channels_pts = in_channels[1]
        
        decoder_embed_dim=256
        #decoder_num_heads=16
        decoder_num_heads=1
        mlp_ratio=1
        norm_layer=nn.LayerNorm
        decoder_depth=1
        self.decoder_embed_img = nn.Linear(in_channels_img, decoder_embed_dim, bias=True)
        self.decoder_embed_pts = nn.Linear(in_channels_pts, decoder_embed_dim, bias=True)
        self.img_mask_tokens = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pts_mask_tokens = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_img = nn.Parameter(torch.zeros(1, 180*180, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_pts = nn.Parameter(torch.zeros(1, 180*180, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 180*180 * decoder_embed_dim, bias=True) # decoder to patch
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed_img = get_2d_sincos_pos_embed(self.decoder_pos_embed_img.shape[-1], 180, cls_token=False)
        self.decoder_pos_embed_img.data.copy_(torch.from_numpy(decoder_pos_embed_img).float().unsqueeze(0))
        decoder_pos_embed_pts = get_2d_sincos_pos_embed(self.decoder_pos_embed_pts.shape[-1], 180, cls_token=False)
        self.decoder_pos_embed_pts.data.copy_(torch.from_numpy(decoder_pos_embed_pts).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.img_mask_tokens, std=.02)
        torch.nn.init.normal_(self.pts_mask_tokens, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def img_masking(self, x):
        _x = x.clone().detach()
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)    
        
        x_masked = self.decoder_embed_img(x_masked)
        img_mask_tokens = self.img_mask_tokens.repeat(N, ids_restore.shape[1] + 1 - L, 1)
        x_ = torch.cat([x_masked, img_mask_tokens],dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
        return _x, x, mask
    
    def pts_masking(self, x):
        _x = x.clone().detach()
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)    
        
        x_masked = self.decoder_embed_pts(x_masked)
        pts_mask_tokens = self.pts_mask_tokens.repeat(N, ids_restore.shape[1] + 1 - L, 1)
        x_ = torch.cat([x_masked, pts_mask_tokens],dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
        return _x, x, mask
    
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        N,IC,H,W = inputs[0].shape
        N,PC,H,W = inputs[1].shape
        img_feat = inputs[0].view(N,IC,H*W).permute(0,2,1).contiguous()
        pts_feat = inputs[1].view(N,PC,H*W).permute(0,2,1).contiguous()
        
        prob = np.random.uniform()
        mask_img = prob < 0.5 and prob > 0.25
        mask_pts = prob < 0.25
        
        if mask_img and inputs[0].requires_grad:
            img_target, img_feat, img_mask = self.img_masking(img_feat)
        else:
            img_feat = self.decoder_embed_img(img_feat)
        
        img_feat = img_feat + self.decoder_pos_embed_img
        for blk in self.decoder_blocks:
            import pdb;pdb.set_trace()
            img_feat = blk(img_feat)
        import pdb;pdb.set_trace()
        img_feat = self.decoder_norm(img_feat)
        # predictor projection
        img_feat = self.decoder_pred(img_feat)
        
        if mask_pts and inputs[1].requires_grad:
            pts_target, pts_feat, pts_mask = self.pts_masking(pts_feat)
        else:
            pts_feat = self.decoder_embed_pts(pts_feat)
        pts_feat = pts_feat + self.decoder_pos_embed_pts
        for blk in self.decoder_blocks:
            pts_feat = blk(pts_feat)
        pts_feat = self.decoder_norm(pts_feat)
        # predictor projection
        pts_feat = self.decoder_pred(pts_feat)

        if mask_img and inputs[0].requires_grad:
            
            loss = (img_feat - img_target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            loss = (loss * img_mask).sum() / img_mask.sum()  # mean loss on removed patches
            return self.conv(torch.cat(inputs, dim=1)), loss
        
        if mask_pts and inputs[1].requires_grad:
            loss = (pts_feat - pts_target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = (loss * pts_mask).sum() / pts_mask.sum()  # mean loss on removed patches
            return self.conv(torch.cat(inputs, dim=1)), loss
        return self.conv(torch.cat(inputs, dim=1)), False

@MODELS.register_module()
class ConvFuser(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # image feature, points feature
        return super().forward(torch.cat(inputs, dim=1))


@MODELS.register_module()
class ModalitySpecificLocalCrossAttentionlayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size):
        super(ModalitySpecificLocalCrossAttentionlayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_hidden_channel = 256
        self.camera_hidden_channel = 80
        
        self.P_IML = LocalContextAttentionBlock_BEV(self.lidar_hidden_channel, self.camera_hidden_channel, self.lidar_hidden_channel, kernel_size)
        self.I_IML = LocalContextAttentionBlock_BEV(self.camera_hidden_channel, self.lidar_hidden_channel, self.camera_hidden_channel, kernel_size)
        self.P_integration = ConvBNReLU(2 * self.lidar_hidden_channel, self.lidar_hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.I_integration = ConvBNReLU(2 * self.camera_hidden_channel, self.camera_hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        img_feat = inputs[0]
        lidar_feat = inputs[1]
        I2I_feat = self.I_IML(img_feat, lidar_feat)
        new_img_feat = self.I_integration(torch.cat((I2I_feat, img_feat),dim=1))
        P2P_feat = self.P_IML(lidar_feat, img_feat)
        new_lidar_feat = self.P_integration(torch.cat((P2P_feat, lidar_feat),dim=1))
        inputs = [new_img_feat, new_lidar_feat]
        return inputs

@MODELS.register_module()
class ModalitySpecificLocalSelfAttentionlayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size):
        super(ModalitySpecificLocalSelfAttentionlayer, self).__init__()
        self.in_channels = in_channels
        self.IML = LocalContextAttentionBlock_BEV(self.in_channels, self.in_channels, self.in_channels, kernel_size)
        self.integration = ConvBNReLU(2 * self.in_channels, self.in_channels, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)

    def forward(self, inputs):
        feat = inputs
        new_feat = self.IML(feat, feat)
        new_feat = self.integration(torch.cat((new_feat, feat),dim=1))
        return new_feat

@MODELS.register_module()
class ModalitySpecificLocalCrossAttention(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_layers=1, kernel_size=9, bn_momentum=0.1,
                bias='auto'):
        super(ModalitySpecificLocalCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_hidden_channel = 256
        self.camera_hidden_channel = 80
        self.conv = nn.Sequential(nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.num_layers=num_layers
        self.cross_attn_list=nn.ModuleList()
        in_channels_img = in_channels[0]
        in_channels_pts = in_channels[1]
        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            in_channels_pts,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            in_channels_img,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        for i in range(num_layers):
            self.cross_attn_list.append(ModalitySpecificLocalCrossAttentionlayer(in_channels, out_channels, kernel_size))
        
        self.bn_momentum = bn_momentum
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        inputs[0] = self.shared_conv_img(inputs[0])
        inputs[1] = self.shared_conv_pts(inputs[1])
        for idx in range(self.num_layers):
            inputs = self.cross_attn_list[idx](inputs)
        return self.conv(torch.cat(inputs, dim=1))
def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb



@MODELS.register_module()
class ModalitySpecificLocalAttentionMask(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mask_ratio=0.5, mask_pts=False, mask_img=False, num_layers=1, kernel_size=9, bn_momentum=0.1, bias='auto', pos_emb=False):
        super(ModalitySpecificLocalAttentionMask, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_hidden_channel = 256
        self.camera_hidden_channel = 80
        self.conv = nn.Sequential(nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        # self.pts_mask_token = nn.Parameter(torch.zeros(1, self.lidar_hidden_channel, 1, 1))
        # trunc_normal_(self.pts_mask_token, mean=0., std=.02)
        self.mask_ratio = mask_ratio
        self.mask_pts = mask_pts
        self.mask_img = mask_img
        self.num_layers=num_layers
        self.cross_attn_list=nn.ModuleList()
        self.img_self_attn_list=nn.ModuleList()
        self.pts_self_attn_list=nn.ModuleList()
        in_channels_img = in_channels[0]
        in_channels_pts = in_channels[1]
        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            in_channels_pts,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            in_channels_img,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        for i in range(num_layers):
            self.img_self_attn_list.append(ModalitySpecificLocalSelfAttentionlayer(in_channels_img, in_channels_img, kernel_size))
        for i in range(num_layers):
            self.pts_self_attn_list.append(ModalitySpecificLocalSelfAttentionlayer(in_channels_pts, in_channels_pts, kernel_size))
        for i in range(num_layers):
            self.cross_attn_list.append(ModalitySpecificLocalCrossAttentionlayer(in_channels, out_channels, kernel_size))
        self.pos_emb = pos_emb
        self.pts_bev_embedding = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        self.img_bev_embedding = nn.Sequential(
            nn.Linear(80 * 2, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, 80)
        )
        self.bn_momentum = bn_momentum
        self.init_weights()
        
    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
        
    def img_masking(self, img_feat):
        clean_img_feat = img_feat.clone().detach()
        BN, C, H, W = img_feat.shape
        self.img_mask_count = int(np.ceil(H*W * self.mask_ratio))
        img_mask_idx = np.random.permutation(H*W)[:self.img_mask_count]
        img_mask = np.zeros(H*W, dtype=int)
        img_mask[img_mask_idx] = 1
        img_mask = img_mask.reshape((H, W))
        img_mask = torch.tensor(img_mask).to(device=img_feat.device)
        # img_mask_tokens = self.img_mask_token.expand(BN,-1,H,W).to(device=img_feat.device)
        img_mask_tokens = torch.zeros(BN,C,H,W).to(device=img_feat.device)
        masked_img_feat = img_feat * (1-img_mask) + img_mask_tokens* img_mask
        return clean_img_feat, masked_img_feat, img_mask
    
    def pts_masking(self, pts_feat):
        clean_pts_feat = pts_feat.clone().detach()
        BN, C, H, W = pts_feat.shape
        self.pts_mask_count = int(np.ceil(H*W * self.mask_ratio))
        pts_mask_idx = np.random.permutation(H*W)[:self.pts_mask_count]
        pts_mask = np.zeros(H*W, dtype=int)
        pts_mask[pts_mask_idx] = 1
        pts_mask = pts_mask.reshape((H, W))
        pts_mask = torch.tensor(pts_mask).to(device=pts_feat.device)
        #pts_mask_tokens = self.pts_mask_token.expand(BN,-1,H,W).to(device=pts_feat.device)
        pts_mask_tokens = torch.zeros(BN,C,H,W).to(device=pts_feat.device)
        masked_pts_feat = pts_feat * (1-pts_mask) + pts_mask_tokens* pts_mask
        return clean_pts_feat, masked_pts_feat, pts_mask
    @property
    def coords_bev(self):
        grid_size = [1440,1440]
        downsample_scale = 8
        x_size, y_size = (
            grid_size[1] // downsample_scale,
            grid_size[0] // downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        return coord_base
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        inputs[0] = self.shared_conv_img(inputs[0])
        inputs[1] = self.shared_conv_pts(inputs[1])
        prob = np.random.uniform()

        mask_img = prob < 0.5 and prob > 0.25
        if mask_img and inputs[0].requires_grad:
            clean_img_feat, masked_img_feat, img_mask = self.img_masking(inputs[0])
            inputs[0] = masked_img_feat
        mask_pts = prob < 0.25
        if mask_pts and inputs[1].requires_grad:
            clean_pts_feat, masked_pts_feat, pts_mask = self.pts_masking(inputs[1])
            inputs[1] = masked_pts_feat
        
        if self.pos_emb:
            B,C,H,W = inputs[0].shape
            B,D,H,W = inputs[1].shape
            img_bev_pos_embeds = self.img_bev_embedding(pos2embed(self.coords_bev.to(inputs[0].device), num_pos_feats=C))
            img_bev_pos_embeds = img_bev_pos_embeds.view(H,W,-1).permute(2,0,1).repeat(B,1,1,1)
            inputs[0] = inputs[0] + img_bev_pos_embeds
            
            pts_bev_pos_embeds = self.pts_bev_embedding(pos2embed(self.coords_bev.to(inputs[1].device), num_pos_feats=D))
            pts_bev_pos_embeds = pts_bev_pos_embeds.view(H,W,-1).permute(2,0,1).repeat(B,1,1,1)
            inputs[1] = inputs[1] + pts_bev_pos_embeds

        for self_idx1 in range(self.num_layers):
            inputs[0] = self.img_self_attn_list[self_idx1](inputs[0])
        for self_idx2 in range(self.num_layers):
            inputs[1] = self.pts_self_attn_list[self_idx2](inputs[1])
        
        if mask_pts and inputs[1].requires_grad:
            pts_loss = F.l1_loss(inputs[1], clean_pts_feat, reduction='none')
            pts_loss = (pts_loss * pts_mask).sum() / (pts_mask.sum() + 1e-5) / 256
        if mask_img and inputs[0].requires_grad:
            img_loss = F.l1_loss(inputs[0], clean_img_feat, reduction='none')
            img_loss = (img_loss * img_mask).sum() / (img_mask.sum() + 1e-5) / 80
        for idx in range(self.num_layers):
            inputs = self.cross_attn_list[idx](inputs)
        if mask_pts and inputs[1].requires_grad:
            return self.conv(torch.cat(inputs, dim=1)), pts_loss
        if mask_img and inputs[0].requires_grad:
            return self.conv(torch.cat(inputs, dim=1)), img_loss
        return self.conv(torch.cat(inputs, dim=1)), False

@MODELS.register_module()
class ModalitySpecificLocalCrossAttentionMask(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mask_ratio=0.5, mask_pts=False, mask_img=False, num_layers=1, kernel_size=9, bn_momentum=0.1, bias='auto', pos_emb=False):
        super(ModalitySpecificLocalCrossAttentionMask, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lidar_hidden_channel = 256
        self.camera_hidden_channel = 80
        self.conv = nn.Sequential(nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        # self.pts_mask_token = nn.Parameter(torch.zeros(1, self.lidar_hidden_channel, 1, 1))
        # trunc_normal_(self.pts_mask_token, mean=0., std=.02)
        self.mask_ratio = mask_ratio
        self.mask_pts = mask_pts
        self.mask_img = mask_img
        self.num_layers=num_layers
        self.cross_attn_list=nn.ModuleList()
        in_channels_img = in_channels[0]
        in_channels_pts = in_channels[1]
        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            in_channels_pts,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            in_channels_img,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        for i in range(num_layers):
            self.cross_attn_list.append(ModalitySpecificLocalCrossAttentionlayer(in_channels, out_channels, kernel_size))
        self.pos_emb = pos_emb
        self.pts_bev_embedding = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        self.img_bev_embedding = nn.Sequential(
            nn.Linear(80 * 2, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, 80)
        )
        self.bn_momentum = bn_momentum
        self.init_weights()
        
    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
        
    def img_masking(self, img_feat):
        clean_img_feat = img_feat.clone().detach()
        BN, C, H, W = img_feat.shape
        self.img_mask_count = int(np.ceil(H*W * self.mask_ratio))
        img_mask_idx = np.random.permutation(H*W)[:self.img_mask_count]
        img_mask = np.zeros(H*W, dtype=int)
        img_mask[img_mask_idx] = 1
        img_mask = img_mask.reshape((H, W))
        img_mask = torch.tensor(img_mask).to(device=img_feat.device)
        # img_mask_tokens = self.img_mask_token.expand(BN,-1,H,W).to(device=img_feat.device)
        img_mask_tokens = torch.zeros(BN,C,H,W).to(device=img_feat.device)
        masked_img_feat = img_feat * (1-img_mask) + img_mask_tokens* img_mask
        return clean_img_feat, masked_img_feat, img_mask
    
    def pts_masking(self, pts_feat):
        clean_pts_feat = pts_feat.clone().detach()
        BN, C, H, W = pts_feat.shape
        self.pts_mask_count = int(np.ceil(H*W * self.mask_ratio))
        pts_mask_idx = np.random.permutation(H*W)[:self.pts_mask_count]
        pts_mask = np.zeros(H*W, dtype=int)
        pts_mask[pts_mask_idx] = 1
        pts_mask = pts_mask.reshape((H, W))
        pts_mask = torch.tensor(pts_mask).to(device=pts_feat.device)
        #pts_mask_tokens = self.pts_mask_token.expand(BN,-1,H,W).to(device=pts_feat.device)
        pts_mask_tokens = torch.zeros(BN,C,H,W).to(device=pts_feat.device)
        masked_pts_feat = pts_feat * (1-pts_mask) + pts_mask_tokens* pts_mask
        return clean_pts_feat, masked_pts_feat, pts_mask
    @property
    def coords_bev(self):
        grid_size = [1440,1440]
        downsample_scale = 8
        x_size, y_size = (
            grid_size[1] // downsample_scale,
            grid_size[0] // downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        return coord_base
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        inputs[0] = self.shared_conv_img(inputs[0])
        inputs[1] = self.shared_conv_pts(inputs[1])
        prob = np.random.uniform()

        mask_img = prob < 0.5 and prob > 0.25
        if mask_img and inputs[0].requires_grad:
            clean_img_feat, masked_img_feat, img_mask = self.img_masking(inputs[0])
            inputs[0] = masked_img_feat
        mask_pts = prob < 0.25
        if mask_pts and inputs[1].requires_grad:
            clean_pts_feat, masked_pts_feat, pts_mask = self.pts_masking(inputs[1])
            inputs[1] = masked_pts_feat
        
        if self.pos_emb:
            B,C,H,W = inputs[0].shape
            B,D,H,W = inputs[1].shape
            img_bev_pos_embeds = self.img_bev_embedding(pos2embed(self.coords_bev.to(inputs[0].device), num_pos_feats=C))
            img_bev_pos_embeds = img_bev_pos_embeds.view(H,W,-1).permute(2,0,1).repeat(B,1,1,1)
            inputs[0] = inputs[0] + img_bev_pos_embeds
            
            pts_bev_pos_embeds = self.pts_bev_embedding(pos2embed(self.coords_bev.to(inputs[1].device), num_pos_feats=D))
            pts_bev_pos_embeds = pts_bev_pos_embeds.view(H,W,-1).permute(2,0,1).repeat(B,1,1,1)
            inputs[1] = inputs[1] + pts_bev_pos_embeds

        for idx in range(self.num_layers):
            inputs = self.cross_attn_list[idx](inputs)
        if mask_pts and inputs[1].requires_grad:
            pts_loss = F.l1_loss(inputs[1], clean_pts_feat, reduction='none')
            pts_loss = (pts_loss * pts_mask).sum() / (pts_mask.sum() + 1e-5) / 256
            return self.conv(torch.cat(inputs, dim=1)), pts_loss
        if mask_img and inputs[0].requires_grad:
            img_loss = F.l1_loss(inputs[0], clean_img_feat, reduction='none')
            img_loss = (img_loss * img_mask).sum() / (img_mask.sum() + 1e-5) / 80
            return self.conv(torch.cat(inputs, dim=1)), img_loss
        return self.conv(torch.cat(inputs, dim=1)), False
    

@MODELS.register_module()
class GatedNetwork(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(GatedNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convbnrelu = nn.Sequential(nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        concat_features = sum(in_channels)
        self.conv_cf1=nn.Conv2d(in_channels=concat_features,out_channels=1, kernel_size=3, padding=1)
        self.conv_cf2=nn.Conv2d(in_channels=concat_features,out_channels=1,kernel_size=3, padding=1)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, features):
        
        img_feat = features[0]
        lidar_feat = features[1]
        Gated_features = []  
        concat_feature = torch.cat([img_feat, lidar_feat], dim=1)
        
        conv_output1 = self.conv_cf1(concat_feature)
        conv_output2 = self.conv_cf2(concat_feature)
        sigmoid_cf1 = self.sigmoid(conv_output1)
        sigmoid_cf2 = self.sigmoid(conv_output2) 
        
        img_gated_feature= sigmoid_cf1 * img_feat
        pts_gated_feature= sigmoid_cf2 * lidar_feat
        Gated_features = torch.cat([img_gated_feature, pts_gated_feature], dim=1)
        output = self.convbnrelu(Gated_features)
        return output

@MODELS.register_module()
class TransFusionHead(nn.Module):

    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        decoder_layer=dict(),
        num_heads=8,
        nms_kernel_size=1,
        bn_momentum=0.1,
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        bias='auto',
        # loss
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean'),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super(TransFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        layers = []
        layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
        layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(MODELS.build(decoder_layer))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                SeparateHead(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                ))

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training # noqa: E501
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, metas):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        fusion_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        fusion_feat_flatten = fusion_feat.view(batch_size,
                                               fusion_feat.shape[1],
                                               -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)

        #################################
        # query initialization
        #################################
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(
                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(
                heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg[
                'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(
                heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(
                heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = fusion_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, fusion_feat_flatten.shape[1], -1),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(
            top_proposals_class,
            num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        #################################
        # transformer decoder layer (Fusion feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat,
                key=fusion_feat_flatten,
                query_pos=query_pos,
                key_pos=bev_pos)

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer['center'] = res_layer['center'] + query_pos.permute(
                0, 2, 1)
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index[:,
                                      None, :].expand(-1, self.num_classes,
                                                      -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]['dense_heatmap'] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in [
                    'dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score'
            ]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second
            index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats, [metas])
        assert len(res) == 1, 'only support one level features.'
        return res

    def predict(self, batch_feats, batch_input_metas):
        preds_dicts = self(batch_feats, batch_input_metas)
        res = self.predict_by_feat(preds_dicts, batch_input_metas)
        return res

    def predict_by_feat(self,
                        preds_dicts,
                        metas,
                        img=None,
                        rescale=False,
                        for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer
            & each batch.
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid()) # noqa: E501
            one_hot = F.one_hot(
                self.query_labels,
                num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0][
                'query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=['pedestrian'],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=['traffic_cone'],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[0],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[1],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[2],
                        radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                # adopt circle nms for different categories
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    ))
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.
                                    test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)

                temp_instances = InstanceData()
                temp_instances.bboxes_3d = metas[0]['box_type_3d'](
                    ret['bboxes'], box_dim=ret['bboxes'].shape[-1])
                temp_instances.scores_3d = ret['scores']
                temp_instances.labels_3d = ret['labels'].int()

                ret_layer.append(temp_instances)

            rets.append(ret_layer)
        assert len(
            rets
        ) == 1, f'only support one layer now, but get {len(rets)} layers'

        return rets[0]

    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    preds_dict: List[dict]):
        """Generate training targets.
        Args:
            batch_gt_instances_3d (List[InstanceData]):
            preds_dict (list[dict]): The prediction results. The index of the
                list is the index of layers. The inner dict contains
                predictions of one mini-batch:
                - center: (bs, 2, num_proposals)
                - height: (bs, 1, num_proposals)
                - dim: (bs, 3, num_proposals)
                - rot: (bs, 2, num_proposals)
                - vel: (bs, 2, num_proposals)
                - cls_logit: (bs, num_classes, num_proposals)
                - query_score: (bs, num_classes, num_proposals)
                - heatmap: The original heatmap before fed into transformer
                    decoder, with shape (bs, 10, h, w)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)
                    [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(batch_gt_instances_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                preds = []
                for i in range(self.num_decoder_layers):
                    pred_one_layer = preds_dict[i][key][batch_idx:batch_idx +
                                                        1]
                    preds.append(pred_one_layer)
                pred_dict[key] = torch.cat(preds)
            list_of_pred_dict.append(pred_dict)

        assert len(batch_gt_instances_3d) == len(list_of_pred_dict)
        res_tuple = multi_apply(
            self.get_targets_single,
            batch_gt_instances_3d,
            list_of_pred_dict,
            np.arange(len(batch_gt_instances_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): ground truth of instances.
            preds_dict (dict): dict of prediction result for a single sample.
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask) [1,
                    num_proposals] # noqa: E501
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
                - torch.Tensor: heatmap targets.
        """
        # 1. Assignment
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height,
            vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign separately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals *
                                                idx_layer:self.num_proposals *
                                                (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals *
                                idx_layer:self.num_proposals *
                                (idx_layer + 1), ]

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )

        # 2. Sampling. Compatible with the interface of `PseudoSampler` in
        # mmdet.
        gt_instances, pred_instances = InstanceData(
            bboxes=gt_bboxes_tensor), InstanceData(priors=bboxes_tensor)
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   pred_instances,
                                                   gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # 3. Create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression
        # and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = (grid_size[:2] // self.train_cfg['out_size_factor']
                            )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1],
                                         feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                          self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                          self.train_cfg['out_size_factor'])

                center = torch.tensor([coor_x, coor_y],
                                      dtype=torch.float32,
                                      device=device)
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
                # NOTE: fix
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                      center_int[[1, 0]], radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    def loss(self, batch_feats, batch_data_samples):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        batch_input_metas, batch_gt_instances_3d = [], []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
        preds_dicts = self(batch_feats, batch_input_metas)
        loss = self.loss_by_feat(preds_dicts, batch_gt_instances_3d)

        return loss

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(batch_gt_instances_3d, preds_dicts[0])
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict['dense_heatmap']).float(),
            heatmap.float(),
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(
                self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                    idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[..., idx_layer *
                                  self.num_proposals:(idx_layer + 1) *
                                  self.num_proposals, ].reshape(-1)
            layer_label_weights = label_weights[
                ..., idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals, ].reshape(-1)
            layer_score = preds_dict['heatmap'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(
                -1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score.float(),
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            layer_center = preds_dict['center'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_height = preds_dict['height'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_rot = preds_dict['rot'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            layer_dim = preds_dict['dim'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot],
                dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, ]
                preds = torch.cat([
                    layer_center, layer_height, layer_dim, layer_rot, layer_vel
                ],
                                  dim=1).permute(
                                      0, 2,
                                      1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(  # noqa: E501
                code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_loss_bbox = self.loss_bbox(
                preds,
                layer_bbox_targets,
                layer_reg_weights,
                avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict