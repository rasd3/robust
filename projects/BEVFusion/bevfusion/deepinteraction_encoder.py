import torch
from mmcv.cnn import build_conv_layer
from torch import nn
from mmdet3d.registry import MODELS
from .encoder_utils import MMRI_I2P, LocalContextAttentionBlock, ConvBNReLU, MMRI_P2I
import pdb
import numpy as np
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
class DeepInteractionEncoderLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(DeepInteractionEncoderLayer, self ).__init__()
        self.I2P_block = MMRI_I2P(hidden_channel, hidden_channel, 0.1)
        # self.P_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        # self.P_out_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.P_integration = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)

        self.P2I_block = MMRI_P2I(hidden_channel, hidden_channel, 9)
        # self.I_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        # self.I_out_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.I_integration = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        
    def forward(self, img_feat, lidar_feat, img_metas, pts_metas):
        batch_size = lidar_feat.shape[0]
        BN, I_C, I_H, I_W = img_feat.shape
        I2P_feat = self.I2P_block(lidar_feat, img_feat.view(batch_size, -1, I_C, I_H, I_W), img_metas, pts_metas)
        # P2P_feat = self.P_IML(lidar_feat, lidar_feat)
        # P_Aug_feat = self.P_out_proj(torch.cat((I2P_feat, P2P_feat),dim=1))
        #new_lidar_feat = self.P_integration(torch.cat((P_Aug_feat, lidar_feat),dim=1))
        new_lidar_feat = self.P_integration(torch.cat((I2P_feat, lidar_feat),dim=1))

        P2I_feat = self.P2I_block(lidar_feat, img_feat.view(batch_size, -1, I_C, I_H, I_W), img_metas, pts_metas)
        # I2I_feat = self.I_IML(img_feat, img_feat)
        # I_Aug_feat = self.I_out_proj(torch.cat((P2I_feat.view(BN, -1, I_H, I_W), I2I_feat),dim=1))
        # new_img_feat = self.I_integration(torch.cat((I_Aug_feat, img_feat),dim=1))
        P2I_feat = P2I_feat.view(BN, -1, I_H, I_W)
        new_img_feat = self.I_integration(torch.cat((P2I_feat, img_feat),dim=1))
        return new_img_feat, new_lidar_feat

@MODELS.register_module()
class DeepInteractionEncoder(nn.Module):
    def __init__(self,
                num_layers=2,
                in_channels_img=64,
                in_channels_pts=128 * 3,
                hidden_channel=128,
                bn_momentum=0.1,
                bias='auto',
                ):
        super(DeepInteractionEncoder, self).__init__()

        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.num_layers = num_layers
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.fusion_blocks.append(DeepInteractionEncoderLayer(hidden_channel))

        self.bn_momentum = bn_momentum
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, img_feats, pts_feats, img_metas, pts_metas):
        new_img_feat = self.shared_conv_img(img_feats)
        new_pts_feat = self.shared_conv_pts(pts_feats)
        #pts_feat_conv = new_pts_feat.clone()
        for i in range(self.num_layers):
            new_img_feat, new_pts_feat = self.fusion_blocks[i](new_img_feat, new_pts_feat, img_metas, pts_metas)     
        #return new_img_feat, [pts_feat_conv, new_pts_feat]
        return new_img_feat, new_pts_feat


@MODELS.register_module()
class DeepInteractionEncoderMask(nn.Module):
    def __init__(self,
                 mask_ratio=0.5,
                 embed_dim=256,
                 pos_embed=True,
                ):
        super(DeepInteractionEncoderMask, self).__init__()
        self.img_H = 32
        self.img_W = 88
        self.pts_H = self.pts_W =  180
        self.img_token_count = self.img_H*self.img_W
        self.pts_token_count = self.pts_H*self.pts_W
        self.mask_ratio = mask_ratio
        self.img_mask_count = int(np.ceil(self.img_token_count * self.mask_ratio))
        self.pts_mask_count = int(np.ceil(self.pts_token_count * self.mask_ratio))
        self.embed_dim = embed_dim
        self.img_mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
        self.pts_mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
        trunc_normal_(self.img_mask_token, mean=0., std=.02)
        trunc_normal_(self.pts_mask_token, mean=0., std=.02)
        self.pos_embed=pos_embed
        if self.pos_embed:
            self.img_absolute_pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
            self.pts_absolute_pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
            trunc_normal_(self.img_absolute_pos_embed, std=.02)
            trunc_normal_(self.pts_absolute_pos_embed, std=.02)
        
    def img_masking(self, img_feats):
        clean_img_feats = img_feats.clone().detach()
        BN, C, H, W = img_feats.shape
        img_mask_idx = np.random.permutation(self.img_token_count)[:self.img_mask_count]
        img_mask = np.zeros(self.img_token_count, dtype=int)
        img_mask[img_mask_idx] = 1
        img_mask = img_mask.reshape((self.img_H, self.img_W))
        img_mask = torch.tensor(img_mask).to(device=img_feats.device)
        img_mask_tokens = self.img_mask_token.expand(BN,-1,H,W).to(device=img_feats.device)
        masked_img_feats = img_feats * (1-img_mask) + img_mask_tokens* img_mask
        return clean_img_feats, masked_img_feats, img_mask
    
    def pts_masking(self, pts_feats):
        clean_pts_feats = pts_feats.clone().detach()
        BN, C, H, W = pts_feats.shape
        pts_mask_idx = np.random.permutation(self.pts_token_count)[:self.pts_mask_count]
        pts_mask = np.zeros(self.pts_token_count, dtype=int)
        pts_mask[pts_mask_idx] = 1
        pts_mask = pts_mask.reshape((self.pts_H, self.pts_W))
        pts_mask = torch.tensor(pts_mask).to(device=pts_feats.device)
        pts_mask_tokens = self.pts_mask_token.expand(BN,-1,H,W).to(device=pts_feats.device)
        masked_pts_feats = pts_feats * (1-pts_mask) + pts_mask_tokens* pts_mask
        return clean_pts_feats, masked_pts_feats, pts_mask
    
    def forward(self, img_feats, pts_feats, img_metas, pts_metas, model):
        test_mode = pts_feats.requires_grad
        if test_mode:
            new_clean_img_feat, new_clean_pts_feat = model(img_feats, pts_feats, img_metas, pts_metas)
            return new_clean_img_feat, new_clean_pts_feat, None
        else:
            prob = np.random.uniform()
            mask_img = prob < 0.3
            mask_pts = prob > 0.3 and prob < 0.6
            clean = prob > 0.6
            if clean:
                new_clean_img_feat, new_clean_pts_feat = model(img_feats, pts_feats, img_metas, pts_metas)
                return new_clean_img_feat, new_clean_pts_feat, None
            clean_img_feats, masked_img_feats, img_mask = self.img_masking(img_feats)
            clean_pts_feats, masked_pts_feats, pts_mask = self.pts_masking(pts_feats)
            if self.pos_embed:
                masked_img_feats = masked_img_feats + self.img_absolute_pos_embed
                masked_pts_feats = masked_pts_feats + self.pts_absolute_pos_embed
            with torch.no_grad():
                new_clean_img_feat, new_clean_pts_feat = model(clean_img_feats, clean_pts_feats, img_metas, pts_metas)
            new_masked_img_feat, new_masked_pts_feat = model(masked_img_feats, masked_pts_feats, img_metas, pts_metas)
            if mask_img:
                img_loss = F.l1_loss(new_masked_img_feat, new_clean_img_feat, reduction='none')
                img_loss = (img_loss * img_mask).sum() / (img_mask.sum() + 1e-5) / self.embed_dim
                total_loss = img_loss
            elif mask_pts:
                pts_loss = F.l1_loss(new_masked_pts_feat, new_clean_pts_feat, reduction='none')
                pts_loss = (pts_loss * pts_mask).sum() / (pts_mask.sum() + 1e-5) / self.embed_dim
                total_loss = pts_loss
            return new_masked_img_feat, new_masked_pts_feat, total_loss