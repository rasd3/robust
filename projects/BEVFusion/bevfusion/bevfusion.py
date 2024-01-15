from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        freeze_img=False,
        freeze_pts=False,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        imgpts_neck: Optional[dict] = None,
        masking_encoder: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        if 'pillarize_cfg' in data_preprocessor:
            pillarize_cfg = data_preprocessor.pop('pillarize_cfg')
        else:
            pillarize_cfg = False
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        if pillarize_cfg:
            self.pts_pillar_layer = Voxelization(**pillarize_cfg)
        else:
            self.pts_pillar_layer = False

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        self.imgpts_neck = MODELS.build(imgpts_neck) if imgpts_neck else None
        self.masking_encoder = MODELS.build(masking_encoder) if masking_encoder else None
        self.bbox_head = MODELS.build(bbox_head)
        self.freeze_img = freeze_img
        self.freeze_pts = freeze_pts
        
        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()
        if self.freeze_img:
            if self.img_backbone is not None:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False

        if self.freeze_pts:
            for name, param in self.named_parameters():
                if 'pts' in name and 'pts_bbox_head' not in name and 'imgpts_neck' not in name:
                    param.requires_grad = False
                if 'pts_bbox_head.decoder.0' in name:
                    param.requires_grad = False
                if 'imgpts_neck.shared_conv_pts' in name:
                    param.requires_grad = False
                if 'pts_bbox_head.heatmap_head' in name and 'pts_bbox_head.heatmap_head_img' not in name:
                    param.requires_grad = False
                if 'pts_bbox_head.prediction_heads.0' in name:
                    param.requires_grad = False
                if 'pts_bbox_head.class_encoding' in name:
                    param.requires_grad = False
            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
            self.pts_voxel_layer.apply(fix_bn)
            self.pts_voxel_encoder.apply(fix_bn)
            self.pts_middle_encoder.apply(fix_bn)
            self.pts_backbone.apply(fix_bn)
            self.pts_neck.apply(fix_bn)
            self.pts_bbox_head.heatmap_head.apply(fix_bn)
            self.pts_bbox_head.class_encoding.apply(fix_bn)
            self.pts_bbox_head.decoder[0].apply(fix_bn)
            self.pts_bbox_head.prediction_heads[0].apply(fix_bn)            
            self.imgpts_neck.shared_conv_pts.apply(fix_bn)
        
        
    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        pts_feats,
        pts_metas
    ) -> torch.Tensor:

        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        if self.imgpts_neck is not None:
            if self.masking_encoder is not None:
                x, pts_feats, mask_loss = self.masking_encoder(x, pts_feats, img_metas, pts_metas, self.imgpts_neck)
            else:    
                x, pts_feats = self.imgpts_neck(x, pts_feats, img_metas, pts_metas)
        x = x.view(B, int(BN / B), C, H, W)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        if self.masking_encoder is not None:
            return x, pts_feats, mask_loss
        else:
            return x, pts_feats, None

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            if self.pts_pillar_layer:
                pts_metas = self.voxelize(points, voxel_type='pillar')
            else:
                pts_metas = None
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x, pts_metas

    @torch.no_grad()
    def voxelize(self, points, voxel_type='voxel'):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            if voxel_type == 'voxel':
                ret = self.pts_voxel_layer(res)
            elif voxel_type == 'pillar':
                ret = self.pts_pillar_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n) # num_points
        feats = torch.cat(feats, dim=0) # voxels
        coords = torch.cat(coords, dim=0) # coors
        
        if voxel_type == 'pillar':
            pts_metas = {}
            pts_metas['pillars'] = feats
            pts_metas['pillar_coors'] = coords
            pts_metas['pts'] = points
        
        # HardSimpleVFE
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()
        if voxel_type == 'pillar':
            pts_metas['pillar_center'] = feats
            pts_metas['pillars_num_points'] = sizes
            return pts_metas
        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, _ = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        pts_feature, pts_metas = self.extract_pts_feat(batch_inputs_dict)
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature, pts_feature, mask_loss = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas,
                                                pts_feature,
                                                pts_metas)
            features.append(img_feature)
        features.append(pts_feature)
        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return x, mask_loss

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, mask_loss = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
        if mask_loss is not None:
            losses.update({'mask_loss':mask_loss})
        losses.update(bbox_loss)

        return losses