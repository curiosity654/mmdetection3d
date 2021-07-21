import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmseg.core import add_prefix
from mmseg.models import SEGMENTORS
from ..builder import build_backbone, build_head, build_neck
from .base import Base3DSegmentor

@SEGMENTORS.register_module()
class Cylinder3D(Base3DSegmentor):
    """3D Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be thrown during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Cylinder3D, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self._init_decode_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head, \
            '3D EncoderDecoder Segmentor should have a decode_head'

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = build_head(decode_head)
        self.num_classes = self.decode_head.num_classes

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(Cylinder3D, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, cylinder_fea, grid_ind):
        """Extract features from points."""
        x = self.backbone(cylinder_fea, grid_ind)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _decode_head_forward_train(self, x, img_metas, pts_semantic_mask, voxel_label, grid_ind):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     pts_semantic_mask,
                                                     voxel_label,
                                                     grid_ind,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def forward_train(self, voxel_feat, pts_semantic_mask, grid_ind, voxel_label, img_metas):
        """Forward function for training.

        Args:
            points (list[torch.Tensor]): List of points of shape [N, C].
            img_metas (list): Image metas.
            pts_semantic_mask (list[torch.Tensor]): List of point-wise semantic
                labels of shape [N].

        Returns:
            dict[str, Tensor]: Losses.
        """
        # extract features using backbone
        x = self.extract_feat(voxel_feat, grid_ind)

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas, pts_semantic_mask, voxel_label, grid_ind)
        losses.update(loss_decode)

        return losses
        

    def forward_test(self, voxel_feat, pts_semantic_mask, grid_ind, img_metas, rescale=False):
        """Forward function for training.

        Args:
            points (list[torch.Tensor]): List of points of shape [N, C].
            img_metas (list): Image metas.
            pts_semantic_mask (list[torch.Tensor]): List of point-wise semantic
                labels of shape [N].

        Returns:
            dict[str, Tensor]: Losses.
        """
        # forward the whole batch and split results
        x = self.extract_feat(voxel_feat, grid_ind)

        seg_logits = self._decode_head_forward_test(x, img_metas)
        seg_prob = F.softmax(seg_logits, dim=1)
        seg_map = seg_prob.argmax(1)  # [B, N]
        # to cpu tensor for consistency with det3d
        seg_map = seg_map.cpu()
        # TODO more elegant?
        # seg_pred = seg_map.split(1)[0]
        # warp in dict
        seg_pred = self.voxlabel2points(seg_map, grid_ind)

        return seg_pred

    def voxlabel2points(self, vox_label, grid_ind):
        point_labels = []
        for i in range(len(grid_ind)):
            point_label = vox_label[i][grid_ind[i][:, 0], grid_ind[i][:, 1], grid_ind[i][:, 2]]
            # idx = grid_ind[i][:,0] + grid_ind[i][:,1]*480 + grid_ind[i][:,2]*480*360
            # vox_label_f = torch.flatten(vox_label[i])
            # point_label = vox_label_f[idx]
            point_labels.append(dict(semantic_mask=point_label))

        return point_labels