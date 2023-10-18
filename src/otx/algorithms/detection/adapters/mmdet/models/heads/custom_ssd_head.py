"""Custom SSD head for OTX template."""
# Copyright (C) 2018-2021 OpenMMLab
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional, Tuple, Union

import torch
from mmcv.cnn import build_activation_layer
from mmdet.models.dense_heads.ssd_head import SSDHead
from mmdet.models.losses import smooth_l1_loss
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.utils.misc import unmap
from mmdet.registry import MODELS
from mmdet.structures.bbox import BaseBoxes, get_box_tensor
from mmdet.utils import InstanceList, OptInstanceList
from mmengine.structures import InstanceData
from torch import Tensor, nn

from otx.algorithms.detection.adapters.mmdet.models.heads.cross_dataset_detector_head import TrackingLossDynamicsMixIn
from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import (
    TrackingLossType,
)

# pylint: disable=too-many-arguments, too-many-locals


@MODELS.register_module()
class CustomSSDHead(SSDHead):
    """CustomSSDHead class for OTX."""

    def __init__(self, *args, bg_loss_weight=-1.0, loss_cls=None, loss_balancing=False, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_cls is None:
            loss_cls = dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                reduction="none",
                loss_weight=1.0,
            )
        self.loss_cls = MODELS.build(loss_cls)
        self.bg_loss_weight = bg_loss_weight
        self.loss_balancing = loss_balancing
        if self.loss_balancing:
            self.loss_weights = torch.nn.Parameter(torch.FloatTensor(2))
            for i in range(2):
                self.loss_weights.data[i] = 0.0

    # TODO: remove this internal method
    # _init_layers of CustomSSDHead(this) and of SSDHead(parent)
    # Initialize almost the same model structure.
    # However, there are subtle differences
    # Theses differences make `load_state_dict_pre_hook()` go wrong
    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        act_cfg = self.act_cfg.copy()
        act_cfg.setdefault("inplace", True)
        for in_channel, num_base_priors in zip(self.in_channels, self.num_base_priors):
            if self.use_depthwise:
                activation_layer = build_activation_layer(act_cfg)

                self.reg_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                        nn.BatchNorm2d(in_channel),
                        activation_layer,
                        nn.Conv2d(in_channel, num_base_priors * 4, kernel_size=1, padding=0),
                    )
                )
                self.cls_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                        nn.BatchNorm2d(in_channel),
                        activation_layer,
                        nn.Conv2d(in_channel, num_base_priors * self.cls_out_channels, kernel_size=1, padding=0),
                    )
                )
            else:
                self.reg_convs.append(nn.Conv2d(in_channel, num_base_priors * 4, kernel_size=3, padding=1))
                self.cls_convs.append(
                    nn.Conv2d(in_channel, num_base_priors * self.cls_out_channels, kernel_size=3, padding=1)
                )

    def loss_by_feat_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        anchor: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        avg_factor: int,
    ) -> Tuple[Tensor, Tensor]:
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchor (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            Tuple[Tensor, Tensor]: A tuple of cls loss and bbox loss of one
            feature map.
        """

        # Re-weigting BG loss
        label_weights = label_weights.reshape(-1)
        if self.bg_loss_weight >= 0.0:
            neg_indices = labels == self.num_classes
            label_weights = label_weights.clone()
            label_weights[neg_indices] = self.bg_loss_weight

        loss_cls_all = self.loss_cls(cls_score, labels, label_weights)
        if len(loss_cls_all.shape) > 1:
            loss_cls_all = loss_cls_all.sum(-1)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = self._get_pos_inds(labels)
        neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls = self._get_loss_cls(avg_factor, loss_cls_all, pos_inds, topk_loss_cls_neg)

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        # TODO: We need to verify that this is working properly.
        # pylint: disable=redundant-keyword-arg
        loss_bbox = self._get_loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor)
        return loss_cls[None], loss_bbox

    def _get_pos_inds(self, labels):
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        return pos_inds

    def _get_loss_bbox(self, bbox_pred, bbox_targets, bbox_weights, avg_factor):
        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=avg_factor,
        )

        return loss_bbox

    def _get_loss_cls(self, avg_factor, loss_cls_all, pos_inds, topk_loss_cls_neg):
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / avg_factor
        return loss_cls

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, List[Tensor]]:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, list[Tensor]]: A dictionary of loss components. the dict
            has components below:

            - loss_cls (list[Tensor]): A list containing each feature map \
            classification loss.
            - loss_bbox (list[Tensor]): A list containing each feature map \
            regression loss.
        """
        losses = super().loss_by_feat(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore
        )
        losses_cls = losses["loss_cls"]
        losses_bbox = losses["loss_bbox"]

        if self.loss_balancing:
            losses_cls, losses_bbox = self._balance_losses(losses_cls, losses_bbox)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _balance_losses(self, losses_cls, losses_reg):
        loss_cls = sum(_loss.mean() for _loss in losses_cls)
        loss_cls = torch.exp(-self.loss_weights[0]) * loss_cls + 0.5 * self.loss_weights[0]

        loss_reg = sum(_loss.mean() for _loss in losses_reg)
        loss_reg = torch.exp(-self.loss_weights[1]) * loss_reg + 0.5 * self.loss_weights[1]

        return (loss_cls, loss_reg)


@MODELS.register_module()
class CustomSSDHeadTrackingLossDynamics(TrackingLossDynamicsMixIn, CustomSSDHead):
    """CustomSSDHead which supports tracking loss dynamics."""

    tracking_loss_types = (TrackingLossType.cls, TrackingLossType.bbox, TrackingLossType.centerness)

    @TrackingLossDynamicsMixIn._wrap_loss
    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, List[Tensor]]:
        """Compute loss from the head and prepare for loss dynamics tracking."""
        return super().loss_by_feat(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore
        )

    @TrackingLossDynamicsMixIn._wrap_loss_single
    def loss_by_feat_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        anchor: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        avg_factor: int,
    ) -> Tuple[Tensor, Tensor]:
        """Compute loss of a single image and increase `self.cur_loss_idx` counter for loss dynamics tracking."""
        return super().loss_by_feat_single(
            cls_score, bbox_pred, anchor, labels, label_weights, bbox_targets, bbox_weights, avg_factor
        )

    def _get_loss_cls(self, avg_factor, loss_cls_all, pos_inds, topk_loss_cls_neg):
        loss_cls_pos = loss_cls_all[pos_inds]
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos.sum() + loss_cls_neg) / avg_factor

        self._store_loss_dyns(loss_cls_pos.detach(), TrackingLossType.cls)
        return loss_cls

    def _get_loss_bbox(self, bbox_pred, bbox_targets, bbox_weights, avg_factor):
        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=avg_factor,
            reduction="none",
        )

        self._store_loss_dyns(loss_bbox[self.pos_inds].detach().mean(-1), TrackingLossType.bbox)
        return self._postprocess_loss(loss_bbox, reduction="mean", avg_factor=avg_factor)

    @TrackingLossDynamicsMixIn._wrap_get_targets(concatenate_last=True)
    def get_targets(
        self,
        anchor_list: List[List[Tensor]],
        valid_flag_list: List[List[Tensor]],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        unmap_outputs: bool = True,
        return_sampling_results: bool = False,
    ) -> tuple:
        """Compute regression and classification targets for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - avg_factor (int): Average factor that is used to average
                  the loss. When using sampling method, avg_factor is usually
                  the sum of positive and negative priors. When using
                  `PseudoSampler`, `avg_factor` is usually equal to the number
                  of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        return super().get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs,
            return_sampling_results,
        )

    def _get_targets_single(
        self,
        flat_anchors: Union[Tensor, BaseBoxes],
        valid_flags: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: Optional[InstanceData] = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """Compute regression and classification targets for anchors in a single image.

        Args:
            flat_anchors (Tensor or :obj:`BaseBoxes`): Multi-level anchors
                of the image, which are concatenated into a single tensor
                or box type of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors, ).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.  Defaults to True.

        Returns:
            tuple:

                - labels (Tensor): Labels of each level.
                - label_weights (Tensor): Label weights of each level.
                - bbox_targets (Tensor): BBox targets of each level.
                - bbox_weights (Tensor): BBox weights of each level.
                - pos_inds (Tensor): positive samples indexes.
                - neg_inds (Tensor): negative samples indexes.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags, img_meta["img_shape"][:2], self.train_cfg["allowed_border"]
        )
        if not inside_flags.any():
            raise ValueError(
                "There is no valid anchor inside the image boundary. Please "
                "check the image size and anchor sizes, or set "
                "``allowed_border`` to -1 to skip the condition."
            )
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags]

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances, gt_instances_ignore)
        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)

        num_valid_anchors = anchors.shape[0]
        target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox else self.bbox_coder.encode_size
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)

        # TODO: Considering saving memory, is it necessary to be long?
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # `bbox_coder.encode` accepts tensor or box type inputs and generates
        # tensor targets. If regressing decoded boxes, the code will convert
        # box type `pos_bbox_targets` to tensor.
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg["pos_weight"] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg["pos_weight"]
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        ########## What we changed from the original mmdet code ###############
        # Store all_pos_assigned_gt_inds to member variable
        # to look up training loss dynamics for each gt_bboxes afterwards
        pos_assigned_gt_inds = anchors.new_full((num_valid_anchors,), -1, dtype=torch.long)
        if len(pos_inds) > 0:
            pos_assigned_gt_inds[pos_inds] = (
                self.cur_batch_idx * self.max_gt_bboxes_len + sampling_result.pos_assigned_gt_inds
            )
        if unmap_outputs:
            pos_assigned_gt_inds = unmap(pos_assigned_gt_inds, num_total_anchors, inside_flags, fill=-1)
        self.pos_assigned_gt_inds_list += [pos_assigned_gt_inds]
        self.cur_batch_idx += 1
        ########################################################################

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result)
