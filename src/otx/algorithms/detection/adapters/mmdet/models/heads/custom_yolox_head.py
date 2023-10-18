"""Custom YOLOX head for OTX template."""

from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from mmdet.models.dense_heads.yolox_head import YOLOXHead
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models.utils.misc import multi_apply
from mmdet.registry import MODELS
from mmdet.utils import OptInstanceList
from mmdet.utils.dist_utils import reduce_mean
from mmengine.structures import InstanceData
from torch import Tensor

from otx.algorithms.detection.adapters.mmdet.models.heads.cross_dataset_detector_head import (
    TrackingLossDynamicsMixIn,
)
from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import TrackingLossType


@MODELS.register_module()
class CustomYOLOXHead(YOLOXHead):
    """CustomYOLOXHead class for OTX."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@MODELS.register_module()
class CustomYOLOXHeadTrackingLossDynamics(TrackingLossDynamicsMixIn, CustomYOLOXHead):
    """CustomYOLOXHead which supports loss dynamics tracking."""

    tracking_loss_types = (TrackingLossType.cls, TrackingLossType.bbox)

    @TrackingLossDynamicsMixIn._wrap_loss
    def loss_by_feat(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        objectnesses: Sequence[Tensor],
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.


        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True
        )

        flatten_cls_preds: Union[List[Tensor], Tensor] = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_pred in cls_scores
        ]
        flatten_bbox_preds: Union[List[Tensor], Tensor] = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds
        ]
        flatten_objectness: Union[List[Tensor], Tensor] = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        # Init variables for loss dynamics tracking
        self.cur_batch_idx = 0
        self.max_gt_bboxes_len = max([len(batch_gt_instance.bboxes) for batch_gt_instance in batch_gt_instances])

        (
            pos_masks,
            cls_targets,
            obj_targets,
            bbox_targets,
            l1_targets,
            num_fg_imgs,
            pos_assigned_gt_inds_list,
        ) = multi_apply(
            self._get_targets_single,
            flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
            flatten_cls_preds.detach(),
            flatten_bboxes.detach(),
            flatten_objectness.detach(),
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(sum(num_fg_imgs), dtype=torch.float, device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)

        # For storing loss dynamics
        pos_assigned_gt_inds = torch.cat(pos_assigned_gt_inds_list, 0)
        self.batch_inds = pos_assigned_gt_inds // self.max_gt_bboxes_len
        self.bbox_inds = pos_assigned_gt_inds % self.max_gt_bboxes_len

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_bbox = (
            self.loss_bbox(
                flatten_bboxes.view(-1, 4)[pos_masks],
                bbox_targets,
                reduction_override="none",
            )
            / num_total_samples
        )
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets) / num_total_samples
        loss_cls = (
            self.loss_cls(
                flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
                cls_targets,
                reduction_override="none",
            )
            / num_total_samples
        )
        self._store_loss_dyns(loss_bbox, TrackingLossType.bbox)
        self._store_loss_dyns(loss_cls.mean(-1), TrackingLossType.cls)

        loss_bbox = weight_reduce_loss(loss_bbox, reduction=self.loss_bbox.reduction)
        loss_cls = weight_reduce_loss(loss_cls, reduction=self.loss_cls.reduction)

        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            loss_l1 = self.loss_l1(flatten_bbox_preds.view(-1, 4)[pos_masks], l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict

    @torch.no_grad()
    def _get_targets_single(
        self,
        priors: Tensor,
        cls_preds: Tensor,
        decoded_bboxes: Tensor,
        objectness: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: Optional[InstanceData] = None,
    ) -> tuple:
        """Compute classification, regression, and objectness targets for priors in a single image.

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        """

        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target, l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
        pred_instances = InstanceData(bboxes=decoded_bboxes, scores=scores.sqrt_(), priors=offset_priors)
        assign_result = self.assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances, gt_instances_ignore=gt_instances_ignore
        )

        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels, self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1

        pos_assigned_gt_inds = self.cur_batch_idx * self.max_gt_bboxes_len + sampling_result.pos_assigned_gt_inds
        self.cur_batch_idx += 1
        self.pos_inds = pos_inds

        return (
            foreground_mask,
            cls_target,
            obj_target,
            bbox_target,
            l1_target,
            num_pos_per_img,
            pos_assigned_gt_inds,
        )
