"""Custom ATSS head for OTX template."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Optional

import torch
from mmdet.models.dense_heads.atss_head import ATSSHead
from mmdet.models.task_modules.prior_generators import anchor_inside_flags
from mmdet.models.utils.misc import multi_apply, unmap
from mmdet.registry import MODELS
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
from mmdet.utils import InstanceList, OptInstanceList
from mmdet.utils.dist_utils import reduce_mean
from mmengine.structures import InstanceData
from torch import Tensor

from otx.algorithms.detection.adapters.mmdet.models.heads.cross_dataset_detector_head import (
    CrossDatasetDetectorHead,
    TrackingLossDynamicsMixIn,
)
from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import TrackingLossType
from otx.algorithms.detection.adapters.mmdet.models.losses.cross_focal_loss import (
    CrossSigmoidFocalLoss,
)

EPS = 1e-12

# pylint: disable=too-many-arguments, too-many-locals,


@MODELS.register_module()
class CustomATSSHead(CrossDatasetDetectorHead, ATSSHead):
    """CustomATSSHead for OTX template."""

    def __init__(self, *args, bg_loss_weight=-1.0, use_qfl=False, qfl_cfg=None, **kwargs):
        if use_qfl:
            kwargs["loss_cls"] = (
                qfl_cfg
                if qfl_cfg
                else dict(
                    type="QualityFocalLoss",
                    use_sigmoid=True,
                    beta=2.0,
                    loss_weight=1.0,
                )
            )
        super().__init__(*args, **kwargs)
        self.bg_loss_weight = bg_loss_weight
        self.use_qfl = use_qfl

    def loss_by_feat(
        self, cls_scores, bbox_preds, centernesses, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore=None
    ):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
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
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )

        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor,
            valid_label_mask,
        ) = cls_reg_targets
        avg_factor = reduce_mean(torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor = multi_apply(
            self.loss_by_feat_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            centernesses,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            valid_label_mask,
            avg_factor=avg_factor,
        )

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_centerness=loss_centerness)

    def loss_by_feat_single(
        self,
        anchors,
        cls_score,
        bbox_pred,
        centerness,
        labels,
        label_weights,
        bbox_targets,
        valid_label_mask,
        avg_factor,
    ):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centerness(Tensor): Centerness scores for each scale level.
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            avg_factor (float): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.
            valid_label_mask (Tensor): Label mask for consideration of ignored
                label with shape (N, num_total_anchors, 1).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        valid_label_mask = valid_label_mask.reshape(-1, self.cls_out_channels)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = self._get_pos_inds(labels)

        if self.use_qfl:
            quality = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(pos_anchors, pos_bbox_targets)
            if self.reg_decoded_bbox:
                pos_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)

            if self.use_qfl:
                quality[pos_inds] = bbox_overlaps(pos_bbox_pred.detach(), pos_bbox_targets, is_aligned=True).clamp(
                    min=1e-6
                )

            # regression loss
            loss_bbox = self._get_loss_bbox(pos_bbox_targets, pos_bbox_pred, centerness_targets)

            # centerness loss
            loss_centerness = self._get_loss_centerness(avg_factor, pos_centerness, centerness_targets)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.0)

        # Re-weigting BG loss
        if self.bg_loss_weight >= 0.0:
            neg_indices = labels == self.num_classes
            label_weights[neg_indices] = self.bg_loss_weight

        if self.use_qfl:
            labels = (labels, quality)  # For quality focal loss arg spec

        # classification loss
        loss_cls = self._get_loss_cls(cls_score, labels, label_weights, valid_label_mask, avg_factor)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def _get_pos_inds(self, labels):
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        return pos_inds

    def _get_loss_cls(self, cls_score, labels, label_weights, valid_label_mask, avg_factor):
        if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=avg_factor, valid_label_mask=valid_label_mask
            )
        else:
            loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor)
        return loss_cls

    def _get_loss_centerness(self, avg_factor, pos_centerness, centerness_targets):
        return self.loss_centerness(pos_centerness, centerness_targets, avg_factor=avg_factor)

    def _get_loss_bbox(self, pos_bbox_targets, pos_bbox_pred, centerness_targets):
        return self.loss_bbox(pos_bbox_pred, pos_bbox_targets, weight=centerness_targets, avg_factor=1.0)

    def get_targets(
        self,
        anchor_list: List[List[Tensor]],
        valid_flag_list: List[List[Tensor]],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        unmap_outputs: bool = True,
    ):
        """Get targets for Detection head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        However, if the detector's head loss uses CrossSigmoidFocalLoss,
        the labels_weights_list consists of (binarized label schema * weights) of batch images
        """
        return self.get_atss_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs,
        )


@MODELS.register_module()
class CustomATSSHeadTrackingLossDynamics(TrackingLossDynamicsMixIn, CustomATSSHead):
    """CustomATSSHead which supports tracking loss dynamics."""

    tracking_loss_types = (TrackingLossType.cls, TrackingLossType.bbox, TrackingLossType.centerness)

    def __init__(self, *args, bg_loss_weight=-1, use_qfl=False, qfl_cfg=None, **kwargs):
        super().__init__(*args, bg_loss_weight=bg_loss_weight, use_qfl=use_qfl, qfl_cfg=qfl_cfg, **kwargs)

    @TrackingLossDynamicsMixIn._wrap_loss
    def loss_by_feat(
        self, cls_scores, bbox_preds, centernesses, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore=None
    ):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
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
            dict[str, Tensor]: A dictionary of loss components.
        """
        return super().loss_by_feat(
            cls_scores, bbox_preds, centernesses, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore
        )

    @TrackingLossDynamicsMixIn._wrap_loss_single
    def loss_by_feat_single(
        self,
        anchors,
        cls_score,
        bbox_pred,
        centerness,
        labels,
        label_weights,
        bbox_targets,
        avg_factors,
        valid_label_mask,
    ):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centerness(Tensor): Centerness scores for each scale level.
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            avg_factors (float): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.
            valid_label_mask (Tensor): Label mask for consideration of ignored
                label with shape (N, num_total_anchors, 1).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        return super().loss_by_feat_single(
            anchors,
            cls_score,
            bbox_pred,
            centerness,
            labels,
            label_weights,
            bbox_targets,
            avg_factors,
            valid_label_mask,
        )

    def _get_loss_cls(self, cls_score, labels, label_weights, valid_label_mask, avg_factor):
        if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                valid_label_mask=valid_label_mask,
                reduction_override="none",
            )
        else:
            loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor, reduction_override="none")

        self._store_loss_dyns(loss_cls[self.pos_inds].detach().mean(-1), TrackingLossType.cls)
        return self._postprocess_loss(loss_cls, self.loss_cls.reduction, avg_factor=avg_factor)

    def _get_loss_centerness(self, avg_factor, pos_centerness, centerness_targets):
        loss_centerness = self.loss_centerness(
            pos_centerness, centerness_targets, avg_factor=avg_factor, reduction_override="none"
        )
        self._store_loss_dyns(loss_centerness, TrackingLossType.centerness)
        return self._postprocess_loss(loss_centerness, self.loss_centerness.reduction, avg_factor=avg_factor)

    def _get_loss_bbox(self, pos_bbox_targets, pos_bbox_pred, centerness_targets):
        loss_bbox = self.loss_bbox(
            pos_bbox_pred, pos_bbox_targets, weight=centerness_targets, avg_factor=1.0, reduction_override="none"
        )
        self._store_loss_dyns(loss_bbox, TrackingLossType.bbox)
        return self._postprocess_loss(loss_bbox, self.loss_centerness.reduction, avg_factor=1.0)

    def _get_targets_single(
        self,
        flat_anchors: Tensor,
        valid_flags: Tensor,
        num_level_anchors: List[int],
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: Optional[InstanceData] = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """Compute regression, classification targets for anchors in a single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors (List[int]): Number of anchors of each scale
                level.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
                sampling_result (:obj:`SamplingResult`): Sampling results.
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
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(
            pred_instances, num_level_anchors_inside, gt_instances, gt_instances_ignore
        )

        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_priors, sampling_result.pos_gt_bboxes)

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
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
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

        return (anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    @TrackingLossDynamicsMixIn._wrap_get_targets(False)
    def get_targets(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Get targets for Detection head."""
        return super().get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            img_metas,
            gt_bboxes_ignore_list,
            gt_labels_list,
            label_channels,
            unmap_outputs,
        )
