"""Base MMDataset for Detection Task."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from mmcv.transforms import Compose
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS
from mmdet.structures.mask.structures import PolygonMasks

from otx.v2.api.entities.dataset_item import DatasetItemEntity
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.label import Domain, LabelEntity
from otx.v2.api.entities.utils.data_utils import get_old_new_img_indices
from otx.v2.api.entities.utils.shape_factory import ShapeFactory


# pylint: disable=invalid-name, too-many-locals, too-many-instance-attributes, super-init-not-called
def get_annotation_mmdet_format(
    dataset_item: DatasetItemEntity,
    labels: list[LabelEntity],
    domain: Domain,
    min_size: int = -1,
) -> dict:
    """Function to convert a OTX annotation to mmdetection format.

    This is used both in the OTXDataset class defined in
    this file as in the custom pipeline element 'LoadAnnotationFromOTXDataset'

    :param dataset_item: DatasetItem for which to get annotations
    :param labels: List of labels that are used in the task
    :return dict: annotation information dict in mmdet format
    """
    width, height = dataset_item.width, dataset_item.height

    # load annotations for item
    gt_bboxes = []
    gt_labels = []
    gt_polygons = []
    gt_ann_ids = []

    label_idx = {label.id: i for i, label in enumerate(labels)}

    for annotation in dataset_item.get_annotations(labels=labels, include_empty=False, preserve_id=True):
        box = ShapeFactory.shape_as_rectangle(annotation.shape)

        if min(box.width * width, box.height * height) < min_size:
            continue

        class_indices = [
            label_idx[label.id] for label in annotation.get_labels(include_empty=False) if label.domain == domain
        ]

        n = len(class_indices)
        gt_bboxes.extend([[box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height] for _ in range(n)])
        if domain != Domain.DETECTION:
            polygon = ShapeFactory.shape_as_polygon(annotation.shape)
            polygon = np.array([p for point in polygon.points for p in [point.x * width, point.y * height]])
            gt_polygons.extend([[polygon] for _ in range(n)])
        gt_labels.extend(class_indices)
        item_id = getattr(dataset_item, "id_", None)
        gt_ann_ids.append((item_id, annotation.id_))

    if len(gt_bboxes) > 0:
        ann_info = {
            "bboxes": np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            "labels": np.array(gt_labels, dtype=int),
            "masks": PolygonMasks(gt_polygons, height=height, width=width) if gt_polygons else [],
            "ann_ids": gt_ann_ids,
        }
    else:
        ann_info = {
            "bboxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.array([], dtype=int),
            "masks": np.zeros((0, 1), dtype=np.float32),
            "ann_ids": [],
        }
    return ann_info


@DATASETS.register_module()
class OTXDetDataset(BaseDetDataset):
    """Wrapper that allows using a OTX dataset to train mmdetection models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX DatasetEntity object.

    The wrapper overwrites some methods of the BaseDetDataset class: prepare_train_img, prepare_test_img and prepipeline
    Naming of certain attributes might seem a bit peculiar but this is due to the conventions set in BaseDetDataset. For
    instance, CustomDatasets expects the dataset items to be stored in the attribute data_infos, which is why it is
    named like that and not dataset_items.

    """

    class _DataInfoProxy:
        """This class is intended to be a wrapper to use it in BaseDetDataset-derived class as `self.data_infos`.

        Instead of using list `data_infos` as in BaseDetDataset, our implementation of dataset OTXDataset
        uses this proxy class with overriden __len__ and __getitem__; this proxy class
        forwards data access operations to otx_dataset and converts the dataset items to the view
        convenient for mmdetection.

        Args:
            otx_dataset (DatasetEntity): DatasetEntity from dataset api
            labels (List[LabelEntity]): List of LabelEntity
        """

        def __init__(self, otx_dataset: DatasetEntity, labels: list[LabelEntity]) -> None:
            self.otx_dataset = otx_dataset
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}

        def __len__(self) -> int:
            return len(self.otx_dataset)

        def __getitem__(self, index: int) -> dict[str, Any]:
            """Prepare a dict 'data_info' that is expected by the mmdet pipeline to handle images and annotations.

            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            """
            dataset = self.otx_dataset
            item = dataset[index]
            ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

            height, width = item.height, item.width

            return {
                "dataset_item": item,
                "width": width,
                "height": height,
                "index": index,
                "ann_info": {"label_list": self.labels},
                "ignored_labels": ignored_labels,
                "bbox_fields": [],
                "mask_fields": [],
                "seg_fields": [],
            }

    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: list[LabelEntity],
        pipeline: Sequence[dict],
        test_mode: bool = False,
        max_refetch: int = 1000,
        **kwargs,
    ) -> None:
        """Initialize OTXDetDataset.

        Args:
            otx_dataset (DatasetEntiy): DatasetEntity from dataset api
            labels (List[LabelEntity]): List of LabelEntity
            pipeline (Sequence[dict]): List of data pipeline
            test_mode (bool): Whether current dataset is for test or not
            max_refetch (int): If ``Basedataset.prepare_data`` get a None img.
                The maximum extra number of cycles to get a valid
            kwargs: Additional kwargs
        """
        dataset_cfg = kwargs.copy()
        _ = dataset_cfg.pop("org_type", None)
        new_classes = dataset_cfg.pop("new_classes", [])
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.CLASSES = [label.name for label in labels]
        self.domain = self.labels[0].domain
        self.test_mode = test_mode
        self.max_refetch = max_refetch

        self._metainfo = {"classes": self.CLASSES, "domain": self.domain}

        # Instead of using list data_infos as in BaseDetDataset, this implementation of dataset
        # uses a proxy class with overriden __len__ and __getitem__; this proxy class
        # forwards data access operations to otx_dataset.
        # Note that list `data_infos` cannot be used here, since OTX dataset class does not have interface to
        # get only annotation of a data item, so we would load the whole data item (including image)
        # even if we need only checking aspect ratio of the image; due to it
        # this implementation of dataset does not uses such tricks as skipping images with wrong aspect ratios or
        # small image size, since otherwise reading the whole dataset during initialization will be required.
        self.data_list: Any = OTXDetDataset._DataInfoProxy(otx_dataset, labels)

        self.proposals = None  # Attribute expected by mmdet but not used for OTX datasets

        if not test_mode:
            self.img_indices = get_old_new_img_indices(self.labels, new_classes, self.otx_dataset)

        self.pipeline = Compose(pipeline)
        self.serialize_data = None  # OTX has own data caching mechanism
        self._fully_initialized = False
        self.full_init()

    def full_init(self) -> None:
        """OTXDetDataset do not have difference between full init and lazy init.

        It is for compatibility with MMEngine
        """
        if self._fully_initialized:
            return

        self._fully_initialized = True

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data_list)

    def get_ann_info(self, idx: int) -> dict:
        """This method is used for evaluation of predictions.

        The BaseDetDataset class implements a method
        BaseDetDataset.evaluate, which uses the class method get_ann_info to retrieve annotations.

        :param idx: index of the dataset item for which to get the annotations
        :return ann_info: dict that contains the coordinates of the bboxes and their corresponding labels
        """
        dataset_item = self.otx_dataset[idx]
        labels = self.labels
        return get_annotation_mmdet_format(dataset_item, labels, self.domain)
