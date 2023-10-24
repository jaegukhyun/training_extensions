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

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from mmcv.transforms import Compose
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS
from mmdet.structures.mask.structures import PolygonMasks
from mmengine.config import Config

from otx.algorithms.common.utils.data import get_old_new_img_indices
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.subset import Subset
from otx.api.utils.shape_factory import ShapeFactory

from .tiling import Tile


# pylint: disable=invalid-name, too-many-locals, too-many-instance-attributes, super-init-not-called
def get_annotation_mmdet_format(
    dataset_item: DatasetItemEntity,
    labels: List[LabelEntity],
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
        ann_info = dict(
            bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(gt_labels, dtype=int),
            masks=PolygonMasks(gt_polygons, height=height, width=width) if gt_polygons else [],
            ann_ids=gt_ann_ids,
        )
    else:
        ann_info = dict(
            bboxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.array([], dtype=int),
            masks=np.zeros((0, 1), dtype=np.float32),
            ann_ids=[],
        )
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
        """

        def __init__(self, otx_dataset, labels):
            self.otx_dataset = otx_dataset
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}

        def __len__(self):
            return len(self.otx_dataset)

        def __getitem__(self, index):
            """Prepare a dict 'data_info' that is expected by the mmdet pipeline to handle images and annotations.

            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            """

            dataset = self.otx_dataset
            item = dataset[index]
            ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

            height, width = item.height, item.width

            data_info = dict(
                dataset_item=item,
                width=width,
                height=height,
                index=index,
                ann_info=dict(label_list=self.labels),
                ignored_labels=ignored_labels,
                bbox_fields=[],
                mask_fields=[],
                seg_fields=[],
            )

            return data_info

    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: List[LabelEntity],
        pipeline: Sequence[dict],
        test_mode: bool = False,
        max_refetch: int = 1000,
        **kwargs,
    ):
        dataset_cfg = kwargs.copy()
        _ = dataset_cfg.pop("org_type", None)
        new_classes = dataset_cfg.pop("new_classes", [])
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.CLASSES = list(label.name for label in labels)
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
        self.data_list = OTXDetDataset._DataInfoProxy(otx_dataset, labels)

        self.proposals = None  # Attribute expected by mmdet but not used for OTX datasets

        if not test_mode:
            self.img_indices = get_old_new_img_indices(self.labels, new_classes, self.otx_dataset)

        self.pipeline = Compose(pipeline)
        self.serialize_data = None  # OTX has own data caching mechanism
        self._fully_initialized = False
        self.full_init()

    def full_init(self):
        """OTXDetDataset do not have difference between full init and lazy init.

        It is for compatibility with MMEngine
        """
        if self._fully_initialized:
            return

        self._fully_initialized = True

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data_list)

    def get_ann_info(self, idx: int):
        """This method is used for evaluation of predictions.

        The BaseDetDataset class implements a method
        BaseDetDataset.evaluate, which uses the class method get_ann_info to retrieve annotations.

        :param idx: index of the dataset item for which to get the annotations
        :return ann_info: dict that contains the coordinates of the bboxes and their corresponding labels
        """
        dataset_item = self.otx_dataset[idx]
        labels = self.labels
        return get_annotation_mmdet_format(dataset_item, labels, self.domain)


# pylint: disable=too-many-arguments
@DATASETS.register_module()
class ImageTilingDataset(OTXDetDataset):
    """A wrapper of tiling dataset.

    Suitable for training small object dataset. This wrapper composed of `Tile`
    that crops an image into tiles and merges tile-level predictions to
    image-level prediction for evaluation.

    Args:
        dataset (Config): The dataset to be tiled.
        pipeline (List): Sequence of transform object or
            config dict to be composed.
        tile_size (int): the length of side of each tile
        min_area_ratio (float, optional): The minimum overlap area ratio
            between a tiled image and its annotations. Ground-truth box is
            discarded if the overlap area is less than this value.
            Defaults to 0.8.
        overlap_ratio (float, optional): ratio of each tile to overlap with
            each of the tiles in its 4-neighborhood. Defaults to 0.2.
        iou_threshold (float, optional): IoU threshold to be used to suppress
            boxes in tiles' overlap areas. Defaults to 0.45.
        max_per_img (int, optional): if there are more than max_per_img bboxes
            after NMS, only top max_per_img will be kept. Defaults to 200.
        max_annotation (int, optional): Limit the number of ground truth by
            randomly select 5000 due to RAM OOM. Defaults to 5000.
        sampling_ratio (flaot): Ratio for sampling entire tile dataset.
        include_full_img (bool): Whether to include full image in the dataset.
    """

    def __init__(
        self,
        dataset: Config,
        pipeline: List[dict],
        tile_size: int,
        min_area_ratio=0.8,
        overlap_ratio=0.2,
        iou_threshold=0.45,
        max_per_img=200,
        max_annotation=5000,
        filter_empty_gt=True,
        test_mode=False,
        sampling_ratio=1.0,
        include_full_img=False,
    ):
        self.dataset = DATASETS.build(dataset)
        self.CLASSES = self.dataset.CLASSES
        data_subset = self.dataset.otx_dataset[0].subset
        self.tile_dataset = Tile(
            self.dataset,
            pipeline,
            tile_size=tile_size,
            overlap=overlap_ratio,
            min_area_ratio=min_area_ratio,
            iou_threshold=iou_threshold,
            max_per_img=max_per_img,
            max_annotation=max_annotation,
            filter_empty_gt=filter_empty_gt if data_subset != Subset.TESTING else False,
            sampling_ratio=sampling_ratio if data_subset != Subset.TESTING else 1.0,
            include_full_img=include_full_img if data_subset != Subset.TESTING else True,
        )
        self.flag = np.zeros(len(self), dtype=np.uint8)
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.num_samples = len(self.dataset)  # number of original samples

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.tile_dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Get training/test tile.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
            True).
        """
        return self.pipeline(self.tile_dataset[idx])

    def get_ann_info(self, idx):
        """Get annotation information of a tile.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation information of a tile.
        """
        return self.tile_dataset.get_ann_info(idx)

    def merge(self, results) -> Union[List[Tuple[np.ndarray, list]], List[np.ndarray]]:
        """Merge tile-level results to image-level results.

        Args:
            results: tile-level results.

        Returns:
            merged_results (list[list | tuple]): Merged results of the dataset.
        """
        return self.tile_dataset.merge(results)

    def merge_vectors(self, feature_vectors: List[np.ndarray], dump_vectors: bool) -> Union[np.ndarray, List[None]]:
        """Merge tile-level feature vectors to image-level feature-vector.

        Args:
            feature_vectors (list[np.ndarray]): tile-level feature vectors.
            dump_vectors (bool): whether to dump vectors.

        Returns:
            merged_vectors (np.ndarray | List[None]): Merged vector for each image.
        """

        if dump_vectors:
            return self.tile_dataset.merge_vectors(feature_vectors)
        else:
            return [None] * self.num_samples

    def merge_maps(self, saliency_maps: List, dump_maps: bool) -> List:
        """Merge tile-level saliency maps to image-level saliency map.

        Args:
            saliency_maps (list[list | np.ndarray]): tile-level saliency maps.
            dump_maps (bool): whether to dump saliency maps.

        Returns:
            merged_maps (List[list | np.ndarray | None]): Merged saliency map for each image.
        """

        if dump_maps:
            return self.tile_dataset.merge_maps(saliency_maps)
        else:
            return [None] * self.num_samples

    def __del__(self):
        """Delete the temporary directory when the object is deleted."""
        if getattr(self, "tmp_dir", False):
            self.tmp_dir.cleanup()
