# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any, Literal

from functools import partial

from torchmetrics.detection import MeanAveragePrecision as TorchmetricsMeanAveragePrecision


class MeanAveragePrecision(TorchmetricsMeanAveragePrecision):
    def __init__(
        self,
        box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
        iou_type: Literal["bbox", "segm"] | tuple[str] = "bbox",
        iou_thresholds: list[float] | None = None,
        rec_thresholds: list[float] | None = None,
        max_detection_thresholds: list[int] | None = None,
        class_metrics: bool = False,
        extended_summary: bool = False,
        average: Literal["macro", "micro"] = "macro",
        backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools",
        boundary_cpu_count: int = 4,
        **kwargs: Any,
    ):
        super().__init__(
            box_format=box_format,
            iou_type=iou_type,
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
            max_detection_thresholds=max_detection_thresholds,
            class_metrics=class_metrics,
            extended_summary=extended_summary,
            average=average,
            backend=backend,
            **kwargs,
        )
        self.boundary_cpu_count = int(boundary_cpu_count)

    @property
    def cocoeval(self) -> object:
        return partial(super().cocoeval, boundary_cpu_count=self.boundary_cpu_count)
