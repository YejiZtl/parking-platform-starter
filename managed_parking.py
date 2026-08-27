from __future__ import annotations

from typing import Any

import cv2
import lap
import numpy as np

from parking_geometry import (
    area_ratio,
    center_distance_ratio,
    load_parking_regions,
    load_roi,
    overlap_iou,
)
from ultralytics import solutions
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults


class ManagedParkingManagement(solutions.ParkingManagement):
    """Parking management with ROI filtering, duplicate removal, and stable slot states."""

    def __init__(
        self,
        *,
        roi_file: str | None = None,
        frame_shape: tuple[int, int] | None = None,
        calibration_reference_shape: tuple[int, int] | None = None,
        duplicate_iou: float = 0.7,
        duplicate_center_ratio: float = 0.2,
        duplicate_size_ratio: float = 1.35,
        slot_overlap_threshold: float = 0.5,
        one_vehicle_one_space: bool = True,
        show_uncertain_slots: bool = True,
        empty_confirmations: int = 2,
        occupied_confirmations: int = 1,
        **kwargs: Any,
    ) -> None:
        json_file = kwargs.get("json_file")
        super().__init__(**kwargs)
        if json_file:
            self.json = load_parking_regions(
                json_file,
                target_shape=frame_shape,
                reference_shape=calibration_reference_shape,
            )
        self.parking_roi = load_roi(
            roi_file,
            target_shape=frame_shape,
            reference_shape=calibration_reference_shape,
        )
        self.duplicate_iou = duplicate_iou
        self.duplicate_center_ratio = duplicate_center_ratio
        self.duplicate_size_ratio = duplicate_size_ratio
        self.slot_overlap_threshold = min(1.0, max(0.0, slot_overlap_threshold))
        self.one_vehicle_one_space = one_vehicle_one_space
        self.show_uncertain_slots = show_uncertain_slots
        self.uncertain_color = (0, 215, 255)
        self.empty_confirmations = max(1, empty_confirmations)
        self.occupied_confirmations = max(1, occupied_confirmations)
        self.slot_states: list[bool | None] = [None] * len(self.json)
        self.empty_streaks = [0] * len(self.json)
        self.occupied_streaks = [0] * len(self.json)
        self.filter_stats = {
            "raw_detections": 0,
            "outside_roi": 0,
            "duplicate_detections_removed": 0,
            "kept_detections": 0,
        }
        self.last_assignments: dict[int, dict[str, float | int | str]] = {}

    @staticmethod
    def _box_list(box) -> list[float]:
        if hasattr(box, "detach"):
            box = box.detach().cpu().tolist()
        return [float(value) for value in box]

    def _center_in_roi(self, box: list[float]) -> bool:
        if self.parking_roi is None:
            return True
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return cv2.pointPolygonTest(self.parking_roi, center, False) >= 0

    def extract_tracks(self, im0: np.ndarray) -> None:
        predict_args = {
            key: value
            for key, value in self.track_add_args.items()
            if key != "tracker" and value is not None
        }
        with self.profilers[0]:
            self.tracks = self.model.predict(
                source=im0,
                classes=self.classes,
                agnostic_nms=True,
                verbose=False,
                **predict_args,
            )[0]
        self.track_data = self.tracks.boxes
        if self.track_data is not None and len(self.track_data) > 0:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.confs = self.track_data.conf.cpu().tolist()
            self.track_ids = list(range(len(self.boxes)))
        else:
            self.boxes, self.clss, self.confs, self.track_ids = [], [], [], []

        if len(self.boxes) == 0:
            self.filter_stats = {
                "raw_detections": 0,
                "outside_roi": 0,
                "duplicate_detections_removed": 0,
                "kept_detections": 0,
            }
            return

        raw_count = len(self.boxes)
        box_lists = [self._box_list(self.get_enclosing_box(box)) for box in self.boxes]
        inside_indices = [
            index for index, box in enumerate(box_lists) if self._center_in_roi(box)
        ]

        kept_indices: list[int] = []
        duplicates_removed = 0
        for index in sorted(inside_indices, key=lambda item: self.confs[item], reverse=True):
            candidate = box_lists[index]
            duplicate = any(
                overlap_iou(candidate, box_lists[kept]) >= self.duplicate_iou
                and center_distance_ratio(candidate, box_lists[kept])
                <= self.duplicate_center_ratio
                and area_ratio(candidate, box_lists[kept]) <= self.duplicate_size_ratio
                for kept in kept_indices
            )
            if duplicate:
                duplicates_removed += 1
            else:
                kept_indices.append(index)

        self.boxes = self.boxes[kept_indices]
        self.clss = [self.clss[index] for index in kept_indices]
        self.track_ids = [self.track_ids[index] for index in kept_indices]
        self.confs = [self.confs[index] for index in kept_indices]
        self.filter_stats = {
            "raw_detections": raw_count,
            "outside_roi": raw_count - len(inside_indices),
            "duplicate_detections_removed": duplicates_removed,
            "kept_detections": len(kept_indices),
        }

    def _stable_state(self, index: int, observed: bool) -> bool:
        current = self.slot_states[index]
        if current is None:
            self.slot_states[index] = observed
            return observed

        if observed == current:
            self.empty_streaks[index] = 0
            self.occupied_streaks[index] = 0
            return current

        if observed:
            self.occupied_streaks[index] += 1
            self.empty_streaks[index] = 0
            if self.occupied_streaks[index] >= self.occupied_confirmations:
                self.slot_states[index] = True
                self.occupied_streaks[index] = 0
        else:
            self.empty_streaks[index] += 1
            self.occupied_streaks[index] = 0
            if self.empty_streaks[index] >= self.empty_confirmations:
                self.slot_states[index] = False
                self.empty_streaks[index] = 0
        return bool(self.slot_states[index])

    def _box_occupies_region(self, box, region_polygon: np.ndarray) -> bool:
        inside, overlap, _ = self._box_region_metrics(box, region_polygon)
        return inside or overlap >= self.slot_overlap_threshold

    def _box_region_metrics(
        self, box, region_polygon: np.ndarray
    ) -> tuple[bool, float, float]:
        x0, y0, x1, y1 = (float(value) for value in self.get_enclosing_box(box))
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        center_inside = cv2.pointPolygonTest(region_polygon, center, False) >= 0

        region = region_polygon.reshape((-1, 2)).astype(np.float32)
        rectangle = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
        )
        intersection_area, _ = cv2.intersectConvexConvex(region, rectangle)
        region_area = max(cv2.contourArea(region), 1.0)
        overlap = float(intersection_area) / region_area
        region_center = np.mean(region, axis=0)
        region_diagonal = max(float(np.linalg.norm(np.ptp(region, axis=0))), 1.0)
        center_distance = float(np.linalg.norm(np.asarray(center) - region_center)) / region_diagonal
        return center_inside, overlap, center_distance

    def _legacy_occupied_indices(self, region_polygons: list[np.ndarray]) -> set[int]:
        return {
            slot_index
            for slot_index, polygon in enumerate(region_polygons)
            if any(self._box_occupies_region(box, polygon) for box in self.boxes)
        }

    def _match_boxes_to_regions(
        self, region_polygons: list[np.ndarray]
    ) -> dict[int, dict[str, float | int | str]]:
        candidate_by_pair: dict[tuple[int, int], dict[str, float | int | str]] = {}

        for box_index, box in enumerate(self.boxes):
            metrics = [self._box_region_metrics(box, polygon) for polygon in region_polygons]
            inside_slots = [index for index, item in enumerate(metrics) if item[0]]

            if inside_slots:
                eligible_slots = inside_slots
            else:
                eligible_slots = [
                    index
                    for index, item in enumerate(metrics)
                    if item[1] >= self.slot_overlap_threshold
                ]

            confidence = float(self.confs[box_index]) if box_index < len(self.confs) else 0.0
            for slot_index in eligible_slots:
                center_inside, overlap, center_distance = metrics[slot_index]
                mode = "center" if center_inside else "overlap"
                score = (
                    (2.0 if center_inside else 0.0)
                    + overlap
                    + confidence * 0.01
                    - min(center_distance, 4.0) * 0.05
                )
                candidate_by_pair[(box_index, slot_index)] = {
                    "slot_index": slot_index,
                    "box_index": box_index,
                    "mode": mode,
                    "score": score,
                    "overlap": overlap,
                    "center_distance": center_distance,
                }

        if not candidate_by_pair:
            return {}

        invalid_cost = 1_000_000.0
        cost_matrix = np.full(
            (len(self.boxes), len(region_polygons)), invalid_cost, dtype=np.float64
        )
        for (box_index, slot_index), candidate in candidate_by_pair.items():
            cost_matrix[box_index, slot_index] = 10.0 - float(candidate["score"])

        _, box_to_slot, _ = lap.lapjv(
            cost_matrix,
            extend_cost=True,
            cost_limit=1000.0,
        )
        assignments: dict[int, dict[str, float | int | str]] = {}
        for box_index, slot_index in enumerate(box_to_slot):
            if slot_index < 0:
                continue
            candidate = candidate_by_pair.get((box_index, int(slot_index)))
            if candidate is None:
                continue
            slot_index = int(slot_index)
            assignments[slot_index] = candidate

        return assignments

    def process(self, im0: np.ndarray) -> SolutionResults:
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, self.line_width)
        occupied_slots = 0
        region_polygons = [
            np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            for region in self.json
        ]
        legacy_occupied = self._legacy_occupied_indices(region_polygons)

        self.last_assignments = self._match_boxes_to_regions(region_polygons)
        strict_occupied_indices = set(self.last_assignments)
        uncertain_indices = legacy_occupied - strict_occupied_indices
        observed_occupied_indices = (
            strict_occupied_indices if self.one_vehicle_one_space else legacy_occupied
        )

        center_matches = sum(
            assignment["mode"] == "center" for assignment in self.last_assignments.values()
        )
        overlap_matches = sum(
            assignment["mode"] == "overlap" for assignment in self.last_assignments.values()
        )
        self.filter_stats.update(
            {
                "legacy_occupied_slots": len(legacy_occupied),
                "matched_slots": len(strict_occupied_indices),
                "reported_occupied_slots": len(observed_occupied_indices),
                "center_matches": center_matches,
                "overlap_matches": overlap_matches,
                "unmatched_detections": max(0, len(self.boxes) - len(strict_occupied_indices)),
                "prevented_multi_slot_claims": max(
                    0, len(legacy_occupied) - len(strict_occupied_indices)
                ),
                "uncertain_slots": len(uncertain_indices),
                "uncertain_slot_ids": [index + 1 for index in sorted(uncertain_indices)],
            }
        )

        for index, region_polygon in enumerate(region_polygons):
            observed_occupied = index in observed_occupied_indices

            occupied = self._stable_state(index, observed_occupied)
            if occupied:
                occupied_slots += 1

            line_color = self.occ if occupied else self.arc
            if occupied and index in uncertain_indices and self.show_uncertain_slots:
                line_color = self.uncertain_color
            cv2.polylines(
                im0,
                [region_polygon],
                isClosed=True,
                color=line_color,
                thickness=2,
            )

        available_slots = len(self.json) - occupied_slots
        self.pr_info["Occupancy"] = occupied_slots
        self.pr_info["Available"] = available_slots
        self.pr_info["Uncertain"] = len(uncertain_indices)
        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)
        if self.parking_roi is not None:
            cv2.polylines(im0, [self.parking_roi], True, (0, 0, 255), 4, cv2.LINE_AA)

        plot_im = annotator.result()
        self.display_output(plot_im)
        return SolutionResults(
            plot_im=plot_im,
            filled_slots=occupied_slots,
            available_slots=available_slots,
            total_tracks=len(self.track_ids),
        )
