from dataclasses import dataclass

import cv2


@dataclass
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int


def filter_bboxes_on_size(bounding_boxes, minh, maxh, minw, maxw):
    """Filters a list of bounding boxes based on their size"""
    regions = []
    for bbox in bounding_boxes:
        w = bbox.x1 - bbox.x0
        h = bbox.y1 - bbox.y0
        if w >= minw and w <= maxw and h >= minh and h <= maxh:
            regions.append(bbox)
    return regions


def filter_bboxes_intersections(bboxes, callback, *args, **kwargs):
    """
    Applies a filter to bboxes intersecting. Callback should accept the two bboxes
    and return the two new bboxes for each pair. Can return None to delete one or both.
    """

    need_filtering = False
    for i, bbox1 in enumerate(bboxes):
        if bbox1 is None:
            continue
        x0_1, y0_1, x1_1, y1_1 = bbox1.x0, bbox1.y0, bbox1.x1, bbox1.y1
        for j, bbox2 in enumerate(bboxes):
            if i != j and bbox2 is not None:
                x0_2, y0_2, x1_2, y1_2 = bbox2.x0, bbox2.y0, bbox2.x1, bbox2.y1

                if (x0_1 < x1_2 and x1_1 > x0_2) and (
                    y0_1 < y1_2 and y1_1 > y0_2
                ):  # Check for intersection
                    bboxes[i], bboxes[j] = callback(bbox1, bbox2, *args, **kwargs)
                    if not need_filtering and bboxes[i] is None or bboxes[j] is None:
                        need_filtering = True

    if need_filtering:
        bboxes = [bbox for bbox in bboxes if bbox is not None]
    return bboxes


def _general_region_intersection_callback(bbox1, bbox2):
    if (
        bbox1.x0 > bbox2.x0
        and bbox1.x1 < bbox2.x1
        and bbox1.y0 > bbox2.y0
        and bbox1.y1 < bbox2.y1
    ):  # concentric regions
        return bbox1, None
    elif (
        bbox1.x0 < bbox2.x1
        and bbox1.x1 > bbox2.x0
        and bbox1.y0 < bbox2.y1
        and bbox1.y1 > bbox2.y0
    ):  # intersecting regions
        return (
            BBox(
                min(bbox1.x0, bbox2.x0),
                min(bbox1.y0, bbox2.y0),
                max(bbox1.x1, bbox2.x1),
                max(bbox1.y1, bbox2.y1),
            ),
            None,
        )
    else:
        # should never be reached
        return bbox1, bbox2


def fix_generic_regions_intersections(regions):
    """Fixes generic regions when intersecting by keeping only the merged regions"""
    return filter_bboxes_intersections(regions, _general_region_intersection_callback)


def draw_boxes(img, boxes, color=(0, 0, 255), padding=True):
    for i, box in enumerate(boxes):
        pad = 0
        if padding:
            pad = i
            if i > len(boxes) / 2:
                i = -i
        img = cv2.rectangle(
            img.copy(),
            (box.x0 + pad, box.y0 + pad),
            (box.x1 - pad, box.y1 - pad),
            color,
            1,
        )
    return img


def sum_coordinates(bboxes, x0, y0):
    """Adds x0 and y0 to all the bboxes in-place"""
    for i in range(len(bboxes)):
        bboxes[i].x0 += x0
        bboxes[i].x1 += x0
        bboxes[i].y0 += y0
        bboxes[i].y1 += y0
