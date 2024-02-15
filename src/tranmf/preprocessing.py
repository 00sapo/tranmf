from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import filters, segmentation

from . import bboxes as bb


def write_image(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)


DEGAN_INSTANCE = None


def run_degan(img):
    img_ = img.astype(np.float32) / 255
    # resize to half size for faster processing and better results
    img_ = cv2.resize(img_, (img.shape[1] // 2, img.shape[0] // 2))
    global DEGAN_INSTANCE
    if DEGAN_INSTANCE is None:
        from degan import DEGAN

        DEGAN_INSTANCE = DEGAN(deb_weights=None, wat_weights=None)
        DEGAN_INSTANCE.load_weights()

    # FIX: the next line takes too much CPU ram
    img_ = DEGAN_INSTANCE.binarize(img_)
    # resize to original size
    img_ = cv2.resize(img_, (img.shape[1], img.shape[0]))
    img_ = img_ <= 0.5
    return img_


def calculate_channel_mask(segmentation_method, channel):
    # Apply median-blur to remove noise
    channel = cv2.medianBlur(channel, 3)

    thresholds = segmentation_method(channel)
    if isinstance(thresholds, np.ndarray):
        # the method returned the mask itself (e.g. DE-GAN)
        mask = thresholds
    else:
        mask = channel < thresholds

    return mask


def combine_masks(masks, operation_mode):
    if operation_mode == "vote":
        mask = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            mask += m
        mask = mask >= len(masks) / 2
        mask = mask.astype(bool)
    elif operation_mode in ["max", "none", "lightness", "gray"]:
        mask = masks[0]
    else:
        raise ValueError(
            "operation_mode must be one of 'vote', 'max', 'lightness', 'none', 'gray'"
        )
    return mask


def remove_isolated_regions(mask, size, distance):
    """
    Removes isolated regions from a mask. Isolated regions are True regions that are
    surrounded by False pixels.

    The minimum size of the isolated regions is given by `size` and the minimum
    distance number of surrounding False pixels is given by `distance`.
    """
    # remove small regions
    kernel = np.ones((size, size), np.uint8)
    no_small_regions = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
    ).astype(bool)
    diff = mask ^ no_small_regions

    # select regions surrounded by at least distance False pixels
    kernel = np.ones((distance + size, distance + size), np.uint8)
    distance_mask = cv2.dilate(no_small_regions.astype(np.uint8), kernel).astype(bool)

    # combine the two masks
    mask[diff & ~distance_mask] = False


def ink_postprocessing(mask, debug_dir=None):
    remove_isolated_regions(mask, 2, 4)
    if debug_dir:
        write_image(mask.astype(np.uint8) * 255, debug_dir / "isolated_ragions_2.png")
    remove_isolated_regions(mask, 3, 9)
    if debug_dir:
        write_image(mask.astype(np.uint8) * 255, debug_dir / "isolated_ragions_3.png")
    return mask


def find_ink(
    img: np.ndarray,
    segmentation_method: str,
    operation_mode: str,
    debug_dir: Path = None,
) -> np.ndarray:
    """
    Finds the ink in an image. The ink is usually defined as the darker area of the image.

    Arguments:
        img: the image as a BGR numpy array
        segmentation_method: the method to use for segmentation, from skimage.filters.threshold_*
        operation_mode: the operation to use for combining the channels, one of "max",
            "lightness", "vote", "gray", or "none". In the latter case, the input image
            whould be a single channel image
    """
    if segmentation_method == "degan":
        segmentation_function = run_degan
    else:
        segmentation_function = getattr(filters, "threshold_" + segmentation_method)
    masks = []

    # Add L* to BGR
    if operation_mode == "lightness":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0][:, :, None]
    elif operation_mode == "max":
        img = np.amax(img, 2)[..., None]
    elif operation_mode == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
    elif operation_mode == "none":
        if img.ndim == 2:
            img = img[..., None]

    for channel in range(img.shape[2]):
        channel_mask = calculate_channel_mask(
            segmentation_function,
            img[:, :, channel],
        )
        masks.append(channel_mask)

    mask = combine_masks(masks, operation_mode)

    # Apply morphing for removing isolated pixels
    mask = ink_postprocessing(mask, debug_dir=debug_dir)
    if debug_dir:
        write_image(mask.astype(np.uint8) * 255, debug_dir / "ink.png")
    return mask


def remove_borders(
    foreground_mask: np.ndarray, border_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Remove borders from a mask with True where is foreground and returns a binary
    image with 255 where the border was"""
    gray = foreground_mask.astype(np.uint8) * 255
    border_size = round(border_size * gray.shape[0])
    noborders = segmentation.clear_border(gray, buffer_size=border_size, bgval=0)
    return noborders


def crop_easy_borders(img: np.ndarray, background_color=(255, 255, 255)):
    """Crops an image to its content. Simple method based on exact matching with
    background_color"""
    easy_content_mask = np.argwhere(img != background_color)
    # a bbox here is (x0, y0, x1, y1)
    content_box = bb.BBox(
        np.min(easy_content_mask[:, 1]),
        np.min(easy_content_mask[:, 0]),
        np.max(easy_content_mask[:, 1]),
        np.max(easy_content_mask[:, 0]),
    )

    return (
        img[content_box.y0 : content_box.y1, content_box.x0 : content_box.x1],
        content_box,
    )


def find_content(noborders, morf1, morf2, debug_dir=None):
    """Applies morphological operations to the image and uses the connected components
    as the content"""
    # dir1
    kernel_h = morf1 * noborders.shape[0]
    kernel_w = morf2 * noborders.shape[1]
    ksize = (round(kernel_h), round(kernel_w))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    connected1 = cv2.morphologyEx(noborders, cv2.MORPH_CLOSE, kernel)
    if debug_dir:
        write_image(connected1, debug_dir / "connected1.png")

    # dir2
    kernel_h = morf2 * noborders.shape[0]
    kernel_w = morf1 * noborders.shape[1]
    ksize = (round(kernel_w), round(kernel_h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    connected2 = cv2.morphologyEx(connected1, cv2.MORPH_CLOSE, kernel)
    if debug_dir:
        write_image(connected2, debug_dir / "connected2.png")

    # get bounding boxes of connected components
    contours, hierarchy = cv2.findContours(
        connected2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append(bb.BBox(x, y, x + w, y + h))
    return bounding_boxes


def generic_analysis(
    img: np.ndarray,
    *,
    morf1: float = 0.04,
    morf2: float = 0.06,
    border_size: float = 0.06,
    region_bounds: tuple[int, int, int, int] = (0.2, 0.99, 0.2, 0.99),
    easy_background_color: tuple[int, int, int] = (255, 255, 255),
    debug_dir: str = None,
    segmentation_method: str = "degan",
    segmentation_operation_mode: str = "vote",
):
    """
    Finds broad regions in the image. Arguments are sorted from the most impactful to the least impactful.

    Args:
    TODO
    """

    if debug_dir:
        debug_dir = Path(debug_dir)

    # First, crop to content
    cropped, crop_box = crop_easy_borders(img, easy_background_color)
    if debug_dir:
        write_image(cropped, debug_dir / "cropped.png")

    # concatenating BGR,L*,GRAY

    ink_mask = find_ink(
        cropped,
        segmentation_method,
        segmentation_operation_mode,
    )

    # remove borders
    noborders = remove_borders(ink_mask, border_size=border_size)
    if debug_dir:
        ink_layer = cropped.copy()
        ink_layer[noborders == 0] = 0
        write_image(ink_layer, debug_dir / "noborders.png")

    # in the background, apply morphological operation horizontally
    # and vertically, then computes the logical and of the two
    # then compute the connected components and take the bounding boxes
    # of the components
    bounding_boxes = find_content(noborders, morf1, morf2, debug_dir=debug_dir)

    min_region_h, max_region_h, min_region_w, max_region_w = region_bounds
    regions = bb.filter_bboxes_on_size(
        bounding_boxes,
        min_region_h * noborders.shape[0],
        max_region_h * noborders.shape[0],
        min_region_w * noborders.shape[1],
        max_region_w * noborders.shape[1],
    )
    regions = bb.fix_generic_regions_intersections(regions)
    if debug_dir:
        broad_regions = bb.draw_boxes(cropped, regions)
        write_image(broad_regions, debug_dir / "broad_regions.png")

    # foreground can only be where there are broad regions, so and the two
    _region_mask = np.zeros(ink_mask.shape, dtype=bool)
    for region in regions:
        _region_mask[region.y0 : region.y1, region.x0 : region.x1] = True
    ink_mask = np.bitwise_and(ink_mask, _region_mask)

    return regions, ink_mask, crop_box


def get_content_box(regions_mask):
    x0, y0, x1, y1 = (
        np.min(np.argwhere(regions_mask)[:, 1]),
        np.min(np.argwhere(regions_mask)[:, 0]),
        np.max(np.argwhere(regions_mask)[:, 1]),
        np.max(np.argwhere(regions_mask)[:, 0]),
    )
    content_box = bb.BBox(x0, y0, x1, y1)
    return content_box


def preprocess_image(img: np.ndarray, debug_dir=None) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesses an image to extract the ink. Returns a boolean matrix. Input must in BGR. Returns the image content and the ink mask."""
    regions, ink_mask, crop_box = generic_analysis(img, debug_dir=debug_dir)
    # put the coordinates of regions in the original image
    bb.sum_coordinates(regions, crop_box.x0, crop_box.y0)
    regions_mask = np.zeros(img.shape[:2], dtype=bool)

    for region in regions:
        regions_mask[region.y0 : region.y1, region.x0 : region.x1] = True

    content_box = get_content_box(regions_mask)
    return (
        img[content_box.y0 : content_box.y1, content_box.x0 : content_box.x1],
        ink_mask[content_box.y0 : content_box.y1, content_box.x0 : content_box.x1],
    )
