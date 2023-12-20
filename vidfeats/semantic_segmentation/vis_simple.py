#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import cv2
import numpy as np

_LARGE_MASK_AREA_THRESH = 120000


def apply_colormask(image_cv, mask, color_rgb, alpha=0.5):
    """
    Overlays a colored mask onto an image with specified transparency.

    Parameters
    ----------
    image_cv : np.ndarray
        The original image onto which the mask will be applied.
    mask : np.ndarray
        A binary mask where 1 indicates the regions to be colored.
    color_rgb : tuple
        The color for the mask. Specified as (R, G, B).
    alpha : float
        The transparency level for the overlay. Default is 0.5.

    Returns
    -------
    np.ndarray
        The image with the overlay applied.
    """
    # Invert the color order to BGR format
    color_bgr = color_rgb[::-1]

    # Create a 3-channel colored mask based on the input mask
    colored_mask = np.dstack([mask] * 3)

    # Apply the color to the mask
    masked = np.ma.MaskedArray(image_cv, mask=colored_mask, fill_value=color_bgr)
    image_overlay = masked.filled()

    # Blend the original image with the overlay
    image_combined = cv2.addWeighted(image_cv, 1 - alpha, image_overlay, alpha, 0)

    return image_combined



def visualize_segments(img_rgb, sem_seg, categs_all, alpha=0.6):
    """
    Visualize semantic segmentation on a frame/image.

    Parameters
    ----------
    img_rgb : np.ndarray
        The input image in RGB format of shape (H, W, 3).
    sem_seg : np.ndarray
        The semantic segmentation mask of shape (H, W). Each pixel's value represents its label.
    categs_all : dict
        A dictionary mapping each label to its corresponding category name and color.
    alpha : float, optional
        The opacity of the colored masks, by default 0.6.

    Returns
    -------
    np.ndarray
        The BGR image with semantic segmentation visualization.
    """
    categs_all = np.array(categs_all)

    # Sort labels by their area (largest first)
    labels, areas = np.unique(sem_seg, return_counts=True)
    sorted_idxs = np.argsort(-areas)
    labels = labels[sorted_idxs]

    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    for label in labels:
        binary_mask = (sem_seg == label).astype(np.uint8)
        text = categs_all[label]['name']
        mask_color = np.array(categs_all[label]['color'])

        img = apply_colormask(img, binary_mask, color_rgb=mask_color, alpha=alpha)
        put_segmentation_label(img, binary_mask, text, mask_color)

    return img



def put_segmentation_label(img, binary_mask, text, color):
    """
    Puts text label on the largest connected component of a binary mask.

    Adapted from:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/visualizer.py

    Parameters
    ----------
    binary_mask : np.ndarray
        The binary mask where connected components are to be identified.
    text : str
        The text to be drawn on the mask.
    color : tuple
        Color of the text in RGB format (e.g., (255, 0, 0) for blue).

    """
    # Identify connected components in the binary mask
    num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)

    # If no components are found, return without drawing text
    if stats[1:, -1].size == 0:
        return

    # Identify the largest component
    largest_component_id = np.argmax(stats[1:, -1]) + 1

    # Draw text on the largest component and other large components
    for cid in range(1, num_cc):
        if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
            # Compute the median position as the text location
            center = np.median(np.nonzero(cc_labels == cid), axis=1)[::-1]
            draw_text(img, text, center.astype(int).tolist(), color=color.tolist()[::-1])



def draw_text(img, text, position, color, 
          font=cv2.FONT_HERSHEY_SIMPLEX,
          font_scale=0.4,
          font_thickness=1,
          text_color_bg=(0, 0, 0)
          ):
    """
    Draws text on an image at a specified position with black background

    Parameters
    ----------
    image : np.ndarray
        The image on which to draw the text.
    text : str
        The text to be drawn.
    position : tuple
        The position (x, y) to draw the text.
    color : tuple
        The color of the text in BGR format.

    Returns
    -------
    None
    """
    
    # Draw the text on the image at the specified position
    x, y = position
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    
    cv2.rectangle(img, (int(x - text_w/2), y), (int(x + text_w/2), y + text_h), text_color_bg, -1)
    
    cv2.putText(img, text, (int(x-text_w/2), int(y + text_h + font_scale - 1)), font, 
                font_scale, color, font_thickness) 
