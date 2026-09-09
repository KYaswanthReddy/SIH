"""
Morphological Cleanup and Boundary Mask Refinement Utilities.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


def refine_boundary_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    min_component_size: int = 25,
    morph_kernel_size: int = 3,
    close_gaps: bool = True
) -> np.ndarray:
    """
    Threshold and morphologically clean predicted cadastral boundary probability map.
    
    Args:
        prob_map: 2D numpy array of probabilities in [0.0, 1.0]
        threshold: Binarization threshold
        min_component_size: Minimum connected component area in pixels to retain
        morph_kernel_size: Size of morphological structuring element
        close_gaps: If True, performs morphological closing to connect micro-gaps
        
    Returns:
        binary_mask: Cleaned 2D binary numpy array in {0, 1}
    """
    # 1. Binarize
    binary = (prob_map >= threshold).astype(np.uint8)

    if binary.sum() == 0:
        return binary

    # 2. Morphological Closing to bridge micro-gaps along boundary lines
    if close_gaps and morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. Filter small isolated noise components
    if min_component_size > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_component_size:
                cleaned[labels == label_idx] = 1
        binary = cleaned

    return binary
