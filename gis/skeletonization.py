"""
Skeletonization and Centerline Thinning for Cadastral Boundaries.
Extracts 1-pixel wide medial axis skeletons and detects graph endpoints and junctions.
"""

from typing import Tuple, List, Dict, Any, Set
import numpy as np
import cv2


def zhang_suen_thinning(binary_mask: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen morphological thinning algorithm for binary masks.
    Reduces thick boundary lines into 1-pixel wide connected skeletons.
    """
    skeleton = (binary_mask > 0).astype(np.uint8)
    h, w = skeleton.shape

    while True:
        # Step 1
        to_delete_step1 = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if skeleton[r, c] != 1:
                    continue
                # 8 neighbors in clockwise order: P2, P3, P4, P5, P6, P7, P8, P9
                p2 = skeleton[r - 1, c]
                p3 = skeleton[r - 1, c + 1]
                p4 = skeleton[r, c + 1]
                p5 = skeleton[r + 1, c + 1]
                p6 = skeleton[r + 1, c]
                p7 = skeleton[r + 1, c - 1]
                p8 = skeleton[r, c - 1]
                p9 = skeleton[r - 1, c - 1]

                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                b_p1 = sum(neighbors)

                # Condition 1: 2 <= B(P1) <= 6
                if not (2 <= b_p1 <= 6):
                    continue

                # Condition 2: A(P1) = number of 0 -> 1 transitions
                a_p1 = sum(
                    (neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1)
                    for i in range(8)
                )
                if a_p1 != 1:
                    continue

                # Condition 3 & 4: P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0
                if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                    continue

                to_delete_step1.append((r, c))

        if not to_delete_step1:
            break
        for r, c in to_delete_step1:
            skeleton[r, c] = 0

        # Step 2
        to_delete_step2 = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if skeleton[r, c] != 1:
                    continue
                p2 = skeleton[r - 1, c]
                p3 = skeleton[r - 1, c + 1]
                p4 = skeleton[r, c + 1]
                p5 = skeleton[r + 1, c + 1]
                p6 = skeleton[r + 1, c]
                p7 = skeleton[r + 1, c - 1]
                p8 = skeleton[r, c - 1]
                p9 = skeleton[r - 1, c - 1]

                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                b_p1 = sum(neighbors)

                if not (2 <= b_p1 <= 6):
                    continue

                a_p1 = sum(
                    (neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1)
                    for i in range(8)
                )
                if a_p1 != 1:
                    continue

                # Condition 3 & 4 for step 2: P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0
                if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
                    continue

                to_delete_step2.append((r, c))

        if not to_delete_step2:
            break
        for r, c in to_delete_step2:
            skeleton[r, c] = 0

    return skeleton


def fast_skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fast morphological skeletonization using morphological kernel operations.
    """
    img = (binary_mask > 0).astype(np.uint8) * 255
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    done = False
    max_iter = 100
    it = 0
    while not done and it < max_iter:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        done = (cv2.countNonZero(img) == 0)
        it += 1

    return (skel > 0).astype(np.uint8)


def detect_endpoints_and_junctions(skeleton: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Detect node categories in a 1-pixel skeleton:
    - Endpoints: pixels with exactly 1 neighbor (degree 1)
    - Junctions: pixels with 3 or more neighbors (degree >= 3)
    """
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    neighbor_count = cv2.filter2D(skeleton.astype(np.float32), -1, kernel)
    skel_mask = skeleton > 0

    endpoints_idx = np.where(skel_mask & (neighbor_count == 1))
    junctions_idx = np.where(skel_mask & (neighbor_count >= 3))

    endpoints = list(zip(endpoints_idx[0], endpoints_idx[1]))
    junctions = list(zip(junctions_idx[0], junctions_idx[1]))

    return endpoints, junctions
