"""HSV glove detection for mirrored OpenCV frames.
Assumes blue glove is on anatomical left hand and red on right.
"""
import cv2
import numpy as np

# Initial OpenCV HSV thresholds; adjust for actual lighting/gloves.
BLUE_RANGES = [((95, 80, 55), (135, 255, 255))]
RED_RANGES = [((0, 85, 55), (12, 255, 255)),
              ((170, 85, 55), (179, 255, 255))]


def _mask_for_ranges(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, np.array(low, dtype=np.uint8),
                           np.array(high, dtype=np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _find_glove(mask, wrist, roi_radius=100):
    """Return (x,y), box or None, and detected area within wrist ROI."""
    if wrist is None:
        return None, None, 0
    height, width = mask.shape
    x, y = map(int, wrist)
    x1, y1 = max(0, x - roi_radius), max(0, y - roi_radius)
    x2, y2 = min(width, x + roi_radius), min(height, y + roi_radius)
    if x1 >= x2 or y1 >= y2:
        return None, None, 0
    cropped = mask[y1:y2, x1:x2]
    contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    min_area = max(100, (roi_radius ** 2) * 0.012)
    if area < min_area:
        return None, None, 0
    m = cv2.moments(contour)
    if m['m00'] <= 0:
        return None, None, 0
    center = (int(m['m10'] / m['m00']) + x1,
              int(m['m01'] / m['m00']) + y1)
    bx, by, bw, bh = cv2.boundingRect(contour)
    return center, (bx + x1, by + y1, bw, bh), area


def detect_gloves(mirrored_frame, left_wrist=None, right_wrist=None,
                  shoulder_width=None):
    """Use pose wrists to restrict color detection and avoid background false positives."""
    hsv = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2HSV)
    blue = _mask_for_ranges(hsv, BLUE_RANGES)
    red = _mask_for_ranges(hsv, RED_RANGES)
    # ROI scales with image/person size; keep practical lower/upper bounds.
    radius = int(np.clip((shoulder_width or 180) * 0.65, 60, 180))
    lc, lb, la = _find_glove(blue, left_wrist, radius)
    rc, rb, ra = _find_glove(red, right_wrist, radius)
    return {'left': lc, 'right': rc, 'left_box': lb, 'right_box': rb,
            'left_area': la, 'right_area': ra}


def fuse_wrist(wrist, glove_center, shoulder_width):
    """Reject implausible glove centers; blend for smoother positional estimate."""
    if wrist is None:
        return glove_center
    if glove_center is None:
        return tuple(wrist)
    distance = np.hypot(wrist[0] - glove_center[0], wrist[1] - glove_center[1])
    if distance > max(45, 0.75 * shoulder_width):
        return tuple(wrist)
    return (int(0.45 * wrist[0] + 0.55 * glove_center[0]),
            int(0.45 * wrist[1] + 0.55 * glove_center[1]))
