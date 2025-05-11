import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def straighten_plate_from_image(original):
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    high_contrast = cv2.equalizeHist(gray)

    versions = {
        'original': original,
        'high': cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR),
    }

    results = []

    for key in ['original', 'high']:
        processed = versions[key]
        gray_for_contours = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray_for_contours, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:
                box = np.array(approx.reshape(4, 2), dtype="float32")
                warped = four_point_transform(original, box)
                results.append(warped)
                break
        else:
            results.append(original)

    return tuple(results)  # (original_warped, high_contrast_warped)
