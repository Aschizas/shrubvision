import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from rembg import remove
from PIL import Image


def extract_toad_from_image(img_path):
    input_image = Image.open(img_path)
    output = remove(input_image)  # RGBA PIL image

    output_array = np.array(output)
    mask = output_array[:, :, 3]
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    original_img = cv2.imread(img_path)
    toad = cv2.bitwise_and(original_img, original_img, mask=mask)

    return toad

def color_histogram(img, plot: bool = False):
    """
    Compute hue histogram, find the target color peak, and create a mask.
    Now includes saturation and value filtering to keep only bright, saturated yellow.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    hist, bins = np.histogram(hue, bins=180, range=(0,180))
    
    hist_float = hist.astype(np.float32)
    hist_2d = hist_float.reshape(1, -1)
    hist_smooth = cv2.GaussianBlur(hist_2d, (9,1), sigmaX=2)
    hist = hist_smooth.reshape(-1)

    height_thresh = np.max(hist)*0.2
    peaks, properties = find_peaks(hist, height_thresh, distance=5)
    peak_heights = properties['peak_heights']
    print("Filtered peaks (bin, height):")
    for p, h in zip(peaks, peak_heights):
        print(p, h)

    target_hue = 25
    distances = np.abs(peaks - target_hue)
    closest_idx = np.argmin(distances)
    vegetation_peak = peaks[closest_idx]

    print(f"Selected color peak: bin={vegetation_peak}")
    if plot:
        plt.figure(figsize=(8,4))
        plt.bar(bins[:-1], hist, width = 1.0)
        plt.title("hue histogram of image")
        plt.xlabel("hue (0 - 180)")
        plt.show()

    margin = 2
    hue_mask = (hue >= (vegetation_peak - margin)) & (hue <= (vegetation_peak + margin))

    sat_mask = sat >= 80       # keep only highly saturated pixels
    val_mask = val >= 60       # keep only bright pixels

    combined_mask = hue_mask & sat_mask & val_mask
    mask = np.zeros(hue.shape, dtype=np.uint8)
    mask[combined_mask] = 255

    result = cv2.bitwise_and(img, img, mask=mask)

    if plot: 
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(mask, cmap="gray")
        axes[0].set_title("Binary Mask")
        axes[0].axis("off")

        axes[1].imshow(result_rgb)
        axes[1].set_title("Masked Result")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    return mask


def extract_interesting_patterns(raw_mask, min_area=1000, max_area=45000,
                                 squiggly_thresh=0.6, concavity_thresh=0.15, plot=False):

    # find contours (ignore hierarchy)
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    kept_contours = []
    for cnt in contours:
        # area filtering only
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            kept_contours.append(cnt)

    # create empty mask
    filtered_mask = np.zeros_like(raw_mask)

    # process each contour individually
    for cnt in kept_contours:
        single_mask = np.zeros_like(raw_mask)
        cv2.drawContours(single_mask, [cnt], -1, color=255, thickness=cv2.FILLED)

        # strong opening/closing on this contour
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        single_mask = cv2.morphologyEx(single_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        single_mask = cv2.morphologyEx(single_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # re-extract contour after morphology
        cnts_after = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(cnts_after) == 0:
            continue
        cnt_final = cnts_after[0]

        # --- circularity filter ---
        perimeter = cv2.arcLength(cnt_final, True)
        area_final = cv2.contourArea(cnt_final)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area_final / (perimeter ** 2)
        if circularity > squiggly_thresh:
            continue

        # --- concavity filter ---
        hull = cv2.convexHull(cnt_final)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        concavity = 1 - (area_final / hull_area)
        if concavity < concavity_thresh:
            continue  # skip contours that are too convex (not squiggly)

        # merge back
        cv2.drawContours(filtered_mask, [cnt_final], -1, color=255, thickness=cv2.FILLED)

    if plot:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(raw_mask, cmap="gray")
        plt.title("Original Mask")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(filtered_mask, cmap="gray")
        plt.title("Filtered Squiggly & Concave Contours")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return filtered_mask

def match_similar_contours(mask1, mask2, min_area=1500, max_area=25000,
                           size_tolerance=0.4, match_thresh=0.5, plot=False):
    """
    Match contours between two masks using OpenCV shape descriptor (Hu moments).
    Allows multiple matches per contour and filters by similar size.
    
    Parameters:
        size_tolerance: fraction, max relative area difference allowed between two contours
                        e.g., 0.3 = only compare contours within ±30% area of each other
    """
    # Extract contours from both masks
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area filtering only
    def filter_contours(contours):
        kept = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                kept.append(cnt)
        return kept

    filtered1 = filter_contours(contours1)
    filtered2 = filter_contours(contours2)

    # Compare contours using Hu moments
    matched1 = []
    matched2 = []

    for cnt1 in filtered1:
        area1 = cv2.contourArea(cnt1)
        for cnt2 in filtered2:
            area2 = cv2.contourArea(cnt2)
            # Skip if areas differ too much
            if abs(area1 - area2) / max(area1, area2) > size_tolerance:
                continue

            score = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0)
            if score < match_thresh:
                matched1.append(cnt1)
                matched2.append(cnt2)
                # allow multiple matches per contour

    # Build masks of matched contours
    matched_mask1 = np.zeros_like(mask1)
    matched_mask2 = np.zeros_like(mask2)
    if matched1:
        cv2.drawContours(matched_mask1, matched1, -1, 255, cv2.FILLED)
    if matched2:
        cv2.drawContours(matched_mask2, matched2, -1, 255, cv2.FILLED)

    if plot:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(matched_mask1, cmap="gray")
        plt.title("Matched Mask 1")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(matched_mask2, cmap="gray")
        plt.title("Matched Mask 2")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return matched_mask1, matched_mask2





def extract_patterns_from_toad(img_path):
    sonneur_img = extract_toad_from_image(img_path)
    toad_mask = color_histogram(sonneur_img, plot=False)
    pattern_mask = extract_interesting_patterns(toad_mask, plot=True)

    return pattern_mask

if __name__ == "__main__":

    sonneur1 = "sonneurs/BOVA004_003.jpg"
    sonneur2 = "sonneurs/BOVA004_002.jpg"

    mask1 = extract_patterns_from_toad(sonneur1)
    mask2 = extract_patterns_from_toad(sonneur2)

    match_similar_contours(mask1, mask2, plot=True)