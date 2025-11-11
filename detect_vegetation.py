import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import pyautogui
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks


import rasterio
import numpy as np

import rasterio
import numpy as np

import rasterio
import numpy as np

def save_comparison_as_geotiff(input_tif_path, img, output_tif_path):
    # Make sure the image is uint8
    img = img.astype(np.uint8)

    # Convert BGR → RGB
    img_rgb = img[:, :, ::-1].copy()

    # Reorder to (bands, height, width)
    img_rgb = np.transpose(img_rgb, (2, 0, 1))

    # Load ONLY CRS and transform
    with rasterio.open(input_tif_path) as src:
        crs = src.crs
        transform = src.transform

    # ✅ MINIMAL PROFILE — nothing extra
    profile = {
        "driver": "GTiff",
        "height": img_rgb.shape[1],
        "width": img_rgb.shape[2],
        "count": img_rgb.shape[0],
        "dtype": img_rgb.dtype,
        "crs": crs,
        "transform": transform,
    }

    print("\nWriting using minimal profile:")
    for k,v in profile.items():
        print(k,":",v)

    with rasterio.open(output_tif_path, "w", **profile) as dst:
        dst.write(img_rgb)


def downsize_img_to_screen(img):
    """
    rescale potentially large images based on current screen size
    """
    
    w,h = pyautogui.size()
    w_ratio = img.shape[0]/w
    h_ratio = img.shape[1]/h

    if max(w_ratio, h_ratio) >= 1:
        fx = 0.9/(max(w_ratio, h_ratio)) # rescale to 0.9 * screen size
        img_resized = cv2.resize(img, None, fx=fx, fy=fx, interpolation=cv2.INTER_LINEAR)
        return img_resized, fx

    return img, 1

def upscale_image_to_target_size(img, scaling_factor):
    """
    upscale image to new w, h

    used to upscale masks back to original image size
    """
    return cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
     

def select_roi(img):
    
    # downsize if image too big for screen
    w, h = img.shape[:2] 
    resized, scaling_factor = downsize_img_to_screen(img)

    x,y,w,h = cv2.selectROI("select the area", resized)

    if w == 0 and h == 0:
        return img
    
    # roi = resized[int(y):int(y+h), int(x):int(x+w)]
    # cv2.imshow("ROI selection", roi)
    # cv2.waitKey(0)
    # Convert to original image coordinates
    x_orig = int(x * 1/scaling_factor)
    y_orig = int(y * 1/scaling_factor)
    w_orig = int(w * 1/scaling_factor)
    h_orig = int(h * 1/scaling_factor)
    
    roi_original = img[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]
    cv2.imshow("original size roi", roi_original)
    cv2.waitKey(0)
    return roi_original

def extract_shadows(img):
    """
    Extract dark spots corresponding to shadows. smooth the shadows to filter noise
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    hist, bins = np.histogram(val, bins=180, range=(0,100))
    dominant_val_index = np.argmax(hist)
    print("dominant value:", dominant_val_index)

    plt.figure(figsize=(8,4))
    plt.bar(bins[:-1], hist, width = 1.0)
    plt.title("value histogram of image")
    plt.xlabel("value (0 - 100)")
    plt.show()

    margin = 0
    val_mask = val < 20
    shadows = np.zeros_like(img)
    shadows[val_mask] = 255

    cv2.imshow("shadows:", downsize_img_to_screen(shadows))
    cv2.waitKey()


def extract_vegetation_mask(img, plot: bool = False):
    """
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    hist, bins = np.histogram(hue, bins=180, range=(0,180))
    
    hist_float = hist.astype(np.float32)
    hist_2d = hist_float.reshape(1, -1)   # shape = 1 x 180, horizontal
    hist_smooth = cv2.GaussianBlur(hist_2d, (9,1), sigmaX=2)  # kernel(9,1) smooths along x-axis
    hist = hist_smooth.reshape(-1)

    height_thresh = np.max(hist)*0.2

    peaks, properties = find_peaks(hist, height_thresh, distance=5)
    peak_heights = properties['peak_heights']
    print("Filtered peaks (bin, height):")
    for p, h in zip(peaks, peak_heights):
        print(p, h)

    target_hue = 60     # vegetation hue

    # Compute distance from each peak to target
    distances = np.abs(peaks - target_hue)
    closest_idx = np.argmin(distances)
    vegetation_peak = peaks[closest_idx]

    print(f"Selected vegetation peak: bin={vegetation_peak}")
    if plot:
        plt.figure(figsize=(8,4))
        plt.bar(bins[:-1], hist, width = 1.0)
        plt.title("hue histogram of image")
        plt.xlabel("hue (0 - 180)")
        plt.show()

    margin = 10
    hue_mask = (hue >= (vegetation_peak - margin)) & (hue <= (vegetation_peak + margin))
    # hue_mask = (hue >= (60 - margin)) & (hue <= (60 + margin))
    val_mask = val < 45
    mask = hue_mask | val_mask
    result = np.zeros_like(img)
    result[mask] = img[mask]

    if plot: 
        img_resized, _ = downsize_img_to_screen(result)
        cv2.imshow("filtered:", img_resized)
        cv2.waitKey()

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    if plot:
        mask_resized, _ = downsize_img_to_screen(mask)
        cv2.imshow("result:", mask_resized)
        cv2.waitKey()

    # reduce noise with open + close operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if plot:
        mask_resized, _ = downsize_img_to_screen(mask)
        cv2.imshow("after morph:", mask_resized)
        cv2.waitKey()

    return mask


def extract_features_mask(img, plot:bool = False):

    green_channel = img[:,:,1]
    green_blurred = cv2.GaussianBlur(green_channel,(3,3),0)
    if plot:
        resized, _ = downsize_img_to_screen(green_blurred)
        cv2.imshow("gaussian blurr:", resized)
        cv2.waitKey()

    _, otsu = cv2.threshold(green_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if plot:
        resized, _ = downsize_img_to_screen(otsu)
        cv2.imshow("otsu:", resized)
        cv2.waitKey()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    # opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    detected = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    detected = cv2.morphologyEx(detected, cv2.MORPH_OPEN, kernel)
    if plot:
        resized, _ = downsize_img_to_screen(detected)
        cv2.imshow("after morph:", resized)
        cv2.waitKey()

    # mask is 0/255 binary
    resized, _ = downsize_img_to_screen(detected)
    ys, xs = np.where(resized == 255)
    coords = np.vstack((xs, ys)).T

    # choose eps based on desired connectivity (pixel distance)
    print("running clustering to find forest patches...")
    db = DBSCAN(eps=21, min_samples=100).fit(coords)
    labels = db.labels_

    # generate cluster mask
    cluster_mask = np.zeros_like(resized, dtype=np.int32)
    cluster_mask[ys, xs] = labels + 1   # +1 because -1 is noise
    labels = cluster_mask   # shape (H, W)
    unique_labels = np.unique(labels)

    # create 3-channel color image
    vis_color = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    forest_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.uint8)

    # assign random color per label
    for lab in unique_labels:
        if lab == 0:
            continue  # 0 means background/no-cluster

        # --- extract cluster mask ---
        cluster = (labels == lab).astype(np.uint8) * 255

        # --- apply morphological closing on THIS cluster only ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        cluster_closed = cv2.morphologyEx(cluster, cv2.MORPH_CLOSE, kernel)

        # --- give random color ---
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)

        # --- write closed cluster into color visualization ---
        vis_color[cluster_closed > 0] = color
        forest_mask[cluster_closed > 0] = 255

    # display
    if plot:
        cv2.imshow("clusters", vis_color)
        cv2.waitKey(0)

    detected_resized, _ = downsize_img_to_screen(detected)
    forest_and_isolated_trees = cv2.bitwise_or(detected_resized, forest_mask)
    if plot:
        cv2.imshow("forest and isolated trees", forest_and_isolated_trees)
        cv2.waitKey(0)
    
    return forest_and_isolated_trees
    
    # cv2.imwrite("output.png", downsize_img_to_screen(detected))

def full_detection_pipeline(img):

    img = select_roi(img)
    # extract_shadows(img)
    vegetation_mask = extract_vegetation_mask(img)
    vegetation_mask, _ = downsize_img_to_screen(vegetation_mask)

    forest_features = extract_features_mask(img)
    final_detection = cv2.bitwise_and(forest_features, forest_features, mask=vegetation_mask)
    
    cv2.imshow("combination", final_detection)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return final_detection

def compare_vegetation_evolution(mask1, mask2):
    # Create empty RGB output
    h, w = mask1.shape
    comparison = np.zeros((h, w, 3), dtype=np.uint8)

    mask1 = (mask1 > 127).astype(np.uint8)
    mask2 = (mask2 > 127).astype(np.uint8)

    both_white = (mask1 == 1) & (mask2 == 1)
    only_mask1 = (mask1 == 1) & (mask2 == 0)
    only_mask2 = (mask1 == 0) & (mask2 == 1)

    # Assign colors
    comparison[both_white] = (255, 255, 255)   # white
    comparison[only_mask1] = (0, 0, 255)       # red (BGR format!)
    comparison[only_mask2] = (255, 0, 0)       # blue (BGR format!)

    # Show the result
    cv2.imshow("Mask Comparison", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return comparison
    

if __name__ == "__main__":
    

    print("reading image...")
    img = cv2.imread("orthophotos/hongrin1.tif")
    print(img.shape)
    mask1 = full_detection_pipeline(img)

    img = cv2.imread("orthophotos/hongrin2.tif")
    mask2 = full_detection_pipeline(img)

    comparison_result = compare_vegetation_evolution(mask1, mask2)
    w,h = img.shape[:2]
    comparison_result = cv2.resize(comparison_result, (w,h))
    save_comparison_as_geotiff("orthophotos/hongrin1.tif", comparison_result, "output.tif")