import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks

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


def extract_vegetation_mask(img):
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


    img_resized, _ = downsize_img_to_screen(result)
    cv2.imshow("filtered:", img_resized)
    cv2.waitKey()

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_resized, _ = downsize_img_to_screen(mask)
    cv2.imshow("result:", mask_resized)
    cv2.waitKey()

    # reduce noise with open + close operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_resized, _ = downsize_img_to_screen(mask)
    cv2.imshow("after morph:", mask_resized)
    cv2.waitKey()

    return mask


def extract_features_mask(img):

    green_channel = img[:,:,1]
    green_blurred = cv2.GaussianBlur(green_channel,(3,3),0)
    resized, _ = downsize_img_to_screen(green_blurred)
    cv2.imshow("gaussian blurr:", resized)
    cv2.waitKey()

    _, otsu = cv2.threshold(green_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    resized, _ = downsize_img_to_screen(otsu)
    cv2.imshow("otsu:", resized)
    cv2.waitKey()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    # opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    detected = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    detected = cv2.morphologyEx(detected, cv2.MORPH_OPEN, kernel)
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
    cv2.imshow("clusters", vis_color)
    cv2.waitKey(0)

    detected_resized, _ = downsize_img_to_screen(detected)
    forest_and_isolated_trees = cv2.bitwise_or(detected_resized, forest_mask)
    cv2.imshow("forest and isolated trees", forest_and_isolated_trees)
    cv2.waitKey(0)
    
    return forest_and_isolated_trees
    
    # cv2.imwrite("output.png", downsize_img_to_screen(detected))


if __name__ == "__main__":
    
    print("reading image...")
    img = cv2.imread("orthophotos/lac.tif")
    
    img = select_roi(img)


    # extract_shadows(img)
    vegetation_mask = extract_vegetation_mask(img)
    vegetation_mask, _ = downsize_img_to_screen(vegetation_mask)

    forest_features = extract_features_mask(img)
    print(vegetation_mask.shape, forest_features.shape)
    final_detection = cv2.bitwise_and(forest_features, forest_features, mask=vegetation_mask)
    
    cv2.imshow("combination", final_detection)
    cv2.waitKey()