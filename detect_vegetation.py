import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
from sklearn.cluster import DBSCAN

def resize_img_to_screen(img):
    """
    rescale potentially large images based on current screen size
    """
    
    w,h = pyautogui.size()
    w_ratio = img.shape[0]/w
    h_ratio = img.shape[1]/h

    if max(w_ratio, h_ratio) >= 1:
        fx = 0.9/(max(w_ratio, h_ratio)) # rescale to 0.9 * screen size
        img_resized = cv2.resize(img, None, fx=fx, fy=fx, interpolation=cv2.INTER_LINEAR)
        return img_resized

    return img

def extract_vegetation_mask(img):
    """
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hue = hsv[:,:,0]
    hist, bins = np.histogram(img_hue, bins=180, range=(0,180))
    dominant_hue_index = np.argmax(hist)
    print("dominant hue:", dominant_hue_index)

    plt.figure(figsize=(8,4))
    plt.bar(bins[:-1], hist, width = 1.0)
    plt.title("hue histogram of image")
    plt.xlabel("hue (0 - 180)")
    plt.show()

    margin = 25
    hue_mask = (img_hue >= (dominant_hue_index - margin)) & (img_hue <= (dominant_hue_index + margin))
    result = np.zeros_like(img)
    result[hue_mask] = img[hue_mask]


    img_resized = resize_img_to_screen(result)
    cv2.imshow("image:", img_resized)
    cv2.waitKey()

    _, mask = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)
    cv2.imshow("image:", resize_img_to_screen(mask))
    cv2.waitKey()



def extract_features_mask(img):

    green_channel = img[:,:,1]
    green_blurred = cv2.GaussianBlur(green_channel,(3,3),0)
    cv2.imshow("gaussian blurr:", resize_img_to_screen(green_blurred))
    cv2.waitKey()

    ret3,otsu = cv2.threshold(green_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow("otsu:", resize_img_to_screen(otsu))
    cv2.waitKey()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    # opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    detected = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("otsu:", resize_img_to_screen(detected))
    cv2.waitKey()

    # mask is 0/255 binary
    ys, xs = np.where(resize_img_to_screen(detected) == 255)
    coords = np.vstack((xs, ys)).T

    # choose eps based on desired connectivity (pixel distance)
    db = DBSCAN(eps=19, min_samples=100).fit(coords)
    labels = db.labels_

    # generate cluster mask
    cluster_mask = np.zeros_like(resize_img_to_screen(detected), dtype=np.int32)
    cluster_mask[ys, xs] = labels + 1   # +1 because -1 is noise
    labels = cluster_mask   # shape (H, W)
    unique_labels = np.unique(labels)

    # create 3-channel color image
    vis_color = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

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

    # display
    cv2.imshow("clusters", vis_color)
    cv2.waitKey(0)
    
    # cv2.imwrite("output.png", resize_img_to_screen(detected))


if __name__ == "__main__":
    
    print("reading image...")
    img = cv2.imread("orthophotos/hongrin1.tif")
    print("shape:", img.shape)
    print("dtype:", img.dtype)

    # extract_vegetation_mask(img)
    extract_features_mask(img)