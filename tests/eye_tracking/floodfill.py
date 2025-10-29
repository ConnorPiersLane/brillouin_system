import cv2
import numpy as np
from matplotlib import pyplot as plt

def floodfill_and_show(path):
    # Read grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Threshold (Otsu for auto binarization)
    _, bw = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV )

    # Flood fill from borders to remove white background
    h, w = bw.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    ff = bw.copy()
    cv2.floodFill(ff, mask, (0,0), 128)
    # cv2.floodFill(ff, mask, (w-1,0), 128)
    # cv2.floodFill(ff, mask, (0,h-1), 128)
    # cv2.floodFill(ff, mask, (w-1,h-1), 128)

    # Replace background (128) with black
    ff[ff == 128] = 0

    # Show results
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original binary")
    plt.imshow(bw, cmap='gray')

    plt.subplot(1,2,2)
    plt.title("Floodfilled background removed")
    plt.imshow(ff, cmap='gray')

    plt.show()

# Example use:
floodfill_and_show("left/pair_0001_left.png")
