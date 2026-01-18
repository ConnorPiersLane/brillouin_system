import cv2
import numpy as np

def mask_outer_lens(image_path, radius=500, show=True, save_path=None):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    # Get image center
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Create a mask (white circle on black background)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply the mask (keep only inside the circle)
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Draw the circle overlay on a copy for visualization
    overlay = image.copy()
    cv2.circle(overlay, center, radius, (0, 255, 0), 2)

    if show:
        cv2.imshow("Original with Circle", overlay)
        cv2.imshow("Masked Region", masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optionally save the result
    if save_path:
        cv2.imwrite(save_path, masked)
        print(f"✅ Saved masked image to: {save_path}")

if __name__ == "__main__":
    # Example usage
    mask_outer_lens("right/pair_0000_right.png", radius=500)
