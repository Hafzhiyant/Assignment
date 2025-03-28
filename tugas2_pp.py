import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_image(image_path):
    """Process an image to detect and fill holes in blood cell images."""
    # Read image
    image = cv2.imread(image_path)
    green_channel = image[:, :, 1]
    filtered = cv2.medianBlur(green_channel, 7)

    # Thresholding
    _, thresh = cv2.threshold(
        filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Invert image to ensure proper hole filling
    thresh_inv = cv2.bitwise_not(thresh)

    # Use Flood Fill to fill holes
    height, width = thresh.shape
    print(f"Image dimensions: {height} x {width}")
    mask = np.zeros((height + 2, width + 2), np.uint8)
    flood_filled = thresh.copy()
    cv2.floodFill(flood_filled, mask, (0, 150), 150)

    # Invert back to get image with filled holes
    filled_holes = cv2.bitwise_not(flood_filled)

    # Combine hole-filled result with initial segmentation
    final_segmented = cv2.bitwise_or(thresh, filled_holes)

    # Morphological Closing to improve object boundaries
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(
        final_segmented, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    return {
        "filtered": filtered,
        "thresh": thresh,
        "thresh_inv": thresh_inv,
        "mask": mask,
        "flood_filled": flood_filled,
        "filled_holes": filled_holes,
        "final": closed,
    }


def display_results(results):
    """Display processing results in a grid of subplots."""
    plt.figure(figsize=(12, 6))

    titles = [
        "Filtered",
        "Thresholding",
        "Threshold Inverse",
        "Mask",
        "Flood Fill",
        "Filled Holes",
        "Final Result",
    ]
    keys = [
        "filtered",
        "thresh",
        "thresh_inv",
        "mask",
        "flood_filled",
        "filled_holes",
        "final",
    ]

    for i, (title, key) in enumerate(zip(titles, keys), 1):
        plt.subplot(1, 7, i)
        plt.title(title)
        plt.imshow(results[key], cmap="gray" if i > 1 else None)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """Main function to process and display image."""
    image_path = r"C:\Users\ediso\Downloads\darah1.png"
    processing_results = process_image(image_path)
    display_results(processing_results)


if __name__ == "__main__":
    main()