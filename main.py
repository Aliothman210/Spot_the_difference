# main.py

# import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Load and convert images from BGR to RGB
# -------------------------------------------------------------
def load_images(path1, path2):
    """Load two images from disk and convert BGR to RGB."""

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # Error handling
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are invalid.")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return img1, img2


# -------------------------------------------------------------
# Compute binary difference map
# -------------------------------------------------------------
def compute_diff_mask(img1, img2, threshold_value=30):
    """Return binary difference mask (0/255) between two RGB images."""

    # Convert to grayscale to simplify difference calculation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Calculate absolute difference and threshold
    diff = cv2.absdiff(gray1, gray2)
    _, diff_mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    # Dilate to connect nearby differences
    kernel = np.ones((5,5), np.uint8)
    dilated_diff = cv2.dilate(diff_mask, kernel, iterations=1)
    return dilated_diff


# -------------------------------------------------------------
# Draw bounding boxes + label boxes on all valid contours
# -------------------------------------------------------------
def draw_contours_with_labels(img1, img2, diff_mask, min_area):
    """
    Draw all contours whose area >= min_area.
    Returns modified copies of images + number of contours drawn.
    """
    # Copies to avoid modifying originals
    out1 = img1.copy()
    out2 = img2.copy()
    # Find contours
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize difference counter
    diff_count = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        diff_count += 1

        # Bounding box (blue)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out1, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.rectangle(out2, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Label (white box + black number)
        label = str(diff_count)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        box_w = tw + 6
        box_h = th + 6

        # White rectangle
        cv2.rectangle(out1, (x, y - box_h - 4), (x + box_w, y - 4), (255, 255, 255), -1)
        cv2.rectangle(out2, (x, y - box_h - 4), (x + box_w, y - 4), (255, 255, 255), -1)

        # Black text
        cv2.putText(out1, label, (x + 3, y - 8), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(out2, label, (x + 3, y - 8), font, font_scale, (0, 0, 0), thickness)

    return out1, out2, diff_count


# -------------------------------------------------------------
# Show two images side-by-side
# -------------------------------------------------------------
def show_side_by_side(img1, img2, title1="Image 1", title2="Image 2"):
    """Display two images next to each other in one figure."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(title2)
    plt.axis("off")

    plt.show()


# -------------------------------------------------------------
# Main interactive controller
# -------------------------------------------------------------
def main():
    """Main interactive loop controlling the difference viewer."""
    img1_path = input("Enter first image path: ")
    img2_path = input("Enter second image path: ")

    img1, img2 = load_images(img1_path, img2_path)

    diff_mask = compute_diff_mask(img1, img2)

    plt.imshow(diff_mask, cmap="gray")
    plt.title("Difference Mask (Binary)")
    plt.axis("off")
    plt.show()

    while True:
        print("\nEnter minimum contour area (50 to 1000), or 'q' to quit:")
        user_input = input("Min Area: ")

        if user_input.lower() == 'q':
            print("Exiting...")
            break

        if not user_input.isdigit():
            print("Please enter a valid number.")
            continue

        min_area = int(user_input)
        if not (50 <= min_area <= 1000):
            print("Value must be between 50 and 1000.")
            continue

        # Draw all contours >= min_area
        out1, out2, count = draw_contours_with_labels(img1, img2, diff_mask, min_area)

        print(f"Contours drawn: {count}")
        show_side_by_side(out1, out2, f"Image 1 ({count} differences)", f"Image 2 ({count} differences)")


# -------------------------------------------------------------
# Run
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
