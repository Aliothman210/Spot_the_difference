# Spot the Difference – CV Task

This project implements a simple computer vision pipeline to detect differences between two images.
The goal is to highlight all regions that changed between Image 1 and Image 2 using OpenCV.

## 🔍 What the project does

- Loads two images.
- Converts them to grayscale.
- Computes the absolute difference between them.
- Applies a binary threshold to extract changed regions.
- Detects contours of the differences.
- Lets the user interactively filter differences by contour area (for example: show only differences larger than 200 pixels).
- Displays both the original image and the highlighted differences side-by-side.

## 🧠 Main idea

The project relies on a very common CV workflow:
- absdiff() → measures pixel-wise changes
- threshold() → converts changes into a binary mask
- findContours() → extracts the changed regions
- Drawing bounding boxes on the differences

This makes it easy to visually understand what has changed between two images.

## ▶️ How to run

Place two input images in the folder and update their paths in main() inside the script.
Then run:

```bash
python difference_viewer.py
```

You will see a binary map of differences and will be prompted to enter an area threshold (e.g., 100, 200, 500).
The program will then show only contours whose area meets your selected threshold.

## 📦 Requirements

- opencv-python
- numpy
- matplotlib

Install them:

```bash
pip install opencv-python numpy matplotlib
```

## 📁 File Structure

```
project/
│
├── difference_viewer.py
└── README.md
```
