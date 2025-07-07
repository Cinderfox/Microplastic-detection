# Image Segmentation and Contour Detection for microplastics

This project performs advanced image segmentation and contour detection on a set of test images using OpenCV and related Python libraries. It is designed for computer vision lab assessments and includes automated evaluation against ground truth data.

## Features
- Adaptive thresholding and preprocessing for bright/dull images
- Contour detection and filtering by area
- Region growing segmentation
- Circular mask application and bounding box drawing
- Automatic evaluation with classification report and confusion matrix

## Project Structure
- `main.py` — Main script containing all image processing, segmentation, and evaluation logic
- `test_images/` — Folder containing test images for evaluation

## Requirements
- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- scikit-learn
- seaborn

Install dependencies with:
```bash
pip install opencv-python numpy matplotlib scikit-learn seaborn
```

## Usage
1. Place your test images in the `test_images/` directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. The script will process all images, display results, and print evaluation metrics.

## Output
- For each image, the script predicts the number of detected objects and compares it to the ground truth.
- Prints total passed/failed cases, classification report, accuracy, and displays a confusion matrix.

## Customization
- Adjust thresholds and parameters in `main.py` for different datasets or requirements.
- Add or modify ground truth labels in the `actual` dictionary inside `main.py`.
