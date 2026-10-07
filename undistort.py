import cv2 as cv
import numpy as np
import os


def undistort_image(image_path):

    # Load calibration data
    cameraMatrix = np.load("cameraMatrix.npy")
    distCoeffs = np.load("distCoeffs.npy")

    # Load image
    image = cv.imread(image_path)

    if image is None:
        raise Exception(f"Could not open image: {image_path}")

    height, width = image.shape[:2]

    # Compute the optimal camera matrix
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
        cameraMatrix,
        distCoeffs,
        (width, height),
        1,
        (width, height)
    )

    # Remove distortion
    corrected = cv.undistort(
        image,
        cameraMatrix,
        distCoeffs,
        None,
        newCameraMatrix
    )

    # Crop away black borders
    x, y, w, h = roi
    corrected = corrected[y:y+h, x:x+w]

    # Save corrected image
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(image_path)
    output_name = os.path.join(output_folder, f"corrected_{filename}")

    if not cv.imwrite(output_name, corrected):
        raise IOError(f"Could not save corrected image: {output_name}")

    print(f"Saved {output_name}")
