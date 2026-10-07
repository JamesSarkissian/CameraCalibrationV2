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

    # Display comparison
    cv.imshow("Original", image)
    cv.imshow("Undistorted", corrected)

    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save corrected image
    filename = os.path.basename(image_path)
    output_name = f"corrected_{filename}"

    cv.imwrite(output_name, corrected)

    print(f"Saved {output_name}")