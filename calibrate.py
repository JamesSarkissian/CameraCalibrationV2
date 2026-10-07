import cv2 as cv
import numpy as np
import glob
import os


def calibrate_camera(image_folder):

    # Checkerboard dimensions (inner corners)
    CHECKERBOARD = (7, 7)

    # Corner refinement criteria
    criteria = (
        cv.TERM_CRITERIA_EPS +
        cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    # 3D object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    image_paths = glob.glob(os.path.join(image_folder, "*"))

    if len(image_paths) == 0:
        raise Exception("No images found.")

    image_size = None
    successful = 0

    for filename in image_paths:

        img = cv.imread(filename)

        if img is None:
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Create a smaller copy of the entire image for detection.
        height, width = gray.shape
        detection_gray = cv.resize(
            gray,
            (max(1, width // 4), max(1, height // 4)),
            interpolation=cv.INTER_AREA
        )

        found, corners = cv.findChessboardCorners(
            detection_gray,
            CHECKERBOARD,
            None
        )

        if found:

            # Map corner coordinates back to the original image.
            corners = corners.reshape(-1, 1, 2)
            scale_x = width / detection_gray.shape[1]
            scale_y = height / detection_gray.shape[0]
            corners[:, :, 0] = (corners[:, :, 0] + 0.5) * scale_x - 0.5
            corners[:, :, 1] = (corners[:, :, 1] + 0.5) * scale_y - 0.5

            # Refine the corner positions at full resolution.
            corners = cv.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            objpoints.append(objp)
            imgpoints.append(corners)

            image_size = gray.shape[::-1]

            successful += 1

            cv.drawChessboardCorners(
                img,
                CHECKERBOARD,
                corners,
                found
            )

            cv.imshow("Calibration", img)
            cv.waitKey(300)

        else:
            print(f"Checkerboard not found: {filename}")

    cv.destroyAllWindows()

    if successful < 5:
        raise Exception("Not enough successful images.")

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    np.save("cameraMatrix.npy", cameraMatrix)
    np.save("distCoeffs.npy", distCoeffs)

    print("\nCalibration Successful!")
    print(f"Images Used: {successful}/{len(image_paths)}")

    print("\nCamera Matrix:")
    print(cameraMatrix)

    print("\nDistortion Coefficients:")
    print(distCoeffs)

    # Mean reprojection error
    total_error = 0

    for i in range(len(objpoints)):

        projected, _ = cv.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            cameraMatrix,
            distCoeffs
        )

        error = cv.norm(
            imgpoints[i],
            projected,
            cv.NORM_L2
        ) / len(projected)

        total_error += error

    print(f"\nMean Reprojection Error: {total_error / len(objpoints):.4f} pixels")
