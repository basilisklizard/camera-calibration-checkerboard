import cv2
import numpy as np
import os
import glob
import shutil


def calib(path, dims, checkerboard_grid_sz):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = dims
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = \
        np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * 0.001 * checkerboard_grid_sz
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob(os.path.join(path, "*.tif"))
    images.sort()

    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK +
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners,
                                        (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # cv2.imshow('img', img)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()

    h, w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if os.path.exists(path + '/results/'):
        shutil.rmtree(path + '/results/')
    os.makedirs(path + '/results/')

    np.savetxt(path + '/results/mtx.csv', mtx, delimiter=',')
    print("dist : \n")
    print(dist)
    np.savetxt(path + '/results/dist.csv', dist, delimiter=',')

    # undistort images
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h),
                                                          1, (w, h))
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(fname[:-4] + '_undist.png', dst)

    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)


if __name__ == "__main__":
    path = 'c1/'
    dims = (7, 7)
    checkerboard_grid_sz = 20 #[in mm]
    calib(path, dims, checkerboard_grid_sz)