import numpy as np
import cv2
import glob
import open3d as o3d

supervise_mode = True

# Your information here
name = 'Moon Hee Joon'
student_id = '2018102106'


if supervise_mode:
    print('name:%s id:%s'%(name, student_id))

# ====================================================================================
# Camera calibration
# ====================================================================================
# Set directory path (images capturing check pattern)
# Example) calibration_dir_path = 'calibration/*.png'
calibration_dir_path = 'cal/cal*.png'
calibration_images = glob.glob(calibration_dir_path)

# intrinsic parameters and distortion coefficient
# With these parameter, you can get undistorted image and new intrinsic parameter of them (K_undist)
K = np.array([], dtype=np.float32) # camera intrinsic parameter
dist = np.array([], dtype=np.float32) # distortion coefficient
# new matrix for undistorted intrinsic parameter
K_undist = np.array([], dtype=np.float32)

# Your code here
# Goals
# 1. Get camera intrinsic parameters from your captured images K, dist, K_undist
# 2. Try to get undistorted images by warping captured images using K_undist
# reference: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

# ****************************** Your code here (M-1) ******************************
# grid count: 9 x 6
# grid size: 24mm x 24mm

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in calibration_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points(after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (1, 1), (-1, -1), criteria)
        imgpoints.append(corners)

# Calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# New undistorted intrinsic parameter
K_undist, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w, h),1, (w, h))

# **********************************************************************************

if supervise_mode:
    print('1-1. Calibration: K matrix')
    print(K)
    print('1-2. Calibration: distortion coefficients')
    print(dist)
    print('1-3. Calibration: Undistorted K matrix')
    print(K_undist)
    for fname in calibration_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        img_undist = cv2.undistort(gray, K, dist, None, K_undist)
        cv2.imshow('undistorted image', img_undist)
        cv2.waitKey(0)

# ====================================================================================
# load stereo images (Left and Right)
# ====================================================================================
#  set your left and right images
imgL = cv2.imread('left1.png')
imgR = cv2.imread('right1.png')

# convert to grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Convert to undistorted images
# imgLU: undistorted image of imgL
# imgRU: undistorted image of imgR
# grayLU: undistorted image of grayL
# grayRU: undistorted image of grayR
imgLU = np.array([])
imgRU = np.array([])
grayLU = np.array([])
grayRU = np.array([])

# undistorted images
# ****************************** Your code here (M-2) ******************************
imgLU = cv2.undistort(imgL, K, dist, None, K_undist)
imgRU = cv2.undistort(imgR, K, dist, None, K_undist)
grayLU = cv2.undistort(grayL, K, dist, None, K_undist)
grayRU = cv2.undistort(grayR, K, dist, None, K_undist)
# **********************************************************************************

if supervise_mode:
    cv2.imshow('rgb undistorted', cv2.hconcat([imgLU, imgRU]))
    cv2.imshow('gray undistorted', cv2.hconcat([grayLU, grayRU]))


# ====================================================================================
# stereo matching (Dense matching)
# ====================================================================================
# Goals
#  1. Get disparity map (8 bit unsigned)
#  Note. The output of disparity function (StereoBM, etc.) is 16-bit
#
# reference: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_depthmap.html
#
disp8 = np.array([], np.uint8)
# ****************************** Your code here (M-3) ******************************
# stereo block matching
stereo = cv2.StereoBM_create(numDisparities=32, blockSize=25)
disparity = stereo.compute(grayLU, grayRU)

# Get Maximum disparity & Increase range from 1 ~ 255
max_disp = np.max(disparity)
disp8 = np.uint8(disparity / max_disp * 255)

# set max disparity and min disparity for post processing
maxdis = 235
mindis = 73

# Ignore untrusted depth
for i in range(h):
    for j in range(w):
        if (disp8[i][j] < mindis or disp8[i][j] > maxdis): disp8[i][j] = 0

# **********************************************************************************
if supervise_mode:
    imgLU[disp8 < 1, :] = 0
    cv2.imshow('disparity', disp8)
    cv2.imshow('Left Post-processing', imgLU)
    cv2.waitKey(0)

# ====================================================================================
# Visualization
# ====================================================================================
# In advance, you should install open3D (open3d.org)
# pip install open3d

pcd = o3d.geometry.PointCloud()

#  pc_points: array(Nx3), each row composed with x, y, z in the 3D coordinate
#  pc_color: array(Nx3), each row composed with R G,B in the rage of 0~1
pc_points = np.array([], np.float32) # 픽셀의 좌표 값
pc_color = np.array([], np.float32) # 픽셀의 RGB 값 (floating point 원래는 0~255이지만 0~1로 최대 max를 1로하기! -> 255만 나누기!)

# 3D reconstruction
# Concatenate pc_points and pc_color
# ****************************** Your code here (M-4) ******************************
# Get intrinsic parameter
# Focal length
fx = K_undist[0][0]
fy = K_undist[1][1]
# Principal point
U0 = K_undist[0][2]
V0 = K_undist[1][2]

# RGB to BGR for pc_points
imgLU = cv2.cvtColor(imgLU, cv2.COLOR_RGB2BGR)

# depth = inverse of disparity
depth = 255 - disp8

for v in range(h):
    for u in range(w):
        if(disp8[v][u] > 0): # ignore disparity (threshold 이하의 값을 가지는 pixel)
            # pc_points
            x = (u - U0) * depth[v][u] / fx
            y = (v - V0) * depth[v][u] / fy
            z = depth[v][u]
            pc_points = np.append(pc_points, np.array(np.float32(([x, y, z]))))
            pc_points = np.reshape(pc_points, (-1, 3))
            # pc_colors
            pc_color = np.append(pc_color, np.array(np.float32(imgLU[v][u] / 255)))
            pc_color = np.reshape(pc_color, (-1, 3))

# **********************************************************************************
#  add position and color to point cloud
pcd.points = o3d.utility.Vector3dVector(pc_points)
pcd.colors = o3d.utility.Vector3dVector(pc_color)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.0412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
cv2.destroyAllWindows()
#  end of code