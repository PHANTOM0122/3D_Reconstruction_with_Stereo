#!/usr/bin/env python
# coding: utf-8

# # [SWCON336] Robot Sensor Data processing
# 
# Professor: Hyo Seok Hwang (hyoseok@khu.ac.kr)
# 
# Teaching Assistant: Hee Joon Moon (wilko97@khu.ac.kr)
# 
# * 본 실습과제에 대한 질문은 수업도우미 문희준(wilko97@khu.ac.kr)에게 메일 주시면 됩니다.

# # Feature기반 3차원 복원
# 
# 본 실습과제에서는 2장의 이미지에서 공통된 특징점들을 SIFT알고리즘을 이용하여 추출한 이후, 이 점들에 대해서 3차원 scene을 복원합니다.<br>
# 
# 
# # 실습 내용
# - Camera Calibration
# - Feature Matching과 RANSAC을 이용한 Outlier rejection
# - Fundamental Matrix 구하기
# - Fundamental Matrix를 이용한 Essential Matrix 구하기
# - Essential Matrix를 이용한 Camera Pose 추정
# - Triangulation을 이용한 3차원 추정
# - Perspective n-point
# - Bundle Adjustment
# 
# # Reference
# - https://cmsc426.github.io/sfm/
# - https://github.com/Ashok93/Structure-From-Motion-SFM-/blob/master/main.py

# # 필요 라이브러리 설치 및 임포트
# 본 과제에서는 Numpy, matplotlib, openCV, open3d를 이용합니다.<br>
# 아래의 코드에서 필요한 라이브러리의 주석을 제거하여 설치하면 됩니다

# pip install numpy # Numpy 설치
# pip install matplotlib # Matplotlib 설치
# pip install opencv-contrib-python opencV # Jupyter notebook에서 opencv를 실행하기 위해 설치합니다
# pip install open3d # 3차원 점들의 시각화를 위한 라이브러리인 open3d를 설치합니다


# Import libraries
import numpy as np
import cv2
import glob
import open3d as o3d
import matplotlib.pyplot as plt

##########################################################################################################
# # Camera Calibration

# - OpenCV Checkerboard: https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html
# - OpenCV Calibration: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
# - https://learnopencv.com/camera-calibration-using-opencv/

# In[3]:


# Set directory path (images capturing check pattern)
calibration_dir_path = 'cal/cal*.png'
calibration_images = glob.glob(calibration_dir_path)


# In[4]:


# intrinsic parameters and distortion coefficient
# With these parameter, you can get undistorted image and new intrinsic parameter of them (K_undist)
K = np.array([], dtype=np.float32) # camera intrinsic parameter
dist = np.array([], dtype=np.float32) # distortion coefficient

# new matrix for undistorted intrinsic parameter
K_undist = np.array([], dtype=np.float32)

print(K)


# In[5]:


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.001)


# In[6]:


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


# In[7]:


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


# In[8]:


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


# In[9]:

# Calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# New undistorted intrinsic parameter
K_undist, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w, h),1, (w, h))


# In[10]:


print('1-1. Calibration: K matrix') # Intrinsic parameter K 출력
print(K)
print('1-2. Calibration: distortion coefficients') # Intrinsic parameter 보정 계수 출력
print(dist)
print('1-3. Calibration: Undistorted K matrix') # 보정된 Intrinsic parameter K 출력
print(K_undist)

for fname in calibration_images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    img_undist = cv2.undistort(gray, K, dist, None, K_undist)


# Set your left and right images
# Image size must be same with calibration Images size(=checkboards Images size) 
imgL = cv2.imread('left2.jpg')
imgR = cv2.imread('right2.jpg')

# convert to grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Convert to undistorted images
imgLU = np.array([]) # imgLU: undistorted image of imgL
imgRU = np.array([]) # imgRU: undistorted image of imgR
grayLU = np.array([]) # grayLU: undistorted image of grayL
grayRU = np.array([]) # grayRU: undistorted image of grayR

# undistorted images 얻기
imgLU = cv2.undistort(imgL, K, dist, None, K_undist)
imgRU = cv2.undistort(imgR, K, dist, None, K_undist)
grayLU = cv2.undistort(grayL, K, dist, None, K_undist)
grayRU = cv2.undistort(grayR, K, dist, None, K_undist)

# Display Original Image and Undistorted Image
cv2.imshow('original image', grayL)
cv2.waitKey(0)
cv2.imshow('undistorted image', grayLU)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot function
# 2개의 좌우 이미지에서 검출된 특징점들을 선들로 이어주어 시각화 하는 함수입니다
def drawlines(img1, img2, lines, pts1, pts2):

    ''' img1 - image on which we draw the epilines for the points in
        img2 lines - corresponding epilines '''

    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

########################################################################################################################
# # Feature matching using SIFT algorithm
########################################################################################################################

# Create SIFT
# sift = cv2.xfeatures2d.SIFT_create() # OpenCV 4.5 미만 버젼 사용중일 시
sift = cv2.SIFT_create() # OpenCV 4.5 이상의 버전 사용중일 시

# Find keypoints and descriptors using SIFT
kp1, des1 = sift.detectAndCompute(imgLU, None)
kp2, des2 = sift.detectAndCompute(imgRU, None)

# FLANN Parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# FLANN Matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Matching
matches = flann.knnMatch(des1, des2, k=2)
good = []
pts1 = []
pts2 = []

# Get Matched points under distance's threshold
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        good.append([m])
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Draws the small circles on the locations of keypoints
img_matched = cv2.drawMatchesKnn(imgLU, kp1, imgRU, kp2, good, None, flags=2)
plt.imshow(img_matched)
plt.show()

# Print number of matched feature points
print('Matched Num:', len(pts1))

# Set array for keypoints
pts1 = np.array(pts1)
pts2 = np.array(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Find epilines corresponding to points in right image (second image) and drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(grayLU, grayRU, lines1, np.int32(pts1), np.int32(pts2))

# Find epilines corresponding to points in left image (first image) and drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(grayLU, grayRU, lines2, np.int32(pts1), np.int32(pts2))

# Display
plt.imshow(img5)
plt.show()

plt.imshow(img6)
plt.show()


# Find new Essential matrix from Fundatmental matrix above.<br>
# Also find projection matrix per each image

# In[18]:


E = np.matmul(np.matmul(np.transpose(K), F), K)
R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
R_t_1 = np.empty((3,4))
P1 = np.matmul(K, R_t_0)
P2 = np.empty((3,4))
print("The new essential matrix is \n" + str(E))

retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
            
print("I+0 \n" + str(R_t_0))

print("Mullllllllllllll \n" + str(np.matmul(R, R_t_0[:3,:3])))

R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())

print("The R_t_0 \n" + str(R_t_0))
print("The R_t_1 \n" + str(R_t_1))

P2 = np.matmul(K, R_t_1)

print("The projection matrix 1 \n" + str(P1))
print("The projection matrix 2 \n" + str(P2))


# # Triangulation
# - Reference : https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c

# In[19]:


pts1 = np.transpose(pts1)
pts2 = np.transpose(pts2)

points_3d = cv2.triangulatePoints(P1, P2, pts1, pts2)
points_3d /= points_3d[3]

opt_variables = np.hstack((P2.ravel(), points_3d.ravel(order="F")))
num_points = len(pts2[0])

X = []
Y = []
Z = []

X = np.concatenate((X, points_3d[0]))
Y = np.concatenate((Y, points_3d[1]))
Z = np.concatenate((Z, points_3d[2]))

points = np.zeros((num_points, 3))
points[:,0] = X
points[:,1] = Y
points[:,2] = Z


# # Visualization 

# In[20]:


pcd = o3d.geometry.PointCloud()

#  pc_points: array(Nx3), each row composed with x, y, z in the 3D coordinate
#  pc_color: array(Nx3), each row composed with R G,B in the rage of 0~1

pc_points = np.array(points, np.float32) # 픽셀의 좌표 값
pc_color = np.array([], np.float32) # 픽셀의 RGB 값 (floating point 원래는 0~255이지만 0~1로 최대 max를 1로하기! -> 255만 나누기!)

# RGB to BGR for pc_points
imgLU = cv2.cvtColor(imgLU, cv2.COLOR_RGB2BGR)

point_color = np.transpose(pts1)

for i in range(len(point_color)):
    u = np.int32(point_color[i][1])
    v = np.int32(point_color[i][0])
    
    # pc_colors
    pc_color = np.append(pc_color, np.array(np.float32(imgLU[u][v] / 255)))
    pc_color = np.reshape(pc_color, (-1, 3))


# In[ ]:


# add position and color to point cloud
pcd.points = o3d.utility.Vector3dVector(pc_points)
pcd.colors = o3d.utility.Vector3dVector(pc_color)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.0412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
cv2.destroyAllWindows()

