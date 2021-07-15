# 3D_Reconstruction
## 3D-Reconstuction using disparity map<br>

## Requirement
- python 3.x
- openCV 3.x
- open3D

## Step
1) Image calibration: Calibrating with checkerboard method, get intrinsic parameter of camera.
2) Stereo Vision(matching): Get disparity map from 2 images using Stereo BM.
3) 3D reconstruction: Get depth map from disparity map. Then using backprojection, visualize 3D using open3d with point cloud from depth map and RGB channels.

## Backprojection algorithm
```python
# Loop through each pixel in the image  
for v in range(height):    
  for u in range(width):    
    # Apply equation in fig 4
    x = (u - u0) * depth[v, u] / fx
    y = (v - v0) * depth[v, u] / fy
    z = depth[v, u]
```

