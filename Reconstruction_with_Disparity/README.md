# 3D_Reconstruction
## 3D-Reconstuction using disparity map<br>

<img src="https://user-images.githubusercontent.com/50229148/125709736-857c5aa1-f431-48bd-846f-88ee052886b7.png"  width="700" height="370">

<img src="https://user-images.githubusercontent.com/50229148/125709808-0c0cd529-f0ed-4f10-97cb-a9bd4a25b6ab.png"  width="350" height="370" /><img src="https://user-images.githubusercontent.com/50229148/125709836-322bee24-a60a-45a2-a773-3deb9df739e2.png"  width="350" height="370">
<img src="https://user-images.githubusercontent.com/50229148/125710002-23587134-fb34-402e-a739-5037b166d4c4.png"  width="350" height="370" /><img src="https://user-images.githubusercontent.com/50229148/125710034-c254ec1e-19df-4098-9966-a810322a3c48.png"  width="350" height="370">


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

