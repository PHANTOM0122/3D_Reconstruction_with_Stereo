# 3D_Reconstruction
## Common Step in both methods
1) **Image calibration**: Calibrating with checkerboard method, get intrinsic parameter of camera.
2) **3D Visualization**: Visualize 3D using open3d with point cloud from depth map and RGB images.

## 3D Reconstuction using disparity map
Disparity map can be obtained between images, simply translated with X-axis. As depth and disparity are inversely propotional, we can estimate relative depth values then reconstruct 3D. The reconstucted 3D point clouds will be dense since depth can be inffered from disparity map, which is dense also.

### Results
<img src="https://user-images.githubusercontent.com/50229148/125709736-857c5aa1-f431-48bd-846f-88ee052886b7.png"  width="700" height="370">

<img src="https://user-images.githubusercontent.com/50229148/125709808-0c0cd529-f0ed-4f10-97cb-a9bd4a25b6ab.png"  width="350" height="370" /><img src="https://user-images.githubusercontent.com/50229148/125709836-322bee24-a60a-45a2-a773-3deb9df739e2.png"  width="350" height="370">
<img src="https://user-images.githubusercontent.com/50229148/125710002-23587134-fb34-402e-a739-5037b166d4c4.png"  width="350" height="370" /><img src="https://user-images.githubusercontent.com/50229148/125710034-c254ec1e-19df-4098-9966-a810322a3c48.png"  width="350" height="370">

## 3D Reconstuction using Feature-based algorighm
![image](https://user-images.githubusercontent.com/50229148/198946804-b7d3d9b9-5810-41da-a382-019cb60c8dec.png)<br>
With SIFT algorithm between two images, obtaining correspondence pairs is possible. We can find find 3D position which is the intersection of rays from each camera. The result 3D point clouds will be sparse in contrast to Disparity method since the position of points cab be calculated from feature points, which are also sparse.

### Results
* **Left & Right Image** <br>
<img src="https://github.com/PHANTOM0122/3D_Reconstruction_with_SFM/blob/main/Reconstruction_with_Features/left2.jpg"  width="350" height="170"/><img src="https://github.com/PHANTOM0122/3D_Reconstruction_with_SFM/blob/main/Reconstruction_with_Features/right2.jpg"  width="350" height="170"><br>

* **Extracted feature points**<br>
<img src="https://github.com/PHANTOM0122/3D_Reconstruction_with_SFM/blob/main/Reconstruction_with_Features/Matched_feature_result.png" width="350" height="170"><br>

* **Sparse points Clouds**<br>
<img src ="https://github.com/PHANTOM0122/3D_Reconstruction_with_SFM/blob/main/Reconstruction_with_Features/points_result.png" width="500" height="500"><br>

## Requirement
- python 3.x
- openCV 3.x
- open3D

### Backprojection algorithm
```python
# Loop through each pixel in the image  
for v in range(height):    
  for u in range(width):    
    # Apply equation in fig 4
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy
    z = depth[v, u]
```


