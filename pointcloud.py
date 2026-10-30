import cv2 as cv
import numpy as np
import open3d as o3d

def process_image(img1,img2):
    
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=21
    )

    disparity = stereo.compute(gray1, gray2)
    disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
    disparity = np.uint8(disparity)
    heatmap = cv.applyColorMap(disparity, cv.COLORMAP_JET)

    colours = []
    points = []
    rows,cols = disparity.shape
    for x in range (rows):
        for y in range(cols):
            z = disparity[x,y]
            points.append((x,y,z))
            colours.append(img1[x,y])

    points = np.array(points, dtype=np.float64)
    colors = np.array(colours, dtype=np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return disparity,heatmap,pcd


