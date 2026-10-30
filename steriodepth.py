import cv2 as cv
import numpy as np
img1 = cv.imread('./im0.png')
img2 = cv.imread('./im1.png')
grey1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
grey2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=21
)
disparity = stereo.compute(grey1, grey2)
disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
disparity = np.uint8(disparity)
heatmap = cv.applyColorMap(disparity, cv.COLORMAP_JET)
cv.imshow('gray', disparity)
cv.imshow("Heatmap", heatmap)
cv.imwrite("heatmap_output.png", heatmap)
cv.imwrite('grayscale.png',disparity)
cv.waitKey(0)