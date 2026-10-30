import streamlit as st
import numpy as np
import cv2 as cv
from pointcloud import process_image
import open3d as o3d

st.title('3D Scene Reconstruction from Stereo Images')
st.write('This application takes two stereo images as input and gives disparity map, colorized depth heatmap and fully colored 3d point cloud as output.')

col1, col2 = st.columns(2)
with col1:
    left_image = st.file_uploader('Upload left image',type=['png','jpg'])
    if left_image is not None:
        st.image(left_image, caption ='Left Image')
with col2:
    right_image = st.file_uploader('Upload right image',type=['png','jpg'])
    if right_image is not None:
        st.image(right_image,caption='Right Image')

if left_image is not None and right_image is not None :

    if st.button('Analyze'): 

        file_bytes1 = np.frombuffer(left_image.read(),np.uint8)
        img1 = cv.imdecode(file_bytes1,cv.IMREAD_COLOR)
        files_bytes2 = np.frombuffer(right_image.read(),np.uint8)
        img2 = cv.imdecode(files_bytes2, cv.IMREAD_COLOR)
        
        with st.spinner('processing .. this may take a minute'):
            disparity, heatmap, pcd = process_image(img1,img2)

            st.session_state['disparity'] = disparity
            st.session_state['heatmap'] = heatmap

            o3d.io.write_point_cloud("pointcloud.ply", pcd)

    if 'heatmap' in st.session_state:
        st.image(st.session_state['disparity'], caption = 'disparity map')
        st.image(st.session_state['heatmap'],caption='depth heatmap')
        
        with open("pointcloud.ply", "rb") as f:
            st.download_button("Download 3d point cloud", f, "pointcloud.ply")


