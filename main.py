import cv2
import numpy as np
import uuid
import streamlit as st
from PIL import Image
from clearml import Task

# Ponds counting and measurement for eFishery Test Case, Problem 2
# - Preprocessing: Grayscale, Gaussian blur, canny edge detection
# - Crop/Ponds Segmentation using KMeans to several cluster
# - Find the crop clusters [ongoing]
# - Count the crop [ongoing]
# - Measure the area of crop [ongoing]
# - Probably using arcgis dataset for better accuracy of area [but lack of dataset in playground]

# streamlit
st.title('Problem 2: Ponds counting and measurement')
st.write('Harap menunggu jika input file belum muncul, sedang initialization ClearML..')

# stopping criteria kmeans
numiters = st.slider('Number of Iterations', 1, 100, 10)
epsilon = st.slider('Epsilon', 1, 10, 1)
attempts = st.slider('Number of Attempts', 1, 100, 10)

# clearml init
task = Task.init(project_name="ef-pond-count", task_name="pond")

uploaded_file = st.file_uploader("Upload the image")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    image = np.array(img)
    
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(gray, (5, 5), 0)

    grayFileName = "output/images/gray/gray-"+uuid.uuid4().hex[:8]+".png"
    im_pil = Image.fromarray(gray)
    im_pil = im_pil.save(grayFileName)

    blurFileName = "output/images/blur/blur-"+uuid.uuid4().hex[:8]+".png"
    im_pil = Image.fromarray(image)
    im_pil = im_pil.save(blurFileName)
        
    T1 = 100

    # canny edge detection
    edges_detected = cv2.Canny(image, T1, 3*T1) 
    images = [image , edges_detected]
    location = [121, 122]

    for loc, edge_image in zip(location, images): 
        fileName = "output/images/edge/edge-"+uuid.uuid4().hex[:8]+".png"
        im_pil = Image.fromarray(edge_image)
        im_pil = im_pil.save(fileName)
        
        # convert image value for KMeans
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # do kmeans processing
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, numiters, epsilon)

        # number of clusters (K)
        k = 3
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        labels = labels.flatten()
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(img.shape)

        # show Segmented Image
        segmentedFileName = "output/images/segmented/segmented-"+uuid.uuid4().hex[:8]+".png"
        im_pil = Image.fromarray(segmented_image)
        im_pil = im_pil.save(segmentedFileName)

        # cluster 0
        masked_image = np.copy(image)
        masked_image = masked_image.reshape((-1, 3))

        cluster = 0
        masked_image[labels == cluster] = [0, 0, 0]
        masked_image = masked_image.reshape(image.shape)

        maskedFileName = "output/images/mask-c0/masked-c0-"+uuid.uuid4().hex[:8]+".png"
        im_pil = Image.fromarray(masked_image)
        im_pil = im_pil.save(maskedFileName)

        # cluster 1
        masked_image1 = np.copy(image)
        masked_image1 = masked_image1.reshape((-1, 3))

        cluster = 1
        masked_image1[labels == cluster] = [0, 0, 0]
        masked_image1 = masked_image1.reshape(image.shape)

        maskedFileName1 = "output/images/mask-c1/masked-c1-"+uuid.uuid4().hex[:8]+".png"
        im_pil = Image.fromarray(masked_image1)
        im_pil = im_pil.save(maskedFileName1)

        # cluster 2
        masked_image2 = np.copy(image)
        masked_image2 = masked_image2.reshape((-1, 3))

        cluster = 2
        masked_image2[labels == cluster] = [0, 0, 0]
        masked_image2 = masked_image2.reshape(image.shape)

        maskedFileName2 = "output/images/mask-c2/masked-c2-"+uuid.uuid4().hex[:8]+".png"
        im_pil = Image.fromarray(masked_image2)
        im_pil = im_pil.save(maskedFileName2)
        
    # write the result
    st.write("Segmentation Cluster 1: ")
    st.image(cv2.cvtColor(masked_image,cv2.COLOR_BGR2RGB))
    st.write("Segmentation Cluster 2: ")
    st.image(cv2.cvtColor(masked_image1,cv2.COLOR_BGR2RGB))
    st.write("Segmentation Cluster 3: ")
    st.image(cv2.cvtColor(masked_image2,cv2.COLOR_BGR2RGB))