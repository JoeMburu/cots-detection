import cv2
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import tempfile
import pandas as pd
from io import StringIO
import PIL
from PIL import Image
import numpy as np

count = 0


def predict(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.getvalue())
    tfile.seek(0)
    
    cap = cv2.VideoCapture(tfile.name)
    prediction_image_placeholder = st.empty()

    if (cap.isOpened() == False):
        st.write("Error opening the video stream or file")

    while(cap.isOpened()):
        # Read a frame from the video
        success, frame = cap.read()       

        if (success):

            # predict on this frame
                        
            pred_results = model.predict(frame, conf=confidence)
            boxes = pred_results[0].boxes
            for result in pred_results:
                boxes = result.boxes
                probs = result.probs            
            
            res_plotted =  pred_results[0].plot()[:, :, ::-1]
            prediction_image_placeholder.image(res_plotted, use_column_width=True, caption='COTS Detection on the video')
        else:
            break 
    cap.release()    

# Replace the relative path to your weight file
model_path = "weights/best.pt"

# Setting page layout
st.set_page_config(
    page_title="COT Starfish Detection using YOLOv8",  # Setting page title
    page_icon="./images/home.png",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Image/Video Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting videos
    #source_vid = st.file_uploader("Choose a video...", type=("mp4", "mp3", "avi"))
    video_file = st.file_uploader("Choose a video...", type=["mp4"])
    
    # Model Options
    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100
   

# Creating main page heading
st.title("COTS Detection using YOLOv8")
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if video_file is not None:

    # video placeholders
    image_placeholder = st.empty()
    

    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.getvalue())
    tfile.seek(0)    

    if st.sidebar.button('Detect COTS'):
        predict(video_file=video_file)
    
    cap = cv2.VideoCapture(tfile.name)

    if (cap.isOpened() == False):
        st.write("Error opening the video stream or file")

    while(cap.isOpened()):
        # Read a frame from the video
        success, frame = cap.read()

        if (success):

            # display this frame          
              
            frame_to_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image_placeholder.image(frame_to_display, caption='Original Video')
        else:
            break 
    cap.release()    


      

   
        







 
