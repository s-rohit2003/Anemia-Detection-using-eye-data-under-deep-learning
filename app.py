import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model(r'C:\Users\ROHIT\Downloads\drive-download-20240218T135952Z-001\anemia_detection_model.h5')

st.title("Eye-Anemia Detection")
upload_image = st.file_uploader(label = 'Upload image for detecting Eye-Anemia', type=['png','jpg','jpeg'])


def detect_anemia(upload_image):
    if upload_image is not None:
        global img
        img = Image.open(upload_image).resize((224,224))
        img_array = img_to_array(img)/255.0
        img_array = np.expand_dims(img_array,axis=0)

        prediction = model.predict(img_array)
        print(f'pred{prediction}')
    
        threshold = 0.5
        prediction_class = 1 if prediction[0][0] > threshold else 0

        return prediction_class


predict = detect_anemia(upload_image)

if upload_image is not None:
    st.image(img, width=400)

if predict == 0:
    st.markdown(f"<h2 style='text-align: left; color: green; font: Times New Roman;'>No Anemia Detected</h2>", 
                unsafe_allow_html=True)
elif predict == 1:
    st.markdown(f"<h2 style='text-align: left; color: red; font: Times New Roman;'>Anemia Detected</h2>", 
                unsafe_allow_html=True)