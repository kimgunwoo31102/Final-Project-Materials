import streamlit as st
import PIL
import cv2
import numpy as np
import model4add
import pandas as pd
import io
import altair as alt
import math

def categorize_cars(num_cars):
    if num_cars <= 5:
        return 5
    elif num_cars <= 10:
        return 10
    elif num_cars <= 15:
        return 15
    elif num_cars <= 20:
        return 20
    elif num_cars <= 30:
        return 30
    else:
        return '30+'

def round_to_nearest_5(num_cars):
    if num_cars <= 30:
        return (math.ceil(num_cars / 5) * 5)
    else:
        return '30+'

def play_video(video_source, conf_threshold):
    camera = cv2.VideoCapture(video_source)
    car_counts = []
    frame_numbers = []
    rounded_counts = []

    st_frame = st.empty()
    st_chart = st.empty()
    
    frame_number = 0
    
    st.write("The blue graph shows the exact number of vehicles.")
    st.write("The red graph represents units of 5.")

    while(camera.isOpened()):
        ret, frame = camera.read()
        
        if ret:
            visualized_image, num_cars = model4add.predict_image(frame, conf_threshold)
            car_counts.append(num_cars)
            frame_numbers.append(frame_number)
            rounded_counts.append(round_to_nearest_5(num_cars))
            frame_number += 1

            st_frame.image(visualized_image, channels="BGR")

            data = pd.DataFrame({
                'Frame': frame_numbers,
                'Cars': car_counts,
                'Rounded Cars': rounded_counts
            })

            base_chart = alt.Chart(data).encode(
                x='Frame:O'
            )

            cars_line = base_chart.mark_line(color='blue').encode(
                y=alt.Y('Cars:Q', title='Number of Cars')
            )

            rounded_cars_line = base_chart.mark_line(color='red').encode(
                y=alt.Y('Rounded Cars:Q', title='Rounded Cars')
            )

            chart = alt.layer(
                cars_line,
                rounded_cars_line
            ).properties(
                title='Number of Cars Detected Over Time'
            )
            st_chart.altair_chart(chart, use_container_width=True)
        else:
            camera.release()
            break

st.set_page_config(
    page_title="Traffic Jam",
    page_icon=":car:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Traffic Jam Detection :car:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 65)) / 100

if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=["jpg", "png"])
    
    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image, num_cars = model4add.predict_image(uploaded_image_cv, conf_threshold)
        
        st.image(visualized_image, channels="BGR")
        st.write(f"Detected cars: {num_cars}")
    else:
        st.image("data/image start.png")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image.")

elif source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose a video.", type=["mp4"])
    
    if input is not None:
        g = io.BytesIO(input.read())
        temporary_location = "data/upload.mp4"
        
        with open(temporary_location, "wb") as out:
            out.write(g.read())
        out.close()
        
        play_video(temporary_location, conf_threshold)
    else:
        st.video("data/video start.mp4")
        st.write("Click on 'Browse Files' in the sidebar to run inference on a video.")

elif source_radio == "WEBCAM":
    play_video(0, conf_threshold)
