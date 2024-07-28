import cv2
import torch
import json
import time
import base64
import random
import imageio
import requests
import tempfile
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from utils.inference import process_frame
from roadsense_frontend.map import create_map

###Initial UI configuration:###
st.set_page_config(page_title="RoadSense Streamlit", page_icon="🛣️", layout="wide")

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
headers = {"Content-Type": "application/json"}

# 43.485321, -80.612374
# 43.518826, -80.502091
# 43.373572, -80.479572
# 43.452845, -80.399314


def create_data(result, location={}, severity=None):
    update_url = "http://localhost:5000/roadcondition/update"
    
    # For docker environment
    # update_url = "http://host.docker.internal:5000/roadcondition/update"
    
    
    js = json.loads(result.tojson())
    for item in js:
        severity = item["class"] if severity is None else severity
        update_payload = {
            "APIKey": token,
            "longitude": location["lg"],
            "latitude": location["lat"],
            "damage_type": item["name"],
            "severity": severity,
        }
        requests.post(update_url, headers=headers, json=update_payload)
        time.sleep(0.1)


def getbase64():
    """### gif from local file"""
    file_ = open("./detection.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    return data_url


def render_app():
    # reduce font sizes for input text boxes
    custom_css = """
        <style>
            .stTextArea textarea {font-size: 13px;}
            div[data-baseweb="select"] > div {font-size: 13px !important;}
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Left sidebar menu
    st.sidebar.header("RoadSense")

    # Set config for a cleaner menu, footer & background:
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if "chat_dialogue" not in st.session_state:
        st.session_state["chat_dialogue"] = []

    if "pre_prompt" not in st.session_state:
        st.session_state["pre_prompt"] = ""

    # Dropdown menu to select the model edpoint:
    # selected_option = st.sidebar.selectbox(
    #     "Choose a Model:", ["CNN", "MobileNetV2", "ResNet50"], key="model"
    # )

    # if selected_option == "CNN":
    #     st.session_state["models"] = "CNN"
    # elif selected_option == "MobileNetV2":
    #     st.session_state["models"] = "MobileNetV2"
    # elif selected_option == "ResNet50":
    #     st.session_state["models"] = "ResNet50"

    btn_col1, btn_col2 = st.sidebar.columns(2)

    def clear_history():
        st.session_state["chat_dialogue"] = []

    clear_chat_history_button = btn_col1.button(
        "Clear History", use_container_width=True, on_click=clear_history
    )

    # add links to relevant resources for users to select
    st.sidebar.write(" ")

    # text1 = "llama2-chatbot"
    # text1_link = "https://github.com/a16z-infra/llama2-chatbot"
    # logo1 = "https://cdn.pixabay.com/photo/2022/01/30/13/33/github-6980894_1280.png"

    # st.sidebar.markdown(
    #     "**Reference Code and Model:**  \n"
    #     f"<img src='{logo1}' style='height: 2em'> [{text1}]({text1_link})  \n",
    #     unsafe_allow_html=True,
    # )

    icon_givors = "https://avatars.githubusercontent.com/u/17698876?v=4"
    icon_hosang = "https://avatars.githubusercontent.com/u/156028780?v=4"
    icon_yeji = "https://avatars.githubusercontent.com/u/38099574?v=4"
    icon_darshik = "https://avatars.githubusercontent.com/u/70846020?v=4"
    icon_lakshay = "https://avatars.githubusercontent.com/u/133694401?v=4"

    st.sidebar.write(" ")
    st.sidebar.markdown(
        "**Contributors:**  \n"
        f"<img src='{icon_darshik}' style='height: 2em'> [{'**Darshik**'}]({'https://github.com/imdarshik'})  \n"
        f"<img src='{icon_yeji}' style='height: 2em'> [{'**Yeji**'}]({'https://github.com/dut0817'})  \n"
        f"<img src='{icon_lakshay}' style='height: 2em'> [{'**Lakshay**'}]({'https://github.com/lakshay1505'})  \n"
        f"<img src='{icon_hosang}' style='height: 2em'> [{'**Hosang**'}]({'https://github.com/hyoon1'})  \n"
        f"<img src='{icon_givors}' style='height: 2em'> [{'**Givors Ku**'}]({'https://github.com/guggg'})",
        unsafe_allow_html=True,
    )

    # st.title(st.session_state["models"])

    uploaded_file = st.file_uploader(
        "Upload an image or video", type=("jpg", "jpeg", "png", "mp4")
    )

    for message in st.session_state.chat_dialogue:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if uploaded_file:

        lg = round(random.uniform(-80.576864, -80.423523), 6)
        lat = round(random.uniform(43.397781, 43.494617), 6)

        location = {"lg": lg, "lat": lat}

        if "mp4" in uploaded_file.type:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            # Capture video stream
            cap = cv2.VideoCapture(tfile.name)

            frame_count = 0
            frames = []

            placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count == 50:
                    break

                model = YOLO("./models/yolov8n_best.pt")
                processed_frame, result, severity = process_frame(frame, model)
                create_data(result, location, severity)

                # for generating gif
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)

                imageio.mimsave("./detection.gif", frames, loop=10000, duration=100)
                base64 = getbase64()
                placeholder.markdown(
                    f'<img src="data:image/gif;base64,{base64}" alt="cat gif">',
                    unsafe_allow_html=True,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_count += 1

                create_map()

            cap.release()
            cv2.destroyAllWindows()

        else:
            model = YOLO("./models/yolov8n_best.pt")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, width=150, channels="BGR")

            with torch.no_grad():
                with st.spinner("Thinking..."):
                    result = model.predict(source=image, conf=0.35, iou=0.5)[0]
                    img = result.save()
                    st.image(img, caption="Result Image", use_column_width=True)
                    create_data(result, location, None)


render_app()
