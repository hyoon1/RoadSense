from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO
import requests
import json
import random
import time

###Initial UI configuration:###
st.set_page_config(page_title="RoadSense", page_icon="🛣️", layout="wide")

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
headers = {"Content-Type": "application/json"}


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

    uploaded_file = st.file_uploader("Upload a Image", type=("jpg", "jpeg", "png"))

    for message in st.session_state.chat_dialogue:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    model = YOLO("./models/yolov8n_best.pt")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=150, channels="BGR")

        with torch.no_grad():
            with st.spinner("Thinking..."):
                result = model.predict(source=image, conf=0.35, iou=0.5)[0]
                img = result.save()
                st.image(img, caption="Result Image", use_column_width=True)
                update_url = "http://localhost:5000/roadcondition/update"

                js = json.loads(result.tojson())
                for item in js:
                    update_payload = {
                        "APIKey": token,
                        "longtitude": round(random.uniform(180, -180), 6),
                        "latitude": round(random.uniform(90, -90), 6),
                        "damage_type": item["name"],
                        "severity": item["class"],
                    }
                    requests.post(update_url, headers=headers, json=update_payload)
                    time.sleep(0.1)


render_app()
