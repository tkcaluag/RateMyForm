# app.py

import streamlit as sl
from PIL import Image

sl.set_page_config(page_title="RateMyForms")

sl.title("RateMyForm")
sl.write("Upload a photo of your exercise form.")

exercise = sl.selectbox(
    "Choose an exercise",
    ["Squat"],
)

upload_file = sl.file_uploader(type=["jpg","jpeg","png"])

if upload_file is not None:
    try:
        image = Image.open(upload_file).convert("RGB")

        sl.subheader("Your Uploaded Photo")
        sl.image(image)

        sl.info("Analyzing exercise pose..."
                "Evaluating...")

    except Exception:
        sl.error("Unable to open this image.")

else:
    sl.info("Upload a photo to get started")
        