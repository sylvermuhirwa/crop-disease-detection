"""Cassava Leaf Check: a simple web app for a first look at a cassava leaf photo.

Run with:  streamlit run app/streamlit_app.py
"""
import json
from pathlib import Path

import streamlit as st
from PIL import Image

import engine                      # quality check, prediction, calibration, heatmap

APP_DIR = Path(__file__).parent
LANGUAGES = {"English": "i18n_en.json", "Kinyarwanda": "i18n_rw.json"}

st.set_page_config(page_title="Cassava Leaf Check", page_icon="🌿", layout="centered")


def show_list(items):
    """Show a list of short steps."""
    for item in items:
        st.markdown(f"- {item}")


# 1. Language: all texts come from the JSON files, so translations can be edited without touching code
choice = st.radio("Language / Ururimi", list(LANGUAGES), horizontal=True)
text = json.load(open(APP_DIR / LANGUAGES[choice], encoding="utf-8"))
ui, adv = text["ui"], text["advice"]

st.title(ui["app_title"])
st.write(ui["app_subtitle"])

# 2. Photo: upload, or take one with the camera
upload = st.file_uploader(ui["upload"], type=["jpg", "jpeg", "png"])
shot = st.camera_input(ui["camera"])
source = shot if shot is not None else upload

if source is None:
    st.info(ui["no_photo"])
    st.caption(ui["footer"])
    st.stop()

img = Image.open(source)
st.image(img, caption=ui["your_photo"])

# 3. Look at the photo
with st.spinner(ui["checking"]):
    result = engine.analyze(img)
status = result["status"]

st.subheader(ui["result"])

if status == "retake":
    st.warning(adv["retake"]["title"])
    st.write(f"**{ui['photo_problem']}:** {adv['retake'][result['problem']]}")
    st.write(f"**{ui['tips_title']}**")
    show_list(adv["retake"]["tips"])

elif status in ("uncertain", "uncertain_healthy"):
    info = adv["classes"][result["label"]]
    st.warning(adv["uncertain"]["title"])
    if status == "uncertain_healthy":
        st.write(adv["uncertain"]["healthy_body"])
    else:
        st.write(adv["uncertain"]["body"].replace("{label}", info["name"]))
    show_list(adv["uncertain"]["do_now"])

else:                               # confident answer
    info = adv["classes"][result["label"]]
    if result["label"] == "Healthy":
        st.success(info["name"])
    else:
        st.error(info["name"])
    st.write(f"**{ui['confidence']}:** {result['confidence']:.0%}")
    st.write(info["what_it_is"])
    st.write(f"**{ui['what_to_do']}**")
    show_list(info["do_now"])
    st.write(f"**{ui['get_help']}:** {info['get_help']}")

# 4. Heatmap (only if the photo passed the quality check, and only when asked for)
if status != "retake":
    if st.checkbox(ui["where_looked"]):
        st.image(engine.heatmap(img), caption=ui["heatmap_note"])

st.caption(adv["disclaimer"])
st.caption(ui["footer"])