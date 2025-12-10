import streamlit as st
from chapters import chapter1, chapter2

st.set_page_config(page_title="Image Processing Demo", layout="wide")

st.title("Image Processing Demonstration")

# Sidebar navigation
st.sidebar.header("Navigation")
chapter = st.sidebar.selectbox(
    "Select Chapter",
    ["Chapter 1: Color Transformations", 
     "Chapter 2: Pixel Operations"]
)

if chapter.startswith("Chapter 1"):
    chapter1.app()
elif chapter.startswith("Chapter 2"):
    chapter2.app()