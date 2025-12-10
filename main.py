import streamlit as st
from chapters import chapter1, chapter2, chapter3, chapter4

st.set_page_config(page_title="Image Processing Demo", layout="wide")

st.title("Image Processing Demonstration")

# Sidebar navigation
st.sidebar.header("Navigation")
chapter = st.sidebar.selectbox(
    "Select Chapter",
    [
        "Chapter 1: Color Transformations", 
        "Chapter 2: Pixel Operations",
        "Chapter 3: Spatial Filtering",
        "Chapter 4: Frequency Domain Filtering"  # NEW CHAPTER ADDED
    ]
)

if chapter.startswith("Chapter 1"):
    chapter1.app()
elif chapter.startswith("Chapter 2"):
    chapter2.app()
elif chapter.startswith("Chapter 3"):
    chapter3.app()
elif chapter.startswith("Chapter 4"):  # NEW
    chapter4.app()