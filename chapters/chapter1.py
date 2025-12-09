import streamlit as st
from chapters.lessons.chapter1 import lesson1_1, lesson2_1, lesson3_1

def app():
    # Sidebar lesson navigation
    lesson = st.sidebar.radio(
        "Select Lesson",
        ["Task 1.1: Split RGB", 
         "Task 2.1: RGB to Grayscale", 
         "Task 3.1: Grayscale to Binary"]
    )

    if lesson.startswith("Task 1.1"):
        lesson1_1.app()
    elif lesson.startswith("Task 2.1"):
        lesson2_1.app()
    elif lesson.startswith("Task 3.1"):
        lesson3_1.app()
