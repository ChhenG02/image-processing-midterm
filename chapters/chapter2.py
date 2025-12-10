import streamlit as st
from chapters.lessons.chapter2 import lesson1_2, lesson2_2, lesson3_2, lesson4_2, lesson5_2, lesson6_2

def app():
    # Sidebar lesson navigation
    lesson = st.sidebar.radio(
        "Select Lesson",
        [
            "Task 1.2: Image Arithmetic",
            "Task 2.2: Noise Reduction",
            "Task 3.2: Transparency & Blending",
            "Task 4.2: Masking & Isolation",
            "Task 5.2: Negative Image",
            "Task 6.2: Histogram & Equalization"
        ]
    )

    if lesson.startswith("Task 1.2"):
        lesson1_2.app()
    elif lesson.startswith("Task 2.2"):
        lesson2_2.app()
    elif lesson.startswith("Task 3.2"):
        lesson3_2.app()
    elif lesson.startswith("Task 4.2"):
        lesson4_2.app()
    elif lesson.startswith("Task 5.2"):
        lesson5_2.app()
    elif lesson.startswith("Task 6.2"):
        lesson6_2.app()