import streamlit as st
from chapters.lessons.chapter3 import lesson1_3, lesson2_3, lesson3_3, lesson4_3, lesson5_3

def app():
    # Sidebar lesson navigation
    lesson = st.sidebar.radio(
        "Select Lesson",
        [
            "Task 1.3: Convolution Basics",
            "Task 2.3: Sobel Edge Detection",
            "Task 3.3: Unsharp Masking",
            "Task 4.3: Sharpening Kernel",
            "Task 5.3: Image Enhancement"
        ]
    )

    if lesson.startswith("Task 1.3"):
        lesson1_3.app()
    elif lesson.startswith("Task 2.3"):
        lesson2_3.app()
    elif lesson.startswith("Task 3.3"):
        lesson3_3.app()
    elif lesson.startswith("Task 4.3"):
        lesson4_3.app()
    elif lesson.startswith("Task 5.3"):
        lesson5_3.app()