import streamlit as st
from chapters.lessons.chapter4 import lesson1_4, lesson2_4, lesson3_4, lesson4_4, lesson5_4

def app():
    # Sidebar lesson navigation
    lesson = st.sidebar.radio(
        "Select Lesson",
        [
            "Task 1.4: Frequencies in Images",
            "Task 2.4: Fourier Transform",
            "Task 3.4: Spectrum Analysis",
            "Task 4.4: Frequency Filters",
            "Task 5.4: Real Applications"
        ]
    )

    if lesson.startswith("Task 1.4"):
        lesson1_4.app()
    elif lesson.startswith("Task 2.4"):
        lesson2_4.app()
    elif lesson.startswith("Task 3.4"):
        lesson3_4.app()
    elif lesson.startswith("Task 4.4"):
        lesson4_4.app()
    elif lesson.startswith("Task 5.4"):
        lesson5_4.app()