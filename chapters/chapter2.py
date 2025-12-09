import streamlit as st
from chapters.lessons.chapter2 import lesson2_1, lesson2_2

def app():
    st.header("Chapter 2")
    
    lesson = st.radio("Select Lesson", ["Task 2.1", "Lesson 2.2"])
    
    if lesson == "Task 2.1":
        lesson2_1.app()
    elif lesson == "Lesson 2.2":
        lesson2_2.app()
