import streamlit as st
from PIL import Image
import numpy as np

def app():
    st.subheader("Task 3.1: Grayscale to Binary")

    # Description section
    st.markdown("""
**Purpose:**  
Transform a grayscale image into a clear black-and-white version using a simple threshold slider. See how different cutoff values reveal or hide details in your image.

**What You'll Do:**  
- Convert an RGB image to grayscale (removing color information)  
- Experiment with a slider to find the perfect black-and-white balance  
- Watch how changing the threshold value highlights different parts of the image

**How It Works:**  
A **binary image** contains only pure black (0) and pure white (255) pixels—no grays.  
**Thresholding** sets a cutoff point:  
- Pixels darker than the threshold → Black  
- Pixels brighter than the threshold → White

**Real-World Use:**  
This technique helps computers identify objects, separate foreground from background, and prepare images for text recognition, medical imaging, and document processing.

**Try This:**  
Move the slider and notice how higher thresholds make the image darker (more black), while lower thresholds make it brighter (more white).
""")

    # Load image
    img = Image.open("public/ch1.1.jpg")
    img_np = np.array(img)

    # Convert to grayscale manually
    gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
    gray = gray.astype(np.uint8)

    # Interactive threshold slider
    threshold = st.slider("Select Threshold", 0, 255, 128)
    binary = (gray > threshold) * 255
    binary = binary.astype(np.uint8)

    # Display images in a row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
        st.markdown("**Original:** Full RGB image")

    with col2:
        st.image(gray, caption="Grayscale Image", use_container_width=True, channels="L")
        st.markdown("**Grayscale:** Intensity-only image (converted manually)")

    with col3:
        st.image(binary, caption=f"Binary Image (threshold={threshold})", use_container_width=True, channels="L")
        st.markdown("**Binary:** Result after applying threshold")
