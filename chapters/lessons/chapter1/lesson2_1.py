import streamlit as st
from PIL import Image
import numpy as np

def app():
    st.subheader("Task 2.1: RGB to Grayscale")

    st.markdown("""
**Purpose:**  
Transform a color image into a grayscale version by removing color information while preserving brightness and contrast. See how different colors translate to shades of gray.

**What You'll Do:**  
- Take a colorful RGB image and convert it to shades of gray  
- Use the human perception formula that weights colors differently  
- Compare the original color image with its grayscale counterpart

**How It Works:**  
Our eyes perceive green as brighter than red, and red as brighter than blue. The conversion formula reflects this:  
**Y = 0.2989 × R + 0.5870 × G + 0.1140 × B**  
- Green contributes most to brightness (59%)  
- Red contributes moderately (30%)  
- Blue contributes least (11%)

**Real-World Use:**  
Grayscale conversion simplifies images for:  
- Faster processing in computer vision  
- Better edge and texture detection  
- Medical imaging (X-rays, MRI)  
- Historic photo restoration  
- Preparing images for printing in black and white

**Notice This:**  
Bright colors like yellow appear as light gray, while dark colors like navy blue appear as dark gray. The overall brightness pattern stays the same, just without the color information.
""")

    # Load image
    img = Image.open("public/ch1.1.jpg")
    img_np = np.array(img)

    # Convert to grayscale manually
    gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
    gray = gray.astype(np.uint8)

    # Display images in a row
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
        st.markdown("**Original:** Full RGB image")

    with col2:
        st.image(gray, caption="Grayscale Image", use_container_width=True, channels="L")
        st.markdown("**Grayscale:** Intensity-only image (converted manually)")
