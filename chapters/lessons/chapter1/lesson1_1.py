import streamlit as st
from PIL import Image
import numpy as np

def app():
    st.subheader("Task 1.1: Split Image Channels")

    # Description section
    st.markdown("""
**Purpose:**  
See how a full-color image is built from three primary color layers. Discover what happens when you separate the Red, Green, and Blue components that combine to create every color you see.

**What You'll Do:**  
- Take a full-color image and split it into its Red, Green, and Blue layers  
- Use the slider to visually move channels from separate to combined  
- Watch how three separate grayscale images overlap to create a full-color image  
- Understand how these three simple colors combine to create complex images

**How It Works:**  
Every digital color image is actually made of **three separate layers**:  
- **Red layer:** Shows where red light is strong  
- **Green layer:** Shows where green light is strong  
- **Blue layer:** Shows where blue light is strong  

When these layers perfectly overlap, they create all the colors in the original image. The slider below lets you control how much they overlap!

**Real-World Use:**  
Splitting color channels helps with:  
- Color correction and photo editing  
- Creating artistic effects and filters  
- Scientific imaging (like satellite photos)  
- Detecting specific colors in computer vision  
- Preparing images for different printing processes

**Try This:**  
Slide from left to right to watch the magic! Start with separated channels, then slide to see them come together in the center to form the original color image.
""")

    # Load image
    img = Image.open("public/ch1.1.jpg")
    img_np = np.array(img)
    
    # Get image dimensions
    height, width = img_np.shape[:2]
    
    # Split channels manually
    red = np.zeros_like(img_np)
    red[:, :, 0] = img_np[:, :, 0]

    green = np.zeros_like(img_np)
    green[:, :, 1] = img_np[:, :, 1]

    blue = np.zeros_like(img_np)
    blue[:, :, 2] = img_np[:, :, 2]
    
    # Create grayscale versions for display
    red_gray = red[:, :, 0]
    green_gray = green[:, :, 1] 
    blue_gray = blue[:, :, 2]
    
    # Create canvas for the animation
    canvas_width = width * 3  # Space for 3 images side by side
    canvas_height = height
    
    # Add slider for channel combination
    st.markdown("---")
    st.markdown("### 🎚️ Channel Combination Control")
    
    combination = st.slider(
        "Slide to combine channels →", 
        min_value=0, 
        max_value=100, 
        value=0,
        help="0% = channels fully separated, 100% = channels perfectly combined in center"
    )
    
    # Calculate positions based on slider
    # At 0%: Red on left, Green in middle, Blue on right
    # At 100%: All in center (overlapping)
    
    # Calculate center position for each channel
    center_x = canvas_width // 2 - width // 2
    
    # Calculate starting positions (when separated)
    red_start_x = 0
    green_start_x = width
    blue_start_x = width * 2
    
    # Calculate current positions based on slider
    progress = combination / 100.0
    
    red_x = int(red_start_x + (center_x - red_start_x) * progress)
    green_x = int(green_start_x + (center_x - green_start_x) * progress)
    blue_x = int(blue_start_x + (center_x - blue_start_x) * progress)
    
    # Create canvas for display
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Place red channel (as red color)
    canvas[:, red_x:red_x+width, 0] = red_gray
    
    # Place green channel (as green color)
    canvas[:, green_x:green_x+width, 1] = green_gray
    
    # Place blue channel (as blue color)
    canvas[:, blue_x:blue_x+width, 2] = blue_gray
    
    # Display the animated result
    st.image(canvas, caption=f"Channel Combination: {combination}%", use_container_width=True)
    
    # Add explanation based on slider position
    if combination == 0:
        st.markdown("**Fully Separated:** Each channel is in its own position showing individual color information")
    elif combination < 100:
        st.markdown(f"**Partially Combined ({combination}%):** Channels are moving toward the center")
    else:
        st.markdown("**Fully Combined (100%):** All channels perfectly overlap to create the original RGB image!")
    
    # Show the original combined image separately for comparison
    if combination < 100:
        st.markdown("---")
        st.markdown("### 🔍 Compare with Original Image")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(canvas, caption="Current Channel Positions", use_container_width=True)
        
        with col2:
            # Show what they're moving toward
            combined_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            combined_canvas[:, :, 0] = red_gray
            combined_canvas[:, :, 1] = green_gray
            combined_canvas[:, :, 2] = blue_gray
            st.image(combined_canvas, caption="Target: Fully Combined Image", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Individual Channel Analysis")
    
    # Display static channel images in a row
    st.markdown("**Individual Channels (Static View):**")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Create red display (black background with red channel)
        red_display = np.zeros_like(img_np)
        red_display[:, :, 0] = img_np[:, :, 0]
        st.image(red_display, caption="Red Channel", use_container_width=True)
        st.markdown("**Red Channel:** Only shows red information")

    with col2:
        # Create green display (black background with green channel)
        green_display = np.zeros_like(img_np)
        green_display[:, :, 1] = img_np[:, :, 1]
        st.image(green_display, caption="Green Channel", use_container_width=True)
        st.markdown("**Green Channel:** Only shows green information")

    with col3:
        # Create blue display (black background with blue channel)
        blue_display = np.zeros_like(img_np)
        blue_display[:, :, 2] = img_np[:, :, 2]
        st.image(blue_display, caption="Blue Channel", use_container_width=True)
        st.markdown("**Blue Channel:** Only shows blue information")