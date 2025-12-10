import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import os

def app():
    st.title("🎵 Understanding Frequencies in Images")
    
    # Progressive learning tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Images are Like Music", 
        "2️⃣ Low vs High Frequencies", 
        "3️⃣ See Frequencies Live",
        "4️⃣ Filter Demo",
        "5️⃣ Practical Exercise"
    ])
    
    # Try to load images from lab5 directory
    image_dir = "public/lab5"
    available_images = []
    
    if os.path.exists(image_dir):
        available_images = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if available_images:
        selected_image = st.sidebar.selectbox("📷 Choose Image", available_images, key="freq_demo_img")
        img_path = os.path.join(image_dir, selected_image)
        img = Image.open(img_path).convert('L')
    else:
        # Create a synthetic image if no images found
        st.sidebar.warning("Using demo image - add images to 'public/lab5'")
        img = create_demo_image()
    
    img_np = np.array(img).astype(float)
    
    # ==================== TAB 1: IMAGES ARE LIKE MUSIC ====================
    with tab1:
        st.header("🎵 Images are Like Music")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Think About Music 🎶
            
            When you listen to a song, it has:
            - **Low frequencies** = Bass (🎸 deep sounds)
            - **Mid frequencies** = Vocals (🎤 main melody)
            - **High frequencies** = Cymbals (🎯 sharp sounds)
            
            Each contributes to the complete sound!
            """)
            
            st.image("https://i.imgur.com/S3mJw3e.png", 
                    caption="Different frequencies in music", use_container_width=True)
        
        with col2:
            st.markdown("""
            ### Images Work the Same Way! 🖼️
            
            Images also have frequencies:
            - **Low frequencies** = Large shapes, smooth areas
            - **Mid frequencies** = Medium details, textures
            - **High frequencies** = Fine details, sharp edges
            
            Every image = mix of these "visual frequencies"!
            """)
            
            st.image("https://i.imgur.com/Q4mJk9B.png", 
                    caption="Different frequencies in images", use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🎯 The Big Idea:
        
        **Instead of editing individual pixels (like in Chapter 3)...**
        
        **We can edit the frequencies!**
        
        - Want to blur? → Reduce **high** frequencies
        - Want to sharpen? → Boost **high** frequencies
        - Want to remove patterns? → Remove **specific** frequencies
        
        It's like using an **equalizer for images**! 🎛️
        """)
        
        # Quick demo
        st.info("""
        **Try this thought experiment:**
        
        1. Close your eyes and imagine a **smooth beach** → Mostly **low frequencies**
        2. Now imagine **detailed grass blades** → Lots of **high frequencies**
        3. Open your eyes and look at any image → See the mix of frequencies!
        """)
    
    # ==================== TAB 2: LOW VS HIGH FREQUENCIES ====================
    with tab2:
        st.header("🔍 Low vs High Frequencies")
        
        st.markdown("### What do frequencies actually look like?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🌊 Low Frequencies
            
            **What they represent:**
            - Slow changes across the image
            - Smooth gradients
            - Large objects/shapes
            - Overall brightness
            
            **Examples:**
            - Sky
            - Blurry backgrounds
            - Gradual shadows
            - Smooth skin tones
            
            **Visual effect:**
            - Gentle waves
            - Soft transitions
            - Blurry areas
            """)
            
            # Generate low frequency pattern
            x = np.linspace(0, 10, 100)
            y = np.linspace(0, 10, 100)
            X, Y = np.meshgrid(x, y)
            low_freq = 128 + 50 * np.sin(0.3*X) + 50 * np.sin(0.3*Y)
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(low_freq, cmap='gray', extent=[0, 10, 0, 10])
            ax.set_title("Low Frequency Pattern")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### 🎯 High Frequencies
            
            **What they represent:**
            - Rapid changes
            - Sharp edges
            - Fine details
            - Textures
            - Noise
            
            **Examples:**
            - Text
            - Hair strands
            - Fabric weave
            - Camera grain
            
            **Visual effect:**
            - Sharp transitions
            - Clear boundaries
            - Detailed patterns
            """)
            
            # Generate high frequency pattern
            high_freq = 128 + 50 * np.sin(5*X) + 50 * np.sin(5*Y)
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(high_freq, cmap='gray', extent=[0, 10, 0, 10])
            ax.set_title("High Frequency Pattern")
            ax.axis('off')
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.markdown("### 🧪 Interactive Experiment")
        
        # Create a simple image with adjustable frequencies
        freq_slider = st.slider("Adjust frequency level:", 0.1, 5.0, 1.0, 0.1)
        
        # Generate pattern based on slider
        demo_freq = 128 + 50 * np.sin(freq_slider*X) + 50 * np.sin(freq_slider*Y)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(demo_freq, cmap='gray')
            ax.set_title(f"Frequency = {freq_slider:.1f}")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            if freq_slider < 1.0:
                st.success("**Low Frequency** - Smooth, gradual changes")
                st.metric("Type", "Mostly LOW frequencies", delta="Slow changes")
            elif freq_slider < 2.0:
                st.info("**Medium Frequency** - Some detail visible")
                st.metric("Type", "Mixed frequencies", delta="Balanced")
            else:
                st.warning("**High Frequency** - Lots of fine detail")
                st.metric("Type", "Mostly HIGH frequencies", delta="Rapid changes")
            
            st.progress(freq_slider / 5.0)
            st.caption(f"Frequency intensity: {freq_slider:.1f}/5.0")
    
    # ==================== TAB 3: SEE FREQUENCIES LIVE ====================
    with tab3:
        st.header("🔬 See Frequencies in Real Images")
        
        st.markdown("""
        ### Let's analyze our image's frequencies!
        
        We'll use the **Fourier Transform** - it's like putting on **frequency glasses** 👓
        """)
        
        # Calculate FFT
        fft_result = fft2(img_np)
        fft_shifted = fftshift(fft_result)
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log(magnitude + 1)  # Log scale for better visualization
        
        # Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👁️ Spatial Domain (What We See)")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img_np, cmap='gray')
            ax.set_title("Original Image")
            ax.axis('off')
            st.pyplot(fig)
            
            # Simple frequency indicators
            edges = cv2.Canny(img_np.astype(np.uint8), 100, 200)
            edge_ratio = np.sum(edges > 0) / edges.size
            
            if edge_ratio < 0.05:
                st.info("**Mostly low frequencies** - smooth image")
            elif edge_ratio < 0.2:
                st.info("**Mixed frequencies** - moderate detail")
            else:
                st.info("**Many high frequencies** - detailed image")
        
        with col2:
            st.markdown("#### 🎵 Frequency Domain (What the FT Sees)")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(magnitude_log, cmap='gray')
            ax.set_title("Frequency Spectrum")
            ax.axis('off')
            st.pyplot(fig)
            
            # Analyze spectrum
            center_y, center_x = magnitude.shape[0]//2, magnitude.shape[1]//2
            center_brightness = magnitude[center_y-10:center_y+10, center_x-10:center_x+10].mean()
            edge_brightness = magnitude[0:20, 0:20].mean()
            
            if center_brightness > edge_brightness * 10:
                st.success("**Dominant low frequencies** - center is bright")
            else:
                st.warning("**Significant high frequencies** - edges are bright")
        
        st.markdown("---")
        
        st.markdown("### 🧭 How to Read the Frequency Spectrum:")
        
        guide_cols = st.columns(3)
        
        with guide_cols[0]:
            st.markdown("**🎯 Center**")
            st.markdown("""
            - **Bright spot** = average brightness
            - **Bright area** = low frequencies
            - Represents smooth areas
            """)
        
        with guide_cols[1]:
            st.markdown("**📏 Middle Ring**")
            st.markdown("""
            - **Medium brightness** = medium frequencies
            - Represents textures
            - Objects of moderate size
            """)
        
        with guide_cols[2]:
            st.markdown("**🎪 Outer Edges**")
            st.markdown("""
            - **Bright spots** = high frequencies
            - Represents edges/details
            - Also shows noise/patterns
            """)
        
        # Interactive exploration
        st.markdown("---")
        st.markdown("### 🖱️ Explore Frequency Regions")
        
        click_option = st.radio(
            "Click on the spectrum to see what it represents:",
            ["Click to show frequency type", "Click to apply filter preview"]
        )
        
        # Create a clickable visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(magnitude_log, cmap='hot')
        ax.set_title("Click anywhere on this spectrum!")
        ax.axis('off')
        
        # Add overlay circles
        circle1 = plt.Circle((center_x, center_y), 30, color='blue', fill=False, linewidth=2)
        circle2 = plt.Circle((center_x, center_y), 80, color='green', fill=False, linewidth=2)
        circle3 = plt.Circle((center_x, center_y), 150, color='red', fill=False, linewidth=2)
        
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        
        st.pyplot(fig)
        
        st.caption("Blue circle = Low frequencies | Green = Medium | Red = High")
    
    # ==================== TAB 4: FILTER DEMO ====================
    with tab4:
        st.header("🎛️ Simple Frequency Filter Demo")
        
        st.markdown("""
        ### Let's apply some basic filters!
        
        Remember: Filters in frequency domain = turning frequencies ON/OFF
        """)
        
        filter_type = st.selectbox(
            "Choose a filter type:",
            ["Low-pass (Blur)", "High-pass (Sharpen)", "Band-pass", "Custom"]
        )
        
        if filter_type != "Custom":
            cutoff = st.slider(f"Cutoff frequency:", 0.01, 0.5, 0.2, 0.01)
        
        # Create filter
        rows, cols = img_np.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create circular mask
        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
        max_dist = np.sqrt(crow**2 + ccol**2)
        normalized_dist = dist_from_center / max_dist
        
        if filter_type == "Low-pass (Blur)":
            filter_mask = np.ones((rows, cols))
            filter_mask[normalized_dist > cutoff] = 0
            description = "Allows LOW frequencies, blocks HIGH frequencies → Smooths/blurs image"
        
        elif filter_type == "High-pass (Sharpen)":
            filter_mask = np.zeros((rows, cols))
            filter_mask[normalized_dist > cutoff] = 1
            description = "Allows HIGH frequencies, blocks LOW frequencies → Sharpens edges"
        
        elif filter_type == "Band-pass":
            cutoff2 = st.slider("Second cutoff:", cutoff + 0.01, 0.5, cutoff + 0.1, 0.01)
            filter_mask = np.zeros((rows, cols))
            filter_mask[(normalized_dist > cutoff) & (normalized_dist < cutoff2)] = 1
            description = "Allows MIDDLE frequencies, blocks LOW and HIGH → Isolates textures"
        
        else:  # Custom
            st.markdown("**Design your own filter:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                low_pass = st.checkbox("Allow low frequencies", True)
            with col2:
                medium_pass = st.checkbox("Allow medium frequencies", True)
            with col3:
                high_pass = st.checkbox("Allow high frequencies", True)
            
            filter_mask = np.zeros((rows, cols))
            if low_pass:
                filter_mask[normalized_dist < 0.2] = 1
            if medium_pass:
                filter_mask[(normalized_dist >= 0.2) & (normalized_dist < 0.4)] = 1
            if high_pass:
                filter_mask[normalized_dist >= 0.4] = 1
            
            description = "Custom frequency selection"
        
        # Apply filter
        fft_result = fft2(img_np)
        fft_shifted = fftshift(fft_result)
        
        filtered_fft = fft_shifted * filter_mask
        filtered_fft_ishift = ifftshift(filtered_fft)
        filtered_img = np.real(ifft2(filtered_fft_ishift))
        filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
        
        # Display results
        st.markdown(f"**Filter effect:** {description}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Original")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Filter Mask")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(filter_mask, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
        
        with col3:
            st.markdown("#### Filtered Result")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(filtered_img, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
        
        # Statistics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            original_std = img_np.std()
            st.metric("Original Detail", f"{original_std:.1f}")
        
        with col2:
            filtered_std = filtered_img.std()
            st.metric("Filtered Detail", f"{filtered_std:.1f}")
        
        with col3:
            change_pct = (filtered_std - original_std) / original_std * 100
            st.metric("Change", f"{change_pct:.1f}%")
            
            if change_pct < -10:
                st.error("Significant detail lost → Image blurred")
            elif change_pct > 10:
                st.success("Detail increased → Image sharpened")
            else:
                st.info("Minimal change → Balanced filtering")
    
    # ==================== TAB 5: PRACTICAL EXERCISE ====================
    with tab5:
        st.header("🧪 Hands-On Exercise")
        
        st.markdown("""
        ### Test Your Understanding!
        
        **Exercise:** Analyze different images and identify their frequency content
        """)
        
        if len(available_images) >= 3:
            sample_images = available_images[:3]
            
            st.markdown("### Image Analysis Challenge:")
            
            for idx, img_name in enumerate(sample_images, 1):
                st.markdown(f"#### Image {idx}: `{img_name}`")
                
                # Load image
                img_path = os.path.join(image_dir, img_name)
                test_img = Image.open(img_path).convert('L')
                test_np = np.array(test_img).astype(float)
                
                # Calculate FFT
                test_fft = fft2(test_np)
                test_fft_shifted = fftshift(test_fft)
                test_magnitude = np.abs(test_fft_shifted)
                test_magnitude_log = np.log(test_magnitude + 1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(test_np, cmap='gray')
                    ax.set_title("Image")
                    ax.axis('off')
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(test_magnitude_log, cmap='gray')
                    ax.set_title("Frequency Spectrum")
                    ax.axis('off')
                    st.pyplot(fig)
                
                # Questions
                st.markdown("**Questions:**")
                
                q1 = st.radio(
                    f"What's the main frequency content of Image {idx}?",
                    ["Mostly LOW frequencies (smooth)", 
                     "Mixed LOW and HIGH frequencies", 
                     "Mostly HIGH frequencies (detailed)"],
                    key=f"q1_{idx}"
                )
                
                q2 = st.multiselect(
                    f"What kind of filter would help Image {idx}?",
                    ["Low-pass filter (to smooth noise)", 
                     "High-pass filter (to sharpen edges)", 
                     "Band-pass filter (to isolate textures)",
                     "Band-stop filter (to remove patterns)"],
                    key=f"q2_{idx}"
                )
                
                if st.button(f"Check Analysis for Image {idx}", key=f"check_{idx}"):
                    # Simple analysis
                    center_y, center_x = test_magnitude.shape[0]//2, test_magnitude.shape[1]//2
                    center_region = test_magnitude[center_y-20:center_y+20, center_x-20:center_x+20].mean()
                    edge_region = test_magnitude[0:40, 0:40].mean()
                    
                    ratio = center_region / (edge_region + 1e-6)
                    
                    if ratio > 5:
                        answer = "Mostly LOW frequencies (smooth)"
                    elif ratio > 1:
                        answer = "Mixed LOW and HIGH frequencies"
                    else:
                        answer = "Mostly HIGH frequencies (detailed)"
                    
                    if q1 == answer:
                        st.success(f"✅ Correct! This image has {answer.lower()}")
                    else:
                        st.warning(f"⚠️ Close! This image actually has {answer.lower()}")
                
                st.markdown("---")
        
        else:
            st.info("""
            **For a complete exercise, add at least 3 images to the `public/lab5` folder:**
            1. A smooth/blurry image (e.g., sky)
            2. A detailed image (e.g., text or fabric)
            3. An image with patterns (e.g., stripes or checks)
            """)
        
        st.markdown("""
        ### 📝 Summary of What You've Learned:
        
        1. **Images have frequencies** - just like music
        2. **Low frequencies** = smooth areas, large shapes
        3. **High frequencies** = sharp edges, fine details
        4. **Fourier Transform** shows us the frequency content
        5. **Filters** let us manipulate frequencies
        6. **Low-pass** = blur, **High-pass** = sharpen
        
        **Next lesson:** We'll dive deeper into the Fourier Transform! 🚀
        """)

def create_demo_image():
    """Create a synthetic image for demonstration"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add low frequency background
    for i in range(size):
        for j in range(size):
            img[i, j] = 128 + 30 * np.sin(0.05 * i) + 30 * np.sin(0.05 * j)
    
    # Add medium frequency pattern
    for i in range(100, 200):
        for j in range(100, 200):
            img[i, j] = 128 + 50 * np.sin(0.1 * i) * np.sin(0.1 * j)
    
    # Add high frequency noise/text
    img[50:70, 50:150] = 255  # White bar
    img[180:200, 180:250] = 50  # Dark bar
    
    return Image.fromarray(img.astype(np.uint8))

if __name__ == "__main__":
    app()