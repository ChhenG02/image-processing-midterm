import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import os

def app():
    st.title("🔍 Spectrum Analysis: Reading the Frequency Story")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Spectrum Patterns", 
        "2️⃣ Noise Detection", 
        "3️⃣ Feature Analysis",
        "4️⃣ Practical Diagnosis"
    ])
    
    # Load images
    image_dir = "public/lab5"
    available_images = []
    
    if os.path.exists(image_dir):
        available_images = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if available_images:
        selected_image = st.sidebar.selectbox("📷 Choose Image", available_images, key="spectrum_img")
        img_path = os.path.join(image_dir, selected_image)
        img = Image.open(img_path).convert('L')
    else:
        img = create_demo_image(include_noise=True)
    
    img_np = np.array(img).astype(float)
    
    # Calculate spectrum
    fft_2d = fft2(img_np)
    fft_shifted = fftshift(fft_2d)
    magnitude = np.abs(fft_shifted)
    magnitude_log = np.log(magnitude + 1)
    
    # ==================== TAB 1: SPECTRUM PATTERNS ====================
    with tab1:
        st.header("📊 Understanding Spectrum Patterns")
        
        st.markdown("""
        The frequency spectrum tells a story about your image. Let's learn to read it!
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖼️ Original Image")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img_np, cmap='gray')
            ax.set_title(f"{selected_image if available_images else 'Demo Image'}")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 🎵 Frequency Spectrum")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(magnitude_log, cmap='gray')
            ax.set_title("Magnitude Spectrum")
            ax.axis('off')
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.markdown("### 🧭 Spectrum Regions Guide")
        
        # Create annotated spectrum
        rows, cols = magnitude_log.shape
        center_y, center_x = rows // 2, cols // 2
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(magnitude_log, cmap='gray')
        
        # Draw circles for different frequency regions
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        labels = ['Center (DC)', 'Very Low', 'Low', 'Medium', 'High']
        radii = [5, 30, 80, 150, 220]
        
        for color, label, radius in zip(colors, labels, radii):
            circle = plt.Circle((center_x, center_y), radius, color=color, 
                              fill=False, linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            ax.text(center_x + radius + 5, center_y, label, 
                   color=color, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.axis('off')
        ax.set_title("Frequency Regions (Color-coded)")
        st.pyplot(fig)
        
        # Region explanations
        st.markdown("#### 📍 What Each Region Means:")
        
        regions_data = {
            "🔴 Center (DC Component)": {
                "description": "Average brightness of entire image",
                "visual_meaning": "Overall light/dark level",
                "if_bright": "Image has high average brightness",
                "if_dark": "Image is generally dark",
                "example": "Bright sky or dark night scene"
            },
            "🟠 Very Low Frequencies": {
                "description": "Slow, gradual changes across image",
                "visual_meaning": "Large smooth areas, gradients",
                "if_bright": "Many smooth transitions",
                "if_dark": "Flat, uniform areas",
                "example": "Sunset gradient, blurry background"
            },
            "🟡 Low Frequencies": {
                "description": "Moderate-sized objects and shapes",
                "visual_meaning": "Main objects, coarse textures",
                "if_bright": "Prominent objects present",
                "if_dark": "Few distinct objects",
                "example": "Buildings, large faces"
            },
            "🟢 Medium Frequencies": {
                "description": "Fine details and textures",
                "visual_meaning": "Patterns, medium textures",
                "if_bright": "Rich textures, patterns",
                "if_dark": "Smooth surfaces",
                "example": "Fabric weave, wall texture"
            },
            "🔵 High Frequencies": {
                "description": "Sharp edges, fine details, noise",
                "visual_meaning": "Edges, fine lines, grain",
                "if_bright": "Sharp edges or noise present",
                "if_dark": "Blurry or smooth",
                "example": "Text, hair, camera noise"
            }
        }
        
        for region, info in regions_data.items():
            with st.expander(f"{region}"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Visual Meaning:** {info['visual_meaning']}")
                st.markdown(f"**If Bright:** {info['if_bright']}")
                st.markdown(f"**If Dark:** {info['if_dark']}")
                st.markdown(f"**Example:** {info['example']}")
        
        st.markdown("---")
        
        # Interactive region analysis
        st.markdown("### 🔍 Interactive Region Analysis")
        
        region_select = st.selectbox(
            "Select a region to analyze:",
            ["Center (DC)", "Very Low", "Low", "Medium", "High", "All Regions"]
        )
        
        # Create mask for selected region
        mask = np.zeros_like(magnitude_log)
        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        if region_select == "Center (DC)":
            mask[dist_from_center < 10] = 1
        elif region_select == "Very Low":
            mask[(dist_from_center >= 10) & (dist_from_center < 50)] = 1
        elif region_select == "Low":
            mask[(dist_from_center >= 50) & (dist_from_center < 100)] = 1
        elif region_select == "Medium":
            mask[(dist_from_center >= 100) & (dist_from_center < 180)] = 1
        elif region_select == "High":
            mask[dist_from_center >= 180] = 1
        else:
            mask = np.ones_like(magnitude_log)
        
        # Calculate statistics
        region_magnitude = magnitude_log[mask == 1]
        if len(region_magnitude) > 0:
            mean_strength = region_magnitude.mean()
            max_strength = region_magnitude.max()
            percent_of_total = region_magnitude.sum() / magnitude_log.sum() * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Strength", f"{mean_strength:.1f}")
            with col2:
                st.metric("Peak Strength", f"{max_strength:.1f}")
            with col3:
                st.metric("% of Total", f"{percent_of_total:.1f}%")
            
            if percent_of_total > 30:
                st.success(f"✅ **{region_select} frequencies dominate** this image")
            elif percent_of_total > 10:
                st.info(f"📊 **{region_select} frequencies are significant**")
            else:
                st.warning(f"⚡ **{region_select} frequencies are minor**")
    
    # ==================== TAB 2: NOISE DETECTION ====================
    with tab2:
        st.header("🚫 Detecting Noise in the Spectrum")
        
        st.markdown("""
        ### How to Spot Noise vs Real Features
        
        Noise creates specific patterns in the frequency spectrum.
        Let's learn to identify them!
        """)
        
        # Create images with different types of noise
        st.markdown("### 🧪 Noise Type Examples")
        
        noise_type = st.selectbox(
            "Select noise type to visualize:",
            ["Clean Image", "Gaussian Noise", "Salt & Pepper", "Periodic Stripe", "Checkerboard"]
        )
        
        # Create or modify image based on noise type
        if noise_type == "Clean Image":
            demo_img = img_np.copy()
            noise_desc = "Original image without added noise"
        elif noise_type == "Gaussian Noise":
            noise = np.random.normal(0, 30, img_np.shape)
            demo_img = np.clip(img_np + noise, 0, 255)
            noise_desc = "Random grain-like noise (common in low light)"
        elif noise_type == "Salt & Pepper":
            demo_img = img_np.copy()
            salt_pepper = np.random.random(img_np.shape)
            demo_img[salt_pepper < 0.01] = 0  # Pepper
            demo_img[salt_pepper > 0.99] = 255  # Salt
            noise_desc = "Random black and white pixels"
        elif noise_type == "Periodic Stripe":
            demo_img = img_np.copy()
            for i in range(img_np.shape[0]):
                if i % 20 < 10:
                    demo_img[i, :] = np.clip(demo_img[i, :] + 50, 0, 255)
            noise_desc = "Regular stripe pattern (scanning artifact)"
        else:  # Checkerboard
            demo_img = img_np.copy()
            for i in range(img_np.shape[0]):
                for j in range(img_np.shape[1]):
                    if (i // 10) % 2 == (j // 10) % 2:
                        demo_img[i, j] = np.clip(demo_img[i, j] + 30, 0, 255)
            noise_desc = "Regular grid pattern (sensor artifact)"
        
        # Calculate spectrum for noisy image
        fft_noisy = fft2(demo_img)
        fft_noisy_shifted = fftshift(fft_noisy)
        magnitude_noisy = np.abs(fft_noisy_shifted)
        magnitude_noisy_log = np.log(magnitude_noisy + 1)
        
        # Display comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Image with Noise")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(demo_img, cmap='gray')
            ax.set_title(f"{noise_type}")
            ax.axis('off')
            st.pyplot(fig)
            st.caption(noise_desc)
        
        with col2:
            st.markdown("#### Spectrum")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(magnitude_noisy_log, cmap='gray')
            ax.set_title("Frequency Spectrum")
            ax.axis('off')
            st.pyplot(fig)
        
        with col3:
            st.markdown("#### Difference from Clean")
            # Calculate difference spectrum
            if noise_type != "Clean Image":
                clean_fft = fft2(img_np)
                clean_fft_shifted = fftshift(clean_fft)
                clean_magnitude = np.abs(clean_fft_shifted)
                clean_magnitude_log = np.log(clean_magnitude + 1)
                
                diff = magnitude_noisy_log - clean_magnitude_log
                
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(diff, cmap='coolwarm', vmin=-2, vmax=2)
                ax.set_title("Difference (Noise Only)")
                ax.axis('off')
                st.pyplot(fig)
                
                plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
            
            else:
                st.info("Clean image - no difference to show")
        
        st.markdown("---")
        
        # Noise detection guide
        st.markdown("### 🔎 Noise Detection Guide")
        
        detection_cols = st.columns(2)
        
        with detection_cols[0]:
            st.markdown("#### ✅ Real Features:")
            st.markdown("""
            - **Smooth gradients** from center
            - **Natural patterns** following image structure
            - **Symmetrical but organic** shapes
            - **Connected** to image content
            
            **Examples:**
            - Radial patterns from circular objects
            - Lines matching image edges
            - Gradual brightness changes
            """)
        
        with detection_cols[1]:
            st.markdown("#### ❌ Noise Patterns:")
            st.markdown("""
            - **Isolated bright spots** away from center
            - **Perfect symmetry** unrelated to content
            - **Sharp spikes** at specific frequencies
            - **Grid patterns** (horizontal/vertical lines)
            
            **Examples:**
            - Single bright dots in corners
            - Symmetrical pairs unrelated to image
            - Straight lines at specific angles
            """)
        
        # Interactive noise identification
        st.markdown("---")
        st.markdown("### 🎯 Practice: Find the Noise")
        
        # Create an image with hidden noise
        practice_img = img_np.copy()
        
        # Add some hidden periodic noise
        rows, cols = practice_img.shape
        for i in range(rows):
            for j in range(cols):
                if (i + j) % 15 == 0:
                    practice_img[i, j] = np.clip(practice_img[i, j] + 30, 0, 255)
        
        # Calculate spectrum
        fft_practice = fft2(practice_img)
        fft_practice_shifted = fftshift(fft_practice)
        magnitude_practice = np.abs(fft_practice_shifted)
        magnitude_practice_log = np.log(magnitude_practice + 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Practice Image")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(practice_img, cmap='gray')
            ax.set_title("Can you see the noise?")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Spectrum Analysis")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(magnitude_practice_log, cmap='gray')
            ax.set_title("Look for patterns!")
            ax.axis('off')
            st.pyplot(fig)
        
        # Quiz
        st.markdown("#### 🧠 Test Your Knowledge")
        
        answer = st.radio(
            "What type of noise is in this image?",
            ["Gaussian noise (random grain)", 
             "Salt & pepper noise (random dots)", 
             "Periodic noise (repeating pattern)",
             "No noise present"]
        )
        
        if st.button("Check Answer"):
            if answer == "Periodic noise (repeating pattern)":
                st.success("✅ Correct! This image has periodic noise (diagonal pattern every 15 pixels)")
                st.markdown("**In the spectrum:** Look for the diagonal line pattern!")
            else:
                st.error("❌ Try again! Look for repeating patterns in the spectrum")
    
    # ==================== TAB 3: FEATURE ANALYSIS ====================
    with tab3:
        st.header("🔍 Analyzing Image Features in Spectrum")
        
        st.markdown("""
        Different image features create distinct patterns in the frequency spectrum.
        Let's analyze them!
        """)
        
        # Create different feature types
        feature_type = st.selectbox(
            "Select feature type to analyze:",
            ["Edges", "Textures", "Patterns", "Shapes", "All Features"]
        )
        
        # Create feature images
        size = 256
        if feature_type == "Edges":
            feature_img = create_edge_image(size)
            feature_desc = "Sharp transitions between regions"
        elif feature_type == "Textures":
            feature_img = create_texture_image(size)
            feature_desc = "Repeating patterns with fine details"
        elif feature_type == "Patterns":
            feature_img = create_pattern_image(size)
            feature_desc = "Regular, repeating structures"
        elif feature_type == "Shapes":
            feature_img = create_shape_image(size)
            feature_desc = "Geometric forms and contours"
        else:
            feature_img = img_np
            feature_desc = "Original image features"
        
        # Calculate spectrum
        fft_feature = fft2(feature_img)
        fft_feature_shifted = fftshift(fft_feature)
        magnitude_feature = np.abs(fft_feature_shifted)
        magnitude_feature_log = np.log(magnitude_feature + 1)
        
        # Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {feature_type}")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(feature_img, cmap='gray')
            ax.set_title(feature_desc)
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Spectrum Signature")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(magnitude_feature_log, cmap='gray')
            ax.set_title("Frequency Pattern")
            ax.axis('off')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Feature spectrum analysis
        st.markdown(f"### 📊 {feature_type} Spectrum Analysis")
        
        if feature_type == "Edges":
            st.info("""
            **Edge Spectrum Signature:**
            - **Bright lines** radiating from center
            - **Direction** matches edge orientation
            - **Strength** proportional to edge sharpness
            - More edges = more bright lines
            
            **Example:** Horizontal edges → vertical lines in spectrum
            """)
        
        elif feature_type == "Textures":
            st.info("""
            **Texture Spectrum Signature:**
            - **Circular patterns** at specific radii
            - **Ring shapes** indicating frequency range
            - **Uniform distribution** in certain directions
            - Dense textures = dense frequency patterns
            
            **Example:** Fine fabric → outer ring in spectrum
            """)
        
        elif feature_type == "Patterns":
            st.info("""
            **Pattern Spectrum Signature:**
            - **Discrete bright spots** at regular intervals
            - **Symmetrical arrangement**
            - **Grid-like patterns** for regular patterns
            - Clear periodic structure
            
            **Example:** Checkerboard → grid of dots in spectrum
            """)
        
        elif feature_type == "Shapes":
            st.info("""
            **Shape Spectrum Signature:**
            - **Smooth gradients** from center
            - **Radial patterns** for circular shapes
            - **Directional patterns** for angular shapes
            - Low to medium frequency dominance
            
            **Example:** Circle → circular patterns in spectrum
            """)
        
        # Feature detection exercise
        st.markdown("---")
        st.markdown("### 🎯 Feature Detection Challenge")
        
        # Create mixed features image
        challenge_img = create_challenge_image(size)
        
        # Calculate spectrum
        fft_challenge = fft2(challenge_img)
        fft_challenge_shifted = fftshift(fft_challenge)
        magnitude_challenge = np.abs(fft_challenge_shifted)
        magnitude_challenge_log = np.log(magnitude_challenge + 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Challenge Image")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(challenge_img, cmap='gray')
            ax.set_title("What features do you see?")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Spectrum")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(magnitude_challenge_log, cmap='gray')
            ax.set_title("Identify from spectrum!")
            ax.axis('off')
            st.pyplot(fig)
        
        # Feature identification quiz
        st.markdown("#### Identify the features:")
        
        features_present = st.multiselect(
            "Select all features present in the image:",
            ["Horizontal edges", "Vertical edges", "Diagonal edges", 
             "Fine textures", "Regular patterns", "Circular shapes",
             "Noise", "Smooth gradients"]
        )
        
        if st.button("Check Feature Analysis"):
            correct = {"Horizontal edges", "Vertical edges", "Circular shapes"}
            selected = set(features_present)
            
            if selected == correct:
                st.success("✅ Perfect! You correctly identified all features!")
                st.markdown("""
                **Analysis:**
                - Bright horizontal line in spectrum → vertical edges in image
                - Bright vertical line in spectrum → horizontal edges in image  
                - Circular pattern → circular shape in image
                """)
            else:
                st.warning("⚠️ Close! Try analyzing the spectrum patterns more carefully")
                st.markdown("**Hint:** Look for bright lines and circular patterns in the spectrum")
    
    # ==================== TAB 4: PRACTICAL DIAGNOSIS ====================
    with tab4:
        st.header("🏥 Practical Image Diagnosis")
        
        st.markdown("""
        Let's apply everything we've learned to diagnose real images!
        """)
        
        if not available_images:
            st.warning("Add images to `public/lab5` for full diagnosis practice")
            st.markdown("Using demo images for practice...")
            
            # Create demo cases
            cases = [
                ("Case 1: Blurry Photo", create_blurry_image(), "Why is this image blurry?"),
                ("Case 2: Noisy Image", create_noisy_image(), "What type of noise is present?"),
                ("Case 3: Patterned Object", create_patterned_image(), "What patterns are visible?"),
                ("Case 4: Sharp Details", create_detailed_image(), "Why is this image sharp?")
            ]
        else:
            # Use actual images
            cases = []
            for i, img_name in enumerate(available_images[:4], 1):
                img_path = os.path.join(image_dir, img_name)
                case_img = Image.open(img_path).convert('L')
                case_np = np.array(case_img).astype(float)
                cases.append((f"Case {i}: {img_name}", case_np, f"Analyze image: {img_name}"))
        
        # Display each case
        for case_title, case_img, case_question in cases:
            st.markdown(f"### {case_title}")
            
            # Calculate spectrum
            fft_case = fft2(case_img)
            fft_case_shifted = fftshift(fft_case)
            magnitude_case = np.abs(fft_case_shifted)
            magnitude_case_log = np.log(magnitude_case + 1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(case_img, cmap='gray')
                ax.set_title("Image")
                ax.axis('off')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(magnitude_case_log, cmap='gray')
                ax.set_title("Spectrum")
                ax.axis('off')
                st.pyplot(fig)
            
            # Diagnosis questions
            st.markdown(f"**{case_question}**")
            
            diagnosis = st.text_area(
                f"Diagnosis for {case_title}:",
                placeholder="Describe what you see in the spectrum and what it means for the image...",
                key=f"diagnosis_{case_title}"
            )
            
            if st.button(f"Get Analysis Help for {case_title}", key=f"help_{case_title}"):
                # Provide analysis based on image characteristics
                center_strength = magnitude_case_log[magnitude_case_log.shape[0]//2, magnitude_case_log.shape[1]//2]
                edge_strength = magnitude_case_log[0, 0]
                
                if center_strength > edge_strength * 2:
                    st.info("💡 **Hint:** Strong low frequencies - image may be smooth/blurry")
                elif edge_strength > center_strength:
                    st.info("💡 **Hint:** Strong high frequencies - image may be sharp/noisy")
                
                # Check for patterns
                std_dev = magnitude_case_log.std()
                if std_dev > 1.5:
                    st.info("💡 **Hint:** High contrast in spectrum - likely has patterns/edges")
                
            st.markdown("---")
        
        st.success("""
        ### 🎓 You've Learned Spectrum Analysis!
        
        **Key Skills:**
        1. **Read frequency spectra** like a map
        2. **Identify noise patterns** vs real features
        3. **Recognize feature signatures** in spectrum
        4. **Diagnose image issues** from frequency patterns
        
        **Next:** Apply this knowledge with frequency filters! 🚀
        """)

# Helper functions
def create_demo_image(include_noise=False):
    """Create a synthetic demo image"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Gradient background
    for i in range(size):
        for j in range(size):
            img[i, j] = 128 + 50 * np.sin(0.03 * i) * np.cos(0.03 * j)
    
    # Add shapes
    cv2.rectangle(img, (50, 50), (100, 100), 200, -1)
    cv2.circle(img, (180, 180), 40, 150, -1)
    
    # Add text/patterns
    for i in range(0, size, 30):
        img[i:i+5, :] = 255  # Horizontal lines
        img[:, i:i+5] = 100  # Vertical lines
    
    if include_noise:
        noise = np.random.normal(0, 20, img.shape)
        img = np.clip(img + noise, 0, 255)
    
    return Image.fromarray(img.astype(np.uint8))

def create_edge_image(size):
    """Create image with clear edges"""
    img = np.zeros((size, size), dtype=np.uint8)
    img[:size//2, :] = 100  # Top half
    img[size//2:, :] = 200  # Bottom half
    cv2.rectangle(img, (50, 50), (200, 200), 255, 10)  # Square
    return img

def create_texture_image(size):
    """Create textured image"""
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            img[i, j] = 128 + 50 * np.sin(0.1 * i) * np.sin(0.1 * j)
    return img

def create_pattern_image(size):
    """Create patterned image"""
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i // 20) % 2 == (j // 20) % 2:
                img[i, j] = 255
            else:
                img[i, j] = 50
    return img

def create_shape_image(size):
    """Create image with shapes"""
    img = np.zeros((size, size), dtype=np.uint8)
    img.fill(100)
    cv2.circle(img, (size//2, size//2), 80, 200, -1)
    cv2.rectangle(img, (50, 150), (150, 250), 255, -1)
    return img

def create_challenge_image(size):
    """Create image with multiple features"""
    img = np.zeros((size, size), dtype=np.uint8)
    img.fill(150)
    
    # Horizontal line
    img[size//2-5:size//2+5, :] = 50
    
    # Vertical line
    img[:, size//2-5:size//2+5] = 50
    
    # Circle
    cv2.circle(img, (size//4, 3*size//4), 40, 255, -1)
    
    return img

def create_blurry_image():
    """Create blurry image"""
    img = np.random.rand(256, 256) * 100 + 100
    img = cv2.GaussianBlur(img.astype(np.float32), (21, 21), 10)
    return np.clip(img, 0, 255).astype(np.uint8)

def create_noisy_image():
    """Create noisy image"""
    img = np.zeros((256, 256), dtype=np.uint8)
    img.fill(128)
    noise = np.random.normal(0, 50, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Add periodic noise
    for i in range(256):
        if i % 10 == 0:
            img[i, :] = np.clip(img[i, :] + 30, 0, 255)
    
    return img

def create_patterned_image():
    """Create image with patterns"""
    img = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            if (i // 25) % 2 == (j // 25) % 2:
                img[i, j] = 200
            else:
                img[i, j] = 50
    return img

def create_detailed_image():
    """Create detailed image"""
    img = np.zeros((256, 256), dtype=np.uint8)
    
    # Add lots of small details
    for i in range(0, 256, 5):
        for j in range(0, 256, 5):
            if (i + j) % 10 == 0:
                img[i:i+2, j:j+2] = 255
    
    # Add text-like patterns
    for i in range(50, 200, 20):
        img[i:i+10, 50:200] = 150
    
    return img

if __name__ == "__main__":
    app()