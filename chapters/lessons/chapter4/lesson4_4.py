import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import os

def app():
    st.title("🎛️ Frequency Filters: Image Processing Equalizer")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Filter Types", 
        "2️⃣ Low-Pass Filters", 
        "3️⃣ High-Pass Filters",
        "4️⃣ Band & Notch Filters",
        "5️⃣ Filter Design Lab"
    ])
    
    # Load image
    image_dir = "public/lab5"
    if os.path.exists(image_dir):
        available_images = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if available_images:
            selected_image = st.sidebar.selectbox("📷 Choose Image", available_images, key="filter_img")
            img_path = os.path.join(image_dir, selected_image)
            img = Image.open(img_path).convert('L')
        else:
            img = create_demo_image()
    else:
        img = create_demo_image()
    
    img_np = np.array(img).astype(float)
    
    # ==================== TAB 1: FILTER TYPES ====================
    with tab1:
        st.header("🎚️ Understanding Filter Types")
        
        st.markdown("""
        Think of frequency filters as an **equalizer for images**!
        Just like you boost bass or treble in music, you can adjust different frequencies in images.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎵 Music Equalizer Analogy
            
            **Music Equalizer:**
            - **Bass (low)** → Deep sounds
            - **Mid** → Vocals, instruments  
            - **Treble (high)** → Cymbals, details
            
            **Adjust sliders to:**
            - Boost bass = warmer sound
            - Cut treble = softer sound
            - Boost mid = clearer vocals
            """)
            
            # Create a visual equalizer
            eq_fig, eq_ax = plt.subplots(figsize=(6, 4))
            frequencies = ['Bass', 'Low Mid', 'Mid', 'High Mid', 'Treble']
            levels = [7, 5, 3, 6, 4]
            bars = eq_ax.bar(frequencies, levels, color=['blue', 'cyan', 'green', 'orange', 'red'])
            eq_ax.set_ylabel('Boost/Cut (dB)')
            eq_ax.set_title('Music Equalizer')
            eq_ax.set_ylim(0, 10)
            st.pyplot(eq_fig)
        
        with col2:
            st.markdown("""
            ### 🖼️ Image Frequency Filters
            
            **Image Frequency Bands:**
            - **Low** → Smooth areas, large shapes
            - **Mid** → Textures, patterns
            - **High** → Edges, fine details, noise
            
            **Apply filters to:**
            - Remove high = blur/smooth
            - Boost high = sharpen
            - Remove specific = clean patterns
            """)
            
            # Create filter visualization
            filter_fig, filter_ax = plt.subplots(figsize=(6, 4))
            filter_types = ['Low-pass', 'Band-pass', 'High-pass', 'Notch']
            applications = ['Blurring', 'Texture', 'Sharpening', 'Noise Removal']
            colors = ['blue', 'green', 'red', 'purple']
            filter_ax.barh(filter_types, [8, 6, 7, 5], color=colors)
            for i, (filt, app) in enumerate(zip(filter_types, applications)):
                filter_ax.text(1, i, f' → {app}', va='center')
            filter_ax.set_xlabel('Effect Strength')
            filter_ax.set_title('Image Filters')
            st.pyplot(filter_fig)
        
        st.markdown("---")
        
        st.markdown("### 🔧 Filter Types Overview")
        
        # Filter type explanations
        filter_data = {
            "🔵 Low-pass Filter": {
                "description": "Allows LOW frequencies, blocks HIGH frequencies",
                "effect": "Smoothing, blurring, noise reduction",
                "analogy": "Putting Vaseline on lens",
                "use_cases": ["Remove noise", "Create bokeh", "Smooth skin", "Preprocess for AI"],
                "visual": "Circle at center (keeps center, blocks edges)"
            },
            "🔴 High-pass Filter": {
                "description": "Allows HIGH frequencies, blocks LOW frequencies",
                "effect": "Sharpening, edge enhancement",
                "analogy": "Increasing contrast on edges",
                "use_cases": ["Edge detection", "Image sharpening", "Detail enhancement", "Text recognition"],
                "visual": "Everything except center (blocks center, keeps edges)"
            },
            "🟢 Band-pass Filter": {
                "description": "Allows MIDDLE frequencies, blocks LOW and HIGH",
                "effect": "Texture isolation, pattern extraction",
                "analogy": "Focusing on specific detail level",
                "use_cases": ["Texture analysis", "Feature extraction", "Pattern recognition", "Quality inspection"],
                "visual": "Ring/donut shape (blocks center and edges)"
            },
            "🟣 Band-stop/Notch Filter": {
                "description": "Blocks SPECIFIC frequencies, allows others",
                "effect": "Remove interference, clean patterns",
                "analogy": "Removing specific instrument from song",
                "use_cases": ["Remove moire patterns", "Clean scan lines", "Remove periodic noise", "Fix compression artifacts"],
                "visual": "Black spots on white (blocks specific spots)"
            }
        }
        
        for filter_name, info in filter_data.items():
            with st.expander(f"{filter_name}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown(f"**Effect:** {info['effect']}")
                    st.markdown(f"**Analogy:** {info['analogy']}")
                    st.markdown("**Use Cases:**")
                    for use_case in info['use_cases']:
                        st.markdown(f"- {use_case}")
                
                with col2:
                    # Create simple filter visualization
                    fig, ax = plt.subplots(figsize=(3, 3))
                    size = 100
                    if "Low-pass" in filter_name:
                        mask = create_circular_mask(size, size, 0.3)
                    elif "High-pass" in filter_name:
                        mask = 1 - create_circular_mask(size, size, 0.3)
                    elif "Band-pass" in filter_name:
                        mask = create_band_mask(size, size, 0.2, 0.4)
                    else:
                        mask = create_notch_mask(size, size)
                    
                    ax.imshow(mask, cmap='gray')
                    ax.set_title(info['visual'])
                    ax.axis('off')
                    st.pyplot(fig)
        
        st.markdown("---")
        
        # Interactive filter preview
        st.markdown("### 👁️ Interactive Filter Preview")
        
        filter_preview = st.selectbox(
            "Preview filter shape:",
            ["Low-pass", "High-pass", "Band-pass", "Band-stop", "Notch"]
        )
        
        # Create filter visualization
        preview_size = 200
        if filter_preview == "Low-pass":
            preview_mask = create_circular_mask(preview_size, preview_size, 0.3)
            preview_desc = "Circular area at center is kept (white)"
        elif filter_preview == "High-pass":
            preview_mask = 1 - create_circular_mask(preview_size, preview_size, 0.3)
            preview_desc = "Circular area at center is blocked (black)"
        elif filter_preview == "Band-pass":
            preview_mask = create_band_mask(preview_size, preview_size, 0.2, 0.4)
            preview_desc = "Ring/donut shape is kept (white)"
        elif filter_preview == "Band-stop":
            preview_mask = 1 - create_band_mask(preview_size, preview_size, 0.2, 0.4)
            preview_desc = "Ring/donut shape is blocked (black)"
        else:  # Notch
            preview_mask = create_notch_mask(preview_size, preview_size)
            preview_desc = "Specific spots are blocked (black dots)"
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(preview_mask, cmap='gray')
            ax.set_title(f"{filter_preview} Filter")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.markdown(f"**{filter_preview} Filter:**")
            st.markdown(preview_desc)
            st.markdown("**White areas:** Frequencies KEPT")
            st.markdown("**Black areas:** Frequencies REMOVED")
            
            # Show frequency effect
            st.markdown("**Effect on image:**")
            if "Low" in filter_preview:
                st.info("Keeps smooth areas, removes details → Blurry image")
            elif "High" in filter_preview:
                st.info("Keeps edges/details, removes smooth areas → Sharp image")
            elif "Band" in filter_preview:
                st.info("Keeps specific detail level → Texture isolation")
            else:
                st.info("Removes specific patterns → Clean image")
    
    # ==================== TAB 2: LOW-PASS FILTERS ====================
    with tab2:
        st.header("🔵 Low-Pass Filters: Smoothing & Blurring")
        
        st.markdown("""
        Low-pass filters remove high frequencies, keeping only smooth, gradual changes.
        This creates a blurred or smoothed effect.
        """)
        
        # Low-pass filter controls
        st.markdown("### 🎛️ Low-Pass Filter Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cutoff = st.slider("Cutoff Frequency", 0.01, 0.5, 0.2, 0.01,
                             help="Lower = more blur, Higher = less blur")
        
        with col2:
            filter_shape = st.selectbox("Filter Shape", 
                                       ["Ideal (sharp)", "Gaussian (smooth)", "Butterworth (adjustable)"])
        
        with col3:
            if filter_shape == "Butterworth (adjustable)":
                order = st.slider("Filter Order", 1, 10, 2, 1,
                                help="Higher order = sharper cutoff")
            else:
                order = 2
        
        # Create low-pass filter
        rows, cols = img_np.shape
        center_y, center_x = rows // 2, cols // 2
        
        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist = dist_from_center / max_dist
        
        if filter_shape == "Ideal (sharp)":
            filter_mask = np.zeros((rows, cols))
            filter_mask[normalized_dist <= cutoff] = 1
        
        elif filter_shape == "Gaussian (smooth)":
            filter_mask = np.exp(-(normalized_dist**2) / (2 * (cutoff**2)))
        
        else:  # Butterworth
            filter_mask = 1 / (1 + (normalized_dist / cutoff) ** (2 * order))
        
        # Apply filter
        fft_img = fft2(img_np)
        fft_shifted = fftshift(fft_img)
        
        filtered_fft = fft_shifted * filter_mask
        filtered_fft_ishift = ifftshift(filtered_fft)
        filtered_img = np.real(ifft2(filtered_fft_ishift))
        filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Low-Pass Filter Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Original Image")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np, cmap='gray')
            ax.set_title("Original")
            ax.axis('off')
            st.pyplot(fig)
            
            # Calculate metrics
            orig_std = img_np.std()
            orig_edges = cv2.Canny(img_np.astype(np.uint8), 100, 200)
            edge_count_orig = np.sum(orig_edges > 0)
            
            st.metric("Detail (std)", f"{orig_std:.1f}")
            st.metric("Edge pixels", f"{edge_count_orig}")
        
        with col2:
            st.markdown("#### Filter Mask")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(filter_mask, cmap='gray')
            ax.set_title(f"Low-pass {filter_shape}")
            ax.axis('off')
            st.pyplot(fig)
            
            # Filter stats
            kept_freq = np.sum(filter_mask > 0.5) / filter_mask.size * 100
            st.metric("Frequencies kept", f"{kept_freq:.1f}%")
            st.metric("Cutoff", f"{cutoff:.2f}")
        
        with col3:
            st.markdown("#### Filtered Image")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(filtered_img, cmap='gray')
            ax.set_title("After Low-pass")
            ax.axis('off')
            st.pyplot(fig)
            
            # Calculate metrics
            filtered_std = filtered_img.std()
            filtered_edges = cv2.Canny(filtered_img, 100, 200)
            edge_count_filtered = np.sum(filtered_edges > 0)
            
            detail_loss = (orig_std - filtered_std) / orig_std * 100
            edge_loss = (edge_count_orig - edge_count_filtered) / edge_count_orig * 100
            
            st.metric("Detail (std)", f"{filtered_std:.1f}", delta=f"-{detail_loss:.1f}%")
            st.metric("Edge pixels", f"{edge_count_filtered}", delta=f"-{edge_loss:.1f}%")
        
        st.markdown("---")
        
        # Comparison slider
        st.markdown("### 🔄 Before/After Comparison")
        
        compare_value = st.slider("Comparison Slider", 0, 100, 50, 1,
                                help="Slide to compare original (left) and filtered (right)")
        
        # Create comparison image
        comparison = np.hstack([
            img_np[:, :img_np.shape[1]//2],
            filtered_img[:, filtered_img.shape[1]//2:]
        ])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        split_point = int(comparison.shape[1] * compare_value / 100)
        
        # Create visual divider
        comparison_display = comparison.copy()
        if 0 < split_point < comparison.shape[1]:
            comparison_display[:, split_point-2:split_point+2] = 255
        
        ax.imshow(comparison_display, cmap='gray')
        ax.axvline(x=split_point, color='red', linestyle='--', linewidth=2)
        ax.text(split_point + 10, 30, "FILTERED", color='red', fontweight='bold')
        ax.text(split_point - 80, 30, "ORIGINAL", color='red', fontweight='bold')
        ax.axis('off')
        ax.set_title("Slide to compare: Original ← → Filtered")
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Applications examples
        st.markdown("### 🏥 Practical Applications of Low-Pass Filters")
        
        app_cols = st.columns(3)
        
        with app_cols[0]:
            st.markdown("#### 📸 Photography")
            st.markdown("""
            - **Portrait mode**: Blur background
            - **Skin smoothing**: Reduce pores/wrinkles
            - **Noise reduction**: In low-light photos
            - **Dreamy effects**: Soft focus look
            """)
        
        with app_cols[1]:
            st.markdown("#### 🏥 Medical Imaging")
            st.markdown("""
            - **MRI/CT scans**: Reduce noise
            - **X-rays**: Smooth grain
            - **Ultrasound**: Clean up artifacts
            - **Microscopy**: Reduce speckle noise
            """)
        
        with app_cols[2]:
            st.markdown("#### 🤖 Computer Vision")
            st.markdown("""
            - **Preprocessing**: Before feature detection
            - **Noise removal**: For better AI accuracy
            - **Motion blur**: In video processing
            - **Data augmentation**: Create variations
            """)
        
        # Interactive experiment
        st.markdown("---")
        st.markdown("### 🧪 Interactive Experiment: Blur Level Test")
        
        test_cutoffs = [0.05, 0.1, 0.2, 0.3, 0.4]
        test_cols = st.columns(len(test_cutoffs))
        
        for idx, test_cutoff in enumerate(test_cutoffs):
            with test_cols[idx]:
                # Create test filter
                test_mask = create_circular_mask(rows, cols, test_cutoff)
                test_filtered_fft = fft_shifted * test_mask
                test_filtered = np.real(ifft2(ifftshift(test_filtered_fft)))
                test_filtered = np.clip(test_filtered, 0, 255).astype(np.uint8)
                
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(test_filtered, cmap='gray')
                ax.set_title(f"Cutoff={test_cutoff}")
                ax.axis('off')
                st.pyplot(fig)
                
                # Calculate blur metric
                blur_metric = test_filtered.std() / img_np.std() * 100
                st.caption(f"Detail: {blur_metric:.0f}%")
                
                if blur_metric > 80:
                    st.caption("Slight blur")
                elif blur_metric > 50:
                    st.caption("Moderate blur")
                else:
                    st.caption("Heavy blur")
    
    # ==================== TAB 3: HIGH-PASS FILTERS ====================
    with tab3:
        st.header("🔴 High-Pass Filters: Sharpening & Edge Detection")
        
        st.markdown("""
        High-pass filters remove low frequencies, keeping only rapid changes (edges and details).
        This creates a sharpened effect or helps detect edges.
        """)
        
        # High-pass filter controls
        st.markdown("### 🎛️ High-Pass Filter Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hp_cutoff = st.slider("Cutoff Frequency", 0.01, 0.5, 0.1, 0.01,
                                key="hp_cutoff",
                                help="Lower = keep more details, Higher = keep only strongest edges")
        
        with col2:
            hp_strength = st.slider("Sharpening Strength", 1.0, 5.0, 2.0, 0.5,
                                  help="Boost factor for high frequencies")
        
        # Create high-pass filter
        rows, cols = img_np.shape
        center_y, center_x = rows // 2, cols // 2
        
        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist = dist_from_center / max_dist
        
        # High-pass is inverse of low-pass
        hp_filter_mask = np.ones((rows, cols))
        hp_filter_mask[normalized_dist < hp_cutoff] = 0
        
        # Apply boost
        edge_region = normalized_dist > hp_cutoff
        hp_filter_mask[edge_region] *= hp_strength
        
        # Apply filter
        fft_img = fft2(img_np)
        fft_shifted = fftshift(fft_img)
        
        filtered_fft_hp = fft_shifted * hp_filter_mask
        filtered_fft_ishift_hp = ifftshift(filtered_fft_hp)
        filtered_img_hp = np.real(ifft2(filtered_fft_ishift_hp))
        filtered_img_hp = np.clip(filtered_img_hp, 0, 255).astype(np.uint8)
        
        # Also create edge-only version
        edge_filter_mask = np.zeros((rows, cols))
        edge_filter_mask[normalized_dist > hp_cutoff] = 1
        
        edge_fft = fft_shifted * edge_filter_mask
        edge_img = np.real(ifft2(ifftshift(edge_fft)))
        edge_img = np.abs(edge_img)  # Take absolute for edge display
        edge_img = np.clip(edge_img * 5, 0, 255).astype(np.uint8)  # Boost for visibility
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 High-Pass Filter Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Original Image")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np, cmap='gray')
            ax.set_title("Original")
            ax.axis('off')
            st.pyplot(fig)
            
            orig_detail = img_np.std()
            st.metric("Detail Level", f"{orig_detail:.1f}")
        
        with col2:
            st.markdown("#### Sharpened Image")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(filtered_img_hp, cmap='gray')
            ax.set_title(f"High-pass (×{hp_strength})")
            ax.axis('off')
            st.pyplot(fig)
            
            sharpened_detail = filtered_img_hp.std()
            detail_boost = (sharpened_detail - orig_detail) / orig_detail * 100
            st.metric("Detail Level", f"{sharpened_detail:.1f}", delta=f"+{detail_boost:.1f}%")
        
        with col3:
            st.markdown("#### Edge Detection")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(edge_img, cmap='gray')
            ax.set_title("Edges Only")
            ax.axis('off')
            st.pyplot(fig)
            
            edge_strength = edge_img.mean()
            st.metric("Edge Strength", f"{edge_strength:.1f}")
        
        st.markdown("---")
        
        # Edge detection comparison
        st.markdown("### 🔍 Edge Detection Comparison")
        
        # Different edge detection methods
        edge_method = st.radio(
            "Edge detection method:",
            ["High-pass Filter", "Sobel Operator", "Canny Edge Detector", "Laplacian"]
        )
        
        if edge_method == "High-pass Filter":
            edge_result = edge_img
        elif edge_method == "Sobel Operator":
            sobel_x = cv2.Sobel(img_np.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_np.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
            edge_result = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_result = np.clip(edge_result, 0, 255).astype(np.uint8)
        elif edge_method == "Canny Edge Detector":
            edge_result = cv2.Canny(img_np.astype(np.uint8), 100, 200)
        else:  # Laplacian
            edge_result = cv2.Laplacian(img_np.astype(np.uint8), cv2.CV_64F)
            edge_result = np.absolute(edge_result)
            edge_result = np.clip(edge_result, 0, 255).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(edge_result, cmap='gray')
        axes[1].set_title(f"{edge_method} - Edges")
        axes[1].axis('off')
        
        st.pyplot(fig)
        
        # Edge statistics
        edge_pixels = np.sum(edge_result > 50)
        total_pixels = edge_result.size
        edge_percentage = edge_pixels / total_pixels * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Edge Pixels", f"{edge_pixels:,}")
        with col2:
            st.metric("Total Pixels", f"{total_pixels:,}")
        with col3:
            st.metric("Edge Coverage", f"{edge_percentage:.1f}%")
        
        st.markdown("---")
        
        # Applications
        st.markdown("### 🚀 Applications of High-Pass Filters")
        
        app_tabs = st.tabs(["Image Enhancement", "Feature Extraction", "Medical Imaging", "Computer Vision"])
        
        with app_tabs[0]:
            st.markdown("""
            **Image Enhancement:**
            - **Unsharp masking**: Professional sharpening technique
            - **Detail enhancement**: Bring out fine textures
            - **Contrast boost**: Make edges pop
            - **Crispness**: For print/display preparation
            
            **Example:** Camera RAW processing uses high-pass filters for clarity.
            """)
        
        with app_tabs[1]:
            st.markdown("""
            **Feature Extraction:**
            - **Edge detection**: Find object boundaries
            - **Texture analysis**: Study material properties
            - **Pattern recognition**: Identify regular structures
            - **Feature matching**: For image alignment/stitching
            
            **Example:** Self-driving cars use edge detection for lane finding.
            """)
        
        with app_tabs[2]:
            st.markdown("""
            **Medical Imaging:**
            - **Tumor detection**: Highlight suspicious edges
            - **Bone fracture**: Enhance crack visibility
            - **Blood vessel**: Trace vascular networks
            - **Cell boundary**: In microscopy images
            
            **Example:** MRI scans use edge enhancement to see tumor boundaries.
            """)
        
        with app_tabs[3]:
            st.markdown("""
            **Computer Vision:**
            - **Object detection**: Find edges for bounding boxes
            - **OCR (text recognition)**: Enhance text clarity
            - **Face recognition**: Extract facial features
            - **Industrial inspection**: Detect defects/imperfections
            
            **Example:** Document scanners use sharpening for better OCR.
            """)
    
    # ==================== TAB 4: BAND & NOTCH FILTERS ====================
    with tab4:
        st.header("🟢 Band & Notch Filters: Selective Frequency Control")
        
        st.markdown("""
        Band and notch filters target specific frequency ranges for precise control.
        - **Band-pass**: Keep only middle frequencies (textures)
        - **Band-stop**: Remove specific frequency range (noise)
        - **Notch**: Remove specific frequency points (periodic noise)
        """)
        
        filter_type = st.selectbox(
            "Select filter type:",
            ["Band-pass Filter", "Band-stop Filter", "Notch Filter", "Multiple Notches"]
        )
        
        # Filter parameters
        if "Band" in filter_type:
            col1, col2 = st.columns(2)
            with col1:
                low_cut = st.slider("Low Cutoff", 0.01, 0.4, 0.1, 0.01)
            with col2:
                high_cut = st.slider("High Cutoff", low_cut + 0.01, 0.5, 0.3, 0.01)
        
        elif "Multiple" in filter_type:
            num_notches = st.slider("Number of notches", 1, 8, 3, 1)
            notch_size = st.slider("Notch size", 1, 20, 5, 1)
        else:
            col1, col2 = st.columns(2)
            with col1:
                notch_u = st.slider("Notch U position", -50, 50, 20, 1)
            with col2:
                notch_v = st.slider("Notch V position", -50, 50, 20, 1)
            notch_size = st.slider("Notch radius", 1, 20, 5, 1)
        
        # Create filter
        rows, cols = img_np.shape
        center_y, center_x = rows // 2, cols // 2
        
        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist = dist_from_center / max_dist
        
        if filter_type == "Band-pass Filter":
            filter_mask = np.zeros((rows, cols))
            filter_mask[(normalized_dist >= low_cut) & (normalized_dist <= high_cut)] = 1
            filter_desc = f"Band-pass: {low_cut:.2f} to {high_cut:.2f}"
        
        elif filter_type == "Band-stop Filter":
            filter_mask = np.ones((rows, cols))
            filter_mask[(normalized_dist >= low_cut) & (normalized_dist <= high_cut)] = 0
            filter_desc = f"Band-stop: removes {low_cut:.2f} to {high_cut:.2f}"
        
        elif filter_type == "Notch Filter":
            filter_mask = np.ones((rows, cols))
            # Create symmetric notches (Fourier symmetry)
            notch_positions = [
                (center_x + notch_u, center_y + notch_v),
                (center_x - notch_u, center_y - notch_v),
                (center_x + notch_u, center_y - notch_v),
                (center_x - notch_u, center_y + notch_v)
            ]
            for nx, ny in notch_positions:
                notch_dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                filter_mask[notch_dist < notch_size] = 0
            filter_desc = f"Notch at ({notch_u}, {notch_v})"
        
        else:  # Multiple notches
            filter_mask = np.ones((rows, cols))
            angles = np.linspace(0, 2*np.pi, num_notches, endpoint=False)
            radius = min(center_x, center_y) * 0.3
            
            for angle in angles:
                nx = center_x + int(radius * np.cos(angle))
                ny = center_y + int(radius * np.sin(angle))
                notch_dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                filter_mask[notch_dist < notch_size] = 0
            filter_desc = f"{num_notches} notches at radius {radius:.0f}"
        
        # Apply filter
        fft_img = fft2(img_np)
        fft_shifted = fftshift(fft_img)
        
        filtered_fft = fft_shifted * filter_mask
        filtered_fft_ishift = ifftshift(filtered_fft)
        filtered_img = np.real(ifft2(filtered_fft_ishift))
        filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
        
        # Display
        st.markdown(f"### 📊 {filter_type} Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np, cmap='gray')
            ax.set_title("Original")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(filter_mask, cmap='gray')
            ax.set_title(filter_desc)
            ax.axis('off')
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(filtered_img, cmap='gray')
            ax.set_title("Filtered")
            ax.axis('off')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Frequency analysis
        st.markdown("### 📈 Frequency Analysis")
        
        # Calculate original and filtered spectra
        magnitude_orig = np.log(np.abs(fft_shifted) + 1)
        magnitude_filtered = np.log(np.abs(filtered_fft) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0, 0].imshow(img_np, cmap='gray')
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(magnitude_orig, cmap='gray')
        axes[0, 1].set_title("Original Spectrum")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(filtered_img, cmap='gray')
        axes[1, 0].set_title("Filtered Image")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(magnitude_filtered, cmap='gray')
        axes[1, 1].set_title("Filtered Spectrum")
        axes[1, 1].axis('off')
        
        st.pyplot(fig)
        
        # Quantitative analysis
        st.markdown("### 📊 Quantitative Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            orig_mean = img_np.mean()
            st.metric("Original Mean", f"{orig_mean:.1f}")
        
        with col2:
            filtered_mean = filtered_img.mean()
            st.metric("Filtered Mean", f"{filtered_mean:.1f}")
        
        with col3:
            orig_std = img_np.std()
            st.metric("Original Std", f"{orig_std:.1f}")
        
        with col4:
            filtered_std = filtered_img.std()
            change_std = (filtered_std - orig_std) / orig_std * 100
            st.metric("Filtered Std", f"{filtered_std:.1f}", delta=f"{change_std:.1f}%")
        
        st.markdown("---")
        
        # Real-world examples
        st.markdown("### 🌍 Real-World Applications")
        
        if "Band-pass" in filter_type:
            st.info("""
            **Band-pass Filter Applications:**
            - **Texture analysis**: Isolate fabric weave patterns
            - **Feature extraction**: Extract specific size features
            - **Medical imaging**: Focus on specific tissue structures
            - **Remote sensing**: Analyze specific geological features
            
            **Example:** In fingerprint analysis, band-pass filters extract ridge patterns.
            """)
        
        elif "Band-stop" in filter_type:
            st.info("""
            **Band-stop Filter Applications:**
            - **Moire removal**: Remove interference patterns
            - **Scan line removal**: Clean up old video scans
            - **Power line removal**: Remove 50/60Hz interference
            - **Pattern suppression**: Remove regular textures
            
            **Example:** In document scanning, remove printer dot patterns.
            """)
        
        else:
            st.info("""
            **Notch Filter Applications:**
            - **Periodic noise removal**: Remove repeating patterns
            - **Interference removal**: Remove specific frequency interference
            - **Artifact removal**: Clean up compression artifacts
            - **Pattern elimination**: Remove specific unwanted patterns
            
            **Example:** In audio/video, remove hum from electrical interference.
            """)
    
    # ==================== TAB 5: FILTER DESIGN LAB ====================
    with tab5:
        st.header("🧪 Filter Design Laboratory")
        
        st.markdown("""
        Design your own custom filters and see their effects in real-time!
        This is where you become a frequency filter expert. 🎓
        """)
        
        # Filter design interface
        st.markdown("### 🎨 Custom Filter Designer")
        
        design_mode = st.radio(
            "Design mode:",
            ["Preset Templates", "Freehand Drawing", "Mathematical Function", "Frequency Selector"]
        )
        
        rows, cols = img_np.shape
        center_y, center_x = rows // 2, cols // 2
        
        if design_mode == "Preset Templates":
            template = st.selectbox(
                "Select template:",
                ["Circular Low-pass", "Circular High-pass", "Ring Band-pass", 
                 "Cross Pattern", "Radial Lines", "Checkerboard", "Spiral"]
            )
            
            # Create template-based filter
            filter_mask = create_template_filter(rows, cols, template)
        
        elif design_mode == "Freehand Drawing":
            st.markdown("Draw your filter mask (white = keep, black = remove):")
            
            # Create a canvas for drawing
            canvas_size = min(300, min(rows, cols))
            drawn_mask = np.ones((canvas_size, canvas_size))
            
            # Simple drawing interface
            col1, col2, col3 = st.columns(3)
            with col1:
                draw_value = st.slider("Draw value", 0.0, 1.0, 1.0, 0.1)
            with col2:
                brush_size = st.slider("Brush size", 1, 20, 5, 1)
            with col3:
                if st.button("Clear Canvas"):
                    drawn_mask = np.ones((canvas_size, canvas_size))
            
            # Display drawing area
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(drawn_mask, cmap='gray', vmin=0, vmax=1)
            ax.set_title("Draw your filter (click positions)")
            ax.axis('off')
            st.pyplot(fig)
            
            # Convert to full size
            filter_mask = cv2.resize(drawn_mask, (cols, rows), interpolation=cv2.INTER_LINEAR)
        
        elif design_mode == "Mathematical Function":
            st.markdown("Define your filter with a mathematical function:")
            
            function = st.text_input(
                "Enter function f(r, θ) where r=distance, θ=angle:",
                value="np.exp(-r**2/0.1) * np.cos(4*θ)**2"
            )
            
            # Create coordinate grids
            y, x = np.ogrid[:rows, :cols]
            y = y - center_y
            x = x - center_x
            
            r = np.sqrt(x**2 + y**2) / max(center_x, center_y)
            theta = np.arctan2(y, x)
            
            try:
                # Evaluate the function
                filter_mask = eval(function, {"np": np, "r": r, "theta": theta})
                filter_mask = np.clip(filter_mask, 0, 1)
            except:
                st.error("Invalid function! Using default.")
                filter_mask = np.ones((rows, cols))
        
        else:  # Frequency Selector
            st.markdown("Select specific frequencies to keep/remove:")
            
            freq_u = st.slider("Horizontal frequency range", -cols//2, cols//2, (-30, 30), 1)
            freq_v = st.slider("Vertical frequency range", -rows//2, rows//2, (-30, 30), 1)
            
            filter_mask = np.zeros((rows, cols))
            filter_mask[center_y+freq_v[0]:center_y+freq_v[1], 
                       center_x+freq_u[0]:center_x+freq_u[1]] = 1
        
        # Apply the custom filter
        fft_img = fft2(img_np)
        fft_shifted = fftshift(fft_img)
        
        filtered_fft = fft_shifted * filter_mask
        filtered_fft_ishift = ifftshift(filtered_fft)
        filtered_img = np.real(ifft2(filtered_fft_ishift))
        filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
        
        # Display results
        st.markdown("---")
        st.markdown("### 🔬 Custom Filter Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np, cmap='gray')
            ax.set_title("Original")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(filter_mask, cmap='gray')
            ax.set_title("Your Custom Filter")
            ax.axis('off')
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(filtered_img, cmap='gray')
            ax.set_title("Filtered Result")
            ax.axis('off')
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Filter analysis
        st.markdown("### 📊 Filter Analysis")
        
        # Calculate filter properties
        kept_freq = np.sum(filter_mask > 0.5) / filter_mask.size * 100
        removed_freq = 100 - kept_freq
        
        low_freq_region = create_circular_mask(rows, cols, 0.2)
        mid_freq_region = create_band_mask(rows, cols, 0.2, 0.4)
        high_freq_region = 1 - create_circular_mask(rows, cols, 0.4)
        
        low_kept = np.sum(filter_mask[low_freq_region > 0.5]) / np.sum(low_freq_region) * 100
        mid_kept = np.sum(filter_mask[mid_freq_region > 0.5]) / np.sum(mid_freq_region) * 100
        high_kept = np.sum(filter_mask[high_freq_region > 0.5]) / np.sum(high_freq_region) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Frequencies Kept", f"{kept_freq:.1f}%")
        
        with col2:
            st.metric("Low Frequencies", f"{low_kept:.1f}%")
        
        with col3:
            st.metric("Mid Frequencies", f"{mid_kept:.1f}%")
        
        with col4:
            st.metric("High Frequencies", f"{high_kept:.1f}%")
        
        # Predict effect
        st.markdown("### 🔮 Predicted Effect:")
        
        if low_kept > 80 and high_kept < 20:
            st.success("**Low-pass effect:** Image will be blurred/smoothed")
        elif low_kept < 20 and high_kept > 80:
            st.success("**High-pass effect:** Image will be sharpened/edges enhanced")
        elif mid_kept > 60 and low_kept < 40 and high_kept < 40:
            st.success("**Band-pass effect:** Textures will be isolated")
        else:
            st.info("**Complex effect:** Mixed frequency response")
        
        st.markdown("---")
        
        # Save/load filters
        st.markdown("### 💾 Filter Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save This Filter"):
                filter_name = st.text_input("Filter name:", "my_custom_filter")
                if filter_name:
                    np.save(f"filter_{filter_name}.npy", filter_mask)
                    st.success(f"Filter saved as 'filter_{filter_name}.npy'")
        
        with col2:
            if st.button("📂 Load Saved Filter"):
                # List saved filters
                saved_filters = [f for f in os.listdir('.') if f.startswith('filter_') and f.endswith('.npy')]
                if saved_filters:
                    selected_filter = st.selectbox("Select filter to load:", saved_filters)
                    if selected_filter:
                        filter_mask = np.load(selected_filter)
                        st.success(f"Loaded {selected_filter}")
                else:
                    st.warning("No saved filters found")
        
        st.markdown("---")
        
        # Challenge exercises
        st.markdown("### 🏆 Filter Design Challenges")
        
        challenge = st.selectbox(
            "Try these challenges:",
            ["None", "Create edge detector", "Remove diagonal stripes", 
             "Isolate circular patterns", "Create artistic effect"]
        )
        
        if challenge != "None":
            st.markdown(f"**Challenge:** {challenge}")
            
            if challenge == "Create edge detector":
                st.markdown("Design a filter that detects edges in all directions")
                hint = "Try a filter that blocks the center but keeps everything else"
            
            elif challenge == "Remove diagonal stripes":
                st.markdown("Design a filter to remove diagonal line patterns")
                hint = "Look for the diagonal lines in the spectrum and block them"
            
            elif challenge == "Isolate circular patterns":
                st.markdown("Design a filter to keep only circular/radial patterns")
                hint = "Circular patterns create circular patterns in spectrum"
            
            else:
                st.markdown("Design an artistic filter for creative effects")
                hint = "Try asymmetric patterns or mathematical functions"
            
            st.info(f"💡 Hint: {hint}")
            
            if st.button("Show Solution"):
                if challenge == "Create edge detector":
                    solution = 1 - create_circular_mask(rows, cols, 0.1)
                    st.image(solution, caption="High-pass filter blocks center", use_container_width=True)
                
                elif challenge == "Remove diagonal stripes":
                    solution = np.ones((rows, cols))
                    # Block diagonal lines
                    for angle in [45, 135, 225, 315]:
                        nx = int(center_x + 100 * np.cos(np.radians(angle)))
                        ny = int(center_y + 100 * np.sin(np.radians(angle)))
                        dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                        solution[dist < 5] = 0
                    st.image(solution, caption="Notches on diagonals", use_container_width=True)
                
                elif challenge == "Isolate circular patterns":
                    solution = create_band_mask(rows, cols, 0.15, 0.25)
                    st.image(solution, caption="Ring-shaped band-pass", use_container_width=True)
                
                else:
                    solution = np.abs(np.sin(0.1*x) * np.cos(0.1*y))
                    st.image(solution, caption="Wave pattern filter", use_container_width=True)
        
        st.success("""
        ### 🎉 Congratulations!
        
        You've completed the Filter Design Lab! You now understand:
        
        1. **Different filter types** and their effects
        2. **How to apply filters** in frequency domain
        3. **How to design custom filters** for specific needs
        4. **Practical applications** of each filter type
        
        You're ready to tackle real-world image processing challenges! 🚀
        """)

# Helper functions
def create_demo_image():
    """Create a synthetic demo image"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add various patterns
    for i in range(size):
        for j in range(size):
            # Gradient background
            img[i, j] = 128 + 30 * np.sin(0.03 * i) + 30 * np.cos(0.03 * j)
    
    # Add shapes
    cv2.rectangle(img, (30, 30), (100, 100), 200, -1)
    cv2.circle(img, (180, 180), 50, 150, -1)
    
    # Add text-like pattern
    for i in range(50, 200, 15):
        img[i:i+3, 50:200] = 255
    
    # Add noise
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255)
    
    return Image.fromarray(img.astype(np.uint8))

def create_circular_mask(h, w, radius_ratio):
    """Create circular mask for low-pass filter"""
    center = (w//2, h//2)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    max_dist = np.sqrt(center[0]**2 + center[1]**2)
    mask = dist_from_center <= max_dist * radius_ratio
    return mask.astype(float)

def create_band_mask(h, w, low_ratio, high_ratio):
    """Create band mask for band-pass filter"""
    center = (w//2, h//2)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    max_dist = np.sqrt(center[0]**2 + center[1]**2)
    normalized_dist = dist_from_center / max_dist
    mask = (normalized_dist >= low_ratio) & (normalized_dist <= high_ratio)
    return mask.astype(float)

def create_notch_mask(h, w):
    """Create notch filter mask"""
    center = (w//2, h//2)
    Y, X = np.ogrid[:h, :w]
    mask = np.ones((h, w))
    
    # Add some notches
    notch_positions = [
        (center[0] + 50, center[1]),
        (center[0] - 50, center[1]),
        (center[0], center[1] + 50),
        (center[0], center[1] - 50)
    ]
    
    for nx, ny in notch_positions:
        dist = np.sqrt((X - nx)**2 + (Y - ny)**2)
        mask[dist < 10] = 0
    
    return mask

def create_template_filter(h, w, template):
    """Create filter from template"""
    center = (w//2, h//2)
    Y, X = np.ogrid[:h, :w]
    
    if template == "Circular Low-pass":
        return create_circular_mask(h, w, 0.3)
    
    elif template == "Circular High-pass":
        return 1 - create_circular_mask(h, w, 0.3)
    
    elif template == "Ring Band-pass":
        return create_band_mask(h, w, 0.2, 0.4)
    
    elif template == "Cross Pattern":
        mask = np.zeros((h, w))
        mask[center[1]-10:center[1]+10, :] = 1
        mask[:, center[0]-10:center[0]+10] = 1
        return mask
    
    elif template == "Radial Lines":
        mask = np.zeros((h, w))
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for angle in angles:
            length = min(h, w) // 2
            x_end = center[0] + int(length * np.cos(angle))
            y_end = center[1] + int(length * np.sin(angle))
            cv2.line(mask, center, (x_end, y_end), 1, 5)
        return mask
    
    elif template == "Checkerboard":
        mask = np.zeros((h, w))
        block_size = 20
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if (i//block_size + j//block_size) % 2 == 0:
                    mask[i:i+block_size, j:j+block_size] = 1
        return mask
    
    else:  # Spiral
        mask = np.zeros((h, w))
        theta = np.arctan2(Y - center[1], X - center[0])
        r = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        spiral = np.sin(5 * theta + 0.1 * r)
        mask[spiral > 0] = 1
        return mask

if __name__ == "__main__":
    app()