import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def app():
    st.title("🏆 Image Enhancement Challenge: Fix Real-World Photos!")
    
    # Progressive learning tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ The Challenge", 
        "2️⃣ Diagnose the Problem", 
        "3️⃣ Apply Solutions",
        "4️⃣ Compare Results",
        "5️⃣ Become an Expert"
    ])
    
    # Load images
    @st.cache_data
    def load_images():
        images = {}
        problems = {}
        descriptions = {}
        
        # Try to load all images
        for i in range(1, 4):
            try:
                img = Image.open(f"public/image{i}.png").convert('L')
                images[f"image{i}"] = np.array(img).astype(float)
                
                # Assign problems based on filename patterns
                if i == 1:
                    problems[f"image{i}"] = "Blurry"
                    descriptions[f"image{i}"] = "Out of focus - needs sharpening"
                elif i == 2:
                    problems[f"image{i}"] = "Noisy"
                    descriptions[f"image{i}"] = "Grainy texture - needs smoothing"
                else:
                    problems[f"image{i}"] = "Low Contrast"
                    descriptions[f"image{i}"] = "Flat appearance - needs contrast boost"
                    
            except Exception as e:
                # Create synthetic images if files not found
                synthetic = np.random.rand(300, 400) * 255
                if i == 1:  # Blurry synthetic
                    synthetic = np.random.rand(300, 400) * 255
                    for _ in range(3):
                        synthetic = (synthetic + np.roll(synthetic, 1, axis=0) + 
                                   np.roll(synthetic, -1, axis=0) + 
                                   np.roll(synthetic, 1, axis=1) + 
                                   np.roll(synthetic, -1, axis=1)) / 5
                    problems[f"image{i}"] = "Blurry"
                    descriptions[f"image{i}"] = "Synthetic blurry image"
                elif i == 2:  # Noisy synthetic
                    clean = np.random.rand(300, 400) * 100 + 100
                    noise = np.random.randn(300, 400) * 50
                    synthetic = np.clip(clean + noise, 0, 255)
                    problems[f"image{i}"] = "Noisy"
                    descriptions[f"image{i}"] = "Synthetic noisy image"
                else:  # Low contrast synthetic
                    synthetic = np.random.rand(300, 400) * 50 + 100
                    problems[f"image{i}"] = "Low Contrast"
                    descriptions[f"image{i}"] = "Synthetic low contrast image"
                
                images[f"image{i}"] = synthetic
        
        return images, problems, descriptions
    
    images, problems, descriptions = load_images()
    
    # ==================== TAB 1: THE CHALLENGE ====================
    with tab1:
        st.header("🎯 Welcome to the Image Enhancement Challenge!")
        
        st.markdown("""
        ### Imagine You're a Photo Doctor! 👨‍⚕️📸
        
        Patients (images) come to you with different problems:
        
        1. **Blurry Brian** - Can't see details clearly 😵
        2. **Noisy Nancy** - Full of grainy spots 🌪️  
        3. **Flat Frank** - Everything looks washed out 🎨
        
        **Your mission:** Diagnose each problem and apply the right treatment! 💊
        
        ---
        
        ### Real-World Impact: Why This Matters 🌍
        
        **Every day, people need to fix photos like these:**
        - 👵 **Grandma's old photos** - Faded and blurry
        - 🏥 **Medical scans** - Need clearer details for diagnosis
        - 📱 **Smartphone photos** - Low light = noisy/blurry
        - 🛰️ **Satellite images** - Atmospheric haze = low contrast
        
        **You'll learn skills used by:** 
        - Photo editors 📷
        - Medical technicians 🏥  
        - Security analysts 🔍
        - Space scientists 🚀
        """)
        
        st.markdown("---")
        
        # Show the three problem images
        st.markdown("### 📸 Meet Our Three Patients:")
        
        cols = st.columns(3)
        
        for idx, img_key in enumerate(["image1", "image2", "image3"]):
            with cols[idx]:
                problem = problems[img_key]
                desc = descriptions[img_key]
                
                # Display image
                img_display = np.clip(images[img_key], 0, 255).astype(np.uint8)
                st.image(img_display, caption=f"Patient {idx+1}: {problem}", 
                        use_container_width=True, channels="L")
                
                # Problem description
                st.markdown(f"**Problem:** {problem}")
                st.markdown(f"**Symptoms:** {desc}")
                
                # Fun icon based on problem
                if problem == "Blurry":
                    st.markdown("**Treatment:** 🪄 Sharpening magic!")
                elif problem == "Noisy":
                    st.markdown("**Treatment:** 🌊 Smoothing waves!")
                else:
                    st.markdown("**Treatment:** ⚡ Contrast boost!")
        
        st.success("""
        ✅ **Ready to help?** Each patient needs different treatment!
        
        **Remember:** You can't use the same medicine for every patient! 💊
        - Blurry → Needs sharpening 🔪
        - Noisy → Needs smoothing 🌊  
        - Flat → Needs contrast ⚡
        """)
        
        st.markdown("---")
        
        # Interactive challenge
        st.markdown("### 🤔 Quick Diagnosis Quiz")
        
        quiz_answer = st.radio(
            "Which image would benefit MOST from edge enhancement?",
            ["Blurry Brian (Patient 1)", "Noisy Nancy (Patient 2)", "Flat Frank (Patient 3)"],
            index=None
        )
        
        if quiz_answer:
            if "Blurry Brian" in quiz_answer:
                st.success("✅ **Correct!** Blurry images need edge enhancement the most!")
                st.balloons()
            else:
                st.warning("⚠️ **Almost!** While others might need some enhancement, blurry images benefit MOST from edge enhancement.")
        
        st.info("""
        💡 **Pro Tip:** 
        - **Blurry = Lost edges** → Restore with sharpening
        - **Noisy = Fake edges** → Remove noise first
        - **Flat = Weak edges** → Boost contrast to reveal
        """)
    
    # ==================== TAB 2: DIAGNOSE THE PROBLEM ====================
    with tab2:
        st.header("🔍 Diagnose the Problem Like a Pro")
        
        # Image selection
        st.markdown("### 🎯 Select a Patient to Examine")
        
        selected_image = st.radio(
            "Choose which image to diagnose:",
            ["Patient 1: Blurry Brian", "Patient 2: Noisy Nancy", "Patient 3: Flat Frank"],
            horizontal=True
        )
        
        # Get selected image data
        if "Patient 1" in selected_image:
            img_key = "image1"
            img_np = images["image1"]
            problem = "Blurry"
        elif "Patient 2" in selected_image:
            img_key = "image2"
            img_np = images["image2"]
            problem = "Noisy"
        else:
            img_key = "image3"
            img_np = images["image3"]
            problem = "Low Contrast"
        
        height, width = img_np.shape
        
        st.markdown(f"### 🩺 Examining: {selected_image}")
        
        # Show image and basic stats
        col1, col2 = st.columns(2)
        
        with col1:
            img_display = np.clip(img_np, 0, 255).astype(np.uint8)
            st.image(img_display, caption=f"Current Condition: {problem}", 
                    use_container_width=True, channels="L")
        
        with col2:
            # Calculate statistics
            min_val = img_np.min()
            max_val = img_np.max()
            mean_val = img_np.mean()
            std_val = img_np.std()
            range_val = max_val - min_val
            
            st.markdown("### 📊 Vital Statistics")
            
            stats = {
                "Brightness Range": f"{min_val:.0f} - {max_val:.0f}",
                "Dynamic Range": f"{range_val:.0f} / 255",
                "Average Brightness": f"{mean_val:.1f}",
                "Contrast (Std Dev)": f"{std_val:.1f}",
                "Contrast Percentage": f"{(std_val/mean_val*100 if mean_val>0 else 0):.1f}%"
            }
            
            for key, value in stats.items():
                st.metric(key, value)
        
        # Diagnostic tests
        st.markdown("---")
        st.markdown("### 🧪 Run Diagnostic Tests")
        
        diagnostic = st.selectbox(
            "Choose a diagnostic test:",
            ["No test selected", "Edge Detection Test", "Noise Analysis", "Histogram Analysis", "Zoom Inspection"]
        )
        
        if diagnostic != "No test selected":
            if diagnostic == "Edge Detection Test":
                st.markdown("**Edge Detection Test:** Checks how clear edges are")
                
                # Simple edge detection
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
                
                # Manual convolution
                def manual_conv(image, kernel):
                    h, w = image.shape
                    kh, kw = kernel.shape
                    pad_h, pad_w = kh//2, kw//2
                    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
                    output = np.zeros_like(image, dtype=float)
                    
                    for i in range(h):
                        for j in range(w):
                            region = padded[i:i+kh, j:j+kw]
                            output[i, j] = np.sum(region * kernel)
                    
                    return output
                
                gx = manual_conv(img_np, sobel_x)
                gy = manual_conv(img_np, sobel_y)
                edges = np.sqrt(gx**2 + gy**2)
                edges_display = np.clip(edges, 0, 255).astype(np.uint8)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img_display, caption="Original", 
                            use_container_width=True, channels="L")
                
                with col2:
                    st.image(edges_display, caption="Edge Map", 
                            use_container_width=True, channels="L")
                
                # Analyze edge strength
                edge_strength = edges.mean()
                strong_edges = np.sum(edges > 50)
                
                st.markdown(f"""
                **Diagnosis:**
                - Average edge strength: {edge_strength:.1f}/255
                - Strong edges detected: {strong_edges:,} pixels
                - Edge clarity: {'❌ Poor' if edge_strength < 30 else '⚠️ Moderate' if edge_strength < 60 else '✅ Good'}
                """)
                
                if problem == "Blurry":
                    st.error("**Confirmed:** Weak edges detected → Needs sharpening!")
                elif problem == "Noisy":
                    st.warning("**Observation:** Many small edges → Could be noise!")
                else:
                    st.info("**Observation:** Edges present but weak → Needs contrast!")
            
            elif diagnostic == "Noise Analysis":
                st.markdown("**Noise Analysis:** Checks for random graininess")
                
                # Simple noise estimation
                # Smooth the image
                smooth_kernel = np.ones((5,5), dtype=float) / 25
                
                def manual_conv(image, kernel):
                    h, w = image.shape
                    kh, kw = kernel.shape
                    pad_h, pad_w = kh//2, kw//2
                    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
                    output = np.zeros_like(image, dtype=float)
                    
                    for i in range(h):
                        for j in range(w):
                            region = padded[i:i+kh, j:j+kw]
                            output[i, j] = np.sum(region * kernel)
                    
                    return output
                
                smoothed = manual_conv(img_np, smooth_kernel)
                noise = np.abs(img_np - smoothed)
                noise_display = np.clip(noise * 3, 0, 255).astype(np.uint8)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img_display, caption="Original", 
                            use_container_width=True, channels="L")
                
                with col2:
                    st.image(noise_display, caption="Estimated Noise (amplified 3×)", 
                            use_container_width=True, channels="L")
                
                # Analyze noise level
                noise_level = noise.mean()
                noisy_pixels = np.sum(noise > 20)
                
                st.markdown(f"""
                **Diagnosis:**
                - Average noise level: {noise_level:.1f}
                - Noisy pixels (>20): {noisy_pixels:,}
                - Noise severity: {'✅ Low' if noise_level < 10 else '⚠️ Moderate' if noise_level < 30 else '❌ High'}
                """)
                
                if problem == "Noisy":
                    st.error("**Confirmed:** High noise level detected → Needs smoothing!")
                else:
                    st.success("**Good news:** Noise level is acceptable")
            
            elif diagnostic == "Histogram Analysis":
                st.markdown("**Histogram Analysis:** Checks brightness distribution")
                
                # Create histogram
                hist, bins = np.histogram(img_np.flatten(), 256, [0, 256])
                
                # Create simple histogram visualization
                hist_viz = np.zeros((100, 256, 3), dtype=np.uint8)
                hist_max = hist.max()
                
                for i in range(256):
                    if hist_max > 0:
                        height = int(hist[i] / hist_max * 100)
                        hist_viz[100-height:100, i, :] = [255, 255, 255]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img_display, caption="Original", 
                            use_container_width=True, channels="L")
                
                with col2:
                    st.image(hist_viz, caption="Brightness Histogram", 
                            use_container_width=True)
                
                # Analyze histogram
                low_values = np.sum(img_np < 64)
                mid_values = np.sum((img_np >= 64) & (img_np < 192))
                high_values = np.sum(img_np >= 192)
                total_pixels = height * width
                
                st.markdown(f"""
                **Brightness Distribution:**
                - Dark pixels (0-63): {low_values:,} ({low_values/total_pixels*100:.1f}%)
                - Mid pixels (64-191): {mid_values:,} ({mid_values/total_pixels*100:.1f}%)
                - Bright pixels (192-255): {high_values:,} ({high_values/total_pixels*100:.1f}%)
                
                **Diagnosis:** {'❌ Poor distribution' if max(low_values, mid_values, high_values)/total_pixels > 0.8 else '⚠️ Could be better' if max(low_values, mid_values, high_values)/total_pixels > 0.6 else '✅ Good distribution'}
                """)
                
                if problem == "Low Contrast":
                    st.error("**Confirmed:** Limited brightness range → Needs contrast enhancement!")
                elif max(low_values, mid_values, high_values)/total_pixels > 0.8:
                    st.warning("**Warning:** One brightness dominates → Could use adjustment")
            
            else:  # Zoom Inspection
                st.markdown("**Zoom Inspection:** Look at fine details")
                
                # Select zoom area
                zoom_level = st.slider("Zoom Level", 2.0, 8.0, 4.0, step=0.5)
                
                center_y, center_x = height // 2, width // 2
                zoom_size = int(100 / zoom_level)
                
                y_start = max(0, center_y - zoom_size)
                y_end = min(height, center_y + zoom_size)
                x_start = max(0, center_x - zoom_size)
                x_end = min(width, center_x + zoom_size)
                
                zoom_area = img_np[y_start:y_end, x_start:x_end]
                zoom_display = np.clip(zoom_area, 0, 255).astype(np.uint8)
                
                st.image(zoom_display, caption=f"Zoom {zoom_level:.1f}× (Center Region)", 
                        use_container_width=True, channels="L")
                
                # Analyze zoomed area
                zoom_std = zoom_area.std()
                
                st.markdown(f"""
                **Zoom Analysis:**
                - Region size: {zoom_area.shape[1]}×{zoom_area.shape[0]} pixels
                - Local contrast: {zoom_std:.1f}
                - Detail visibility: {'❌ Poor' if zoom_std < 15 else '⚠️ Moderate' if zoom_std < 30 else '✅ Good'}
                
                **What to look for:**
                - **Blurry:** No fine details, smooth transitions
                - **Noisy:** Random speckles, grain texture
                - **Flat:** Limited shades, washed out look
                """)
        
        # Final diagnosis
        st.markdown("---")
        st.markdown("### 📋 Final Diagnosis Report")
        
        diagnosis_card = f"""
        **Patient:** {selected_image}
        
        **Primary Issue:** {problem}
        
        **Recommended Treatment:**
        """
        
        if problem == "Blurry":
            diagnosis_card += """
            - Primary: Edge enhancement (sharpening)
            - Secondary: Contrast boost
            - Avoid: Heavy smoothing
            """
        elif problem == "Noisy":
            diagnosis_card += """
            - Primary: Noise reduction (smoothing)
            - Secondary: Edge-preserving techniques
            - Avoid: Aggressive sharpening
            """
        else:  # Low Contrast
            diagnosis_card += """
            - Primary: Contrast enhancement
            - Secondary: Edge sharpening
            - Avoid: Over-saturation
            """
        
        st.info(diagnosis_card)
        
        st.success("""
        ✅ **Diagnosis complete!** 
        
        Now let's apply the right treatment in the next tab! 💊
        """)
    
    # ==================== TAB 3: APPLY SOLUTIONS ====================
    with tab3:
        st.header("💊 Apply the Right Treatment")
        
        st.markdown("""
        ### Time for Treatment! 🏥
        
        Based on your diagnosis, choose the right enhancement technique.
        
        **Remember:** Different problems need different medicines! 💊
        """)
        
        # Image selection
        st.markdown("### 🎯 Select Patient to Treat")
        
        treatment_image = st.radio(
            "Choose which image to enhance:",
            ["Patient 1: Blurry Brian", "Patient 2: Noisy Nancy", "Patient 3: Flat Frank"],
            horizontal=True,
            key="treatment_select"
        )
        
        # Get selected image
        if "Patient 1" in treatment_image:
            img_key = "image1"
            img_np = images["image1"]
            problem = "Blurry"
            default_method = "Unsharp Masking"
        elif "Patient 2" in treatment_image:
            img_key = "image2"
            img_np = images["image2"]
            problem = "Noisy"
            default_method = "Gaussian Blur"
        else:
            img_key = "image3"
            img_np = images["image3"]
            problem = "Low Contrast"
            default_method = "Histogram Equalization"
        
        height, width = img_np.shape
        
        # Manual convolution function
        def manual_convolution(image, kernel):
            h, w = image.shape
            kh, kw = kernel.shape
            pad_h, pad_w = kh//2, kw//2
            
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
            output = np.zeros_like(image, dtype=float)
            
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kh, j:j+kw]
                    output[i, j] = np.sum(region * kernel)
            
            return output
        
        st.markdown(f"### 🩹 Treating: {treatment_image}")
        
        # Problem-specific treatment options
        if problem == "Blurry":
            st.markdown("**Treatment Strategy:** Sharpening to restore lost details")
            
            treatment = st.radio(
                "Choose sharpening technique:",
                ["Unsharp Masking", "Direct Kernel", "Smart Sharpening"],
                horizontal=True
            )
            
            st.markdown("---")
            
            if treatment == "Unsharp Masking":
                col1, col2 = st.columns(2)
                
                with col1:
                    blur_size = st.slider("Blur Size", 3, 9, 5, step=2,
                                         help="Larger = affect larger features")
                
                with col2:
                    amount = st.slider("Sharpening Amount", 0.5, 2.5, 1.2, step=0.1,
                                      help="Higher = stronger sharpening")
                
                if st.button("🪄 Apply Unsharp Masking", type="primary"):
                    with st.spinner("Sharpening image..."):
                        # Create blur kernel
                        blur_kernel = np.ones((blur_size, blur_size), dtype=float)
                        blur_kernel /= (blur_size * blur_size)
                        
                        # Apply blur
                        blurred = manual_convolution(img_np, blur_kernel)
                        
                        # Create mask and sharpen
                        mask = img_np - blurred
                        enhanced = img_np + mask * amount
                        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                        
                        st.session_state.enhanced_img = enhanced
                        st.session_state.method_used = f"Unsharp Masking (Size={blur_size}, Amount={amount})"
            
            elif treatment == "Direct Kernel":
                col1, col2 = st.columns(2)
                
                with col1:
                    center_strength = st.slider("Center Strength", 3.0, 11.0, 5.0, step=0.5,
                                               help="Higher = stronger sharpening")
                
                with col2:
                    neighbor_weight = st.slider("Neighbor Weight", -2.0, 0.0, -1.0, step=0.1,
                                               help="More negative = stronger edge emphasis")
                
                if st.button("⚡ Apply Direct Kernel", type="primary"):
                    with st.spinner("Applying sharpening kernel..."):
                        # Create kernel
                        kernel = np.array([[neighbor_weight, neighbor_weight, neighbor_weight],
                                          [neighbor_weight, center_strength, neighbor_weight],
                                          [neighbor_weight, neighbor_weight, neighbor_weight]], dtype=float)
                        
                        enhanced = manual_convolution(img_np, kernel)
                        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                        
                        st.session_state.enhanced_img = enhanced
                        st.session_state.method_used = f"Direct Kernel (Center={center_strength}, Neighbor={neighbor_weight})"
            
            else:  # Smart Sharpening
                st.info("""
                **Smart Sharpening:** Combines multiple techniques
                1. Mild noise reduction first
                2. Targeted edge enhancement
                3. Final contrast adjustment
                """)
                
                if st.button("🧠 Apply Smart Sharpening", type="primary"):
                    with st.spinner("Applying smart enhancement..."):
                        # Step 1: Mild smoothing
                        smooth_kernel = np.ones((3,3), dtype=float) / 9
                        smoothed = manual_convolution(img_np, smooth_kernel)
                        
                        # Step 2: Edge enhancement
                        edge_kernel = np.array([[0, -1, 0],
                                               [-1, 5, -1],
                                               [0, -1, 0]], dtype=float)
                        edged = manual_convolution(smoothed, edge_kernel)
                        
                        # Step 3: Contrast stretch
                        min_val = edged.min()
                        max_val = edged.max()
                        if max_val > min_val:
                            enhanced = (edged - min_val) * (255 / (max_val - min_val))
                        else:
                            enhanced = edged
                        
                        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                        
                        st.session_state.enhanced_img = enhanced
                        st.session_state.method_used = "Smart Sharpening (3-step process)"
        
        elif problem == "Noisy":
            st.markdown("**Treatment Strategy:** Noise reduction while preserving edges")
            
            treatment = st.radio(
                "Choose noise reduction technique:",
                ["Gaussian Blur", "Edge-Preserving", "Adaptive Smoothing"],
                horizontal=True
            )
            
            st.markdown("---")
            
            if treatment == "Gaussian Blur":
                col1, col2 = st.columns(2)
                
                with col1:
                    kernel_size = st.slider("Kernel Size", 3, 7, 5, step=2,
                                           help="Larger = more smoothing")
                
                with col2:
                    sigma = st.slider("Sigma", 0.5, 2.5, 1.2, step=0.1,
                                     help="Higher = wider smoothing")
                
                if st.button("🌊 Apply Gaussian Blur", type="primary"):
                    with st.spinner("Reducing noise..."):
                        # Create Gaussian kernel
                        size = kernel_size
                        kernel = np.zeros((size, size), dtype=float)
                        center = size // 2
                        
                        for i in range(size):
                            for j in range(size):
                                x, y = i - center, j - center
                                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                        
                        kernel /= np.sum(kernel)
                        
                        enhanced = manual_convolution(img_np, kernel)
                        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                        
                        st.session_state.enhanced_img = enhanced
                        st.session_state.method_used = f"Gaussian Blur (Size={size}, Sigma={sigma})"
            
            elif treatment == "Edge-Preserving":
                col1, col2 = st.columns(2)
                
                with col1:
                    smooth_amount = st.slider("Smoothing Strength", 1.0, 5.0, 2.0, step=0.5,
                                             help="Higher = more smoothing")
                
                with col2:
                    edge_threshold = st.slider("Edge Protection", 10, 100, 40,
                                              help="Higher = preserve more edges")
                
                if st.button("🛡️ Apply Edge-Preserving", type="primary"):
                    with st.spinner("Preserving edges while smoothing..."):
                        # Simple edge-preserving filter
                        # First, detect edges
                        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
                        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
                        
                        gx = manual_convolution(img_np, sobel_x)
                        gy = manual_convolution(img_np, sobel_y)
                        edges = np.sqrt(gx**2 + gy**2)
                        
                        # Create smoothed version
                        smooth_kernel = np.ones((5,5), dtype=float) / 25
                        smoothed = manual_convolution(img_np, smooth_kernel)
                        
                        # Blend based on edge strength
                        edge_mask = np.clip(edges / edge_threshold, 0, 1)
                        enhanced = edge_mask * img_np + (1 - edge_mask) * smoothed
                        enhanced = np.clip(enhanced * smooth_amount, 0, 255).astype(np.uint8)
                        
                        st.session_state.enhanced_img = enhanced
                        st.session_state.method_used = f"Edge-Preserving (Strength={smooth_amount}, Threshold={edge_threshold})"
            
            else:  # Adaptive Smoothing
                st.info("""
                **Adaptive Smoothing:** 
                - Smooths more in flat areas
                - Smooths less near edges
                - Automatically adjusts
                """)
                
                if st.button("🤖 Apply Adaptive Smoothing", type="primary"):
                    with st.spinner("Adaptively smoothing..."):
                        enhanced = np.zeros_like(img_np, dtype=float)
                        
                        # Simple adaptive smoothing
                        for i in range(1, height-1):
                            for j in range(1, width-1):
                                # Get local region
                                region = img_np[i-1:i+2, j-1:j+2]
                                local_std = region.std()
                                
                                # Adaptive kernel size based on local variation
                                if local_std < 10:  # Flat area
                                    # Use 5×5 smoothing
                                    if i > 2 and i < height-3 and j > 2 and j < width-3:
                                        region_large = img_np[i-2:i+3, j-2:j+3]
                                        enhanced[i, j] = region_large.mean()
                                    else:
                                        enhanced[i, j] = region.mean()
                                elif local_std < 30:  # Moderate variation
                                    # Use 3×3 smoothing
                                    enhanced[i, j] = region.mean()
                                else:  # High variation (edge)
                                    # Keep original
                                    enhanced[i, j] = img_np[i, j]
                        
                        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                        
                        st.session_state.enhanced_img = enhanced
                        st.session_state.method_used = "Adaptive Smoothing"
        
        else:  # Low Contrast
            st.markdown("**Treatment Strategy:** Boost contrast and reveal details")
            
            treatment = st.radio(
                "Choose contrast enhancement technique:",
                ["Histogram Equalization", "Contrast Stretch", "Local Contrast"],
                horizontal=True
            )
            
            st.markdown("---")
            
            if treatment == "Histogram Equalization":
                st.info("""
                **Histogram Equalization:**
                - Redistributes brightness values
                - Uses full 0-255 range
                - Automatic adjustment
                """)
                
                if st.button("📊 Apply Histogram Equalization", type="primary"):
                    with st.spinner("Equalizing histogram..."):
                        # Manual histogram equalization
                        hist, bins = np.histogram(img_np.flatten(), 256, [0, 256])
                        cdf = hist.cumsum()
                        cdf_normalized = cdf * 255 / cdf[-1]
                        enhanced = np.interp(img_np.flatten(), bins[:-1], cdf_normalized)
                        enhanced = enhanced.reshape(img_np.shape).astype(np.uint8)
                        
                        st.session_state.enhanced_img = enhanced
                        st.session_state.method_used = "Histogram Equalization"
            
            elif treatment == "Contrast Stretch":
                col1, col2 = st.columns(2)
                
                with col1:
                    black_point = st.slider("Black Point", 0, 100, int(img_np.min()),
                                           help="Values below this become 0")
                
                with col2:
                    white_point = st.slider("White Point", 155, 255, int(img_np.max()),
                                           help="Values above this become 255")
                
                if st.button("⚡ Apply Contrast Stretch", type="primary"):
                    with st.spinner("Stretching contrast..."):
                        # Linear contrast stretch
                        enhanced = (img_np - black_point) * (255 / (white_point - black_point))
                        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                        
                        st.session_state.enhanced_img = enhanced
                        st.session_state.method_used = f"Contrast Stretch ({black_point}→0, {white_point}→255)"
            
            else:  # Local Contrast
                col1, col2 = st.columns(2)
                
                with col1:
                    local_size = st.slider("Local Region Size", 5, 21, 11, step=2,
                                          help="Size of neighborhood to analyze")
                
                with col2:
                    contrast_boost = st.slider("Contrast Boost", 1.0, 3.0, 1.5, step=0.1,
                                              help="How much to enhance local contrast")
                
                if st.button("🔍 Apply Local Contrast", type="primary"):
                    with st.spinner("Enhancing local contrast..."):
                        enhanced = np.zeros_like(img_np, dtype=float)
                        half_size = local_size // 2
                        
                        for i in range(height):
                            for j in range(width):
                                # Get local region
                                i_start = max(0, i - half_size)
                                i_end = min(height, i + half_size + 1)
                                j_start = max(0, j - half_size)
                                j_end = min(width, j + half_size + 1)
                                
                                region = img_np[i_start:i_end, j_start:j_end]
                                local_mean = region.mean()
                                local_std = region.std()
                                
                                # Enhance based on local statistics
                                if local_std > 0:
                                    enhanced[i, j] = local_mean + contrast_boost * (img_np[i, j] - local_mean)
                                else:
                                    enhanced[i, j] = img_np[i, j]
                        
                        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                        
                        st.session_state.enhanced_img = enhanced
                        st.session_state.method_used = f"Local Contrast (Size={local_size}, Boost={contrast_boost})"
        
        # Show treatment applied message
        if 'enhanced_img' in st.session_state:
            st.success(f"✅ **Treatment applied!** Method: {st.session_state.method_used}")
            
            # Quick preview
            st.markdown("### 👀 Quick Preview of Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(np.clip(img_np, 0, 255).astype(np.uint8), 
                        caption="Before Treatment", 
                        use_container_width=True, channels="L")
            
            with col2:
                st.image(st.session_state.enhanced_img, 
                        caption="After Treatment", 
                        use_container_width=True, channels="L")
            
            st.info("**Next:** Go to Tab 4 for detailed comparison and analysis! →")
    
    # ==================== TAB 4: COMPARE RESULTS ====================
    with tab4:
        st.header("📊 Compare and Evaluate Results")
        
        st.markdown("""
        ### Time for Results Review! 📋
        
        Let's analyze how effective our treatment was and compare different approaches.
        """)
        
        # Check if we have enhanced image
        if 'enhanced_img' not in st.session_state:
            st.warning("⚠️ **No treatment applied yet!** Please go to Tab 3 and apply a treatment first.")
            st.info("**Quick tip:** Choose a patient and click one of the treatment buttons!")
        else:
            # Get current image info
            if 'treatment_select' in st.session_state:
                treatment_text = st.session_state.treatment_select
                if "Patient 1" in treatment_text:
                    img_np = images["image1"]
                    problem = "Blurry"
                elif "Patient 2" in treatment_text:
                    img_np = images["image2"]
                    problem = "Noisy"
                else:
                    img_np = images["image3"]
                    problem = "Low Contrast"
            else:
                # Default to image1 if not set
                img_np = images["image1"]
                problem = "Blurry"
            
            enhanced = st.session_state.enhanced_img
            method = st.session_state.method_used
            
            height, width = img_np.shape
            
            st.markdown(f"### 📋 Case: {problem} | Treatment: {method}")
            
            # Show before/after comparison
            st.markdown("---")
            st.markdown("### 👁️ Visual Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                before_display = np.clip(img_np, 0, 255).astype(np.uint8)
                st.image(before_display, caption="🟥 BEFORE Treatment", 
                        use_container_width=True, channels="L")
            
            with col2:
                st.image(enhanced, caption="🟩 AFTER Treatment", 
                        use_container_width=True, channels="L")
            
            # Side-by-side composite
            st.markdown("---")
            st.markdown("### 🔄 Side-by-Side View")
            
            composite = np.concatenate([before_display, enhanced], axis=1)
            st.image(composite, caption="Left: Before | Right: After", 
                    use_container_width=True, channels="L")
            
            # Statistical comparison
            st.markdown("---")
            st.markdown("### 📈 Statistical Analysis")
            
            # Calculate statistics
            before_min, before_max = img_np.min(), img_np.max()
            before_mean, before_std = img_np.mean(), img_np.std()
            before_range = before_max - before_min
            
            after_min, after_max = enhanced.min(), enhanced.max()
            after_mean, after_std = enhanced.mean(), enhanced.std()
            after_range = after_max - after_min
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Brightness Range", 
                         f"{before_range:.0f} → {after_range:.0f}",
                         f"{after_range-before_range:+.0f}")
            
            with col2:
                st.metric("Contrast (Std Dev)", 
                         f"{before_std:.1f} → {after_std:.1f}",
                         f"{after_std-before_std:+.1f}")
            
            with col3:
                st.metric("Average Brightness", 
                         f"{before_mean:.1f} → {after_mean:.1f}",
                         f"{after_mean-before_mean:+.1f}")
            
            with col4:
                # Calculate improvement percentage
                if problem == "Blurry":
                    # For blurry, want higher contrast
                    if before_std > 0:
                        improvement = (after_std - before_std) / before_std * 100
                    else:
                        improvement = 0
                elif problem == "Noisy":
                    # For noisy, want cleaner image (harder to quantify simply)
                    improvement = 0  # Would need noise estimation
                else:  # Low contrast
                    # Want higher dynamic range
                    if before_range > 0:
                        improvement = (after_range - before_range) / before_range * 100
                    else:
                        improvement = 0
                
                st.metric("Improvement", f"{improvement:.1f}%")
            
            # Quality metrics
            st.markdown("---")
            st.markdown("### 🎯 Quality Metrics")
            
            def calculate_metrics(original, enhanced):
                # Mean Squared Error
                mse = np.mean((original.astype(float) - enhanced.astype(float))**2)
                
                # Peak Signal-to-Noise Ratio
                psnr = 10 * np.log10((255**2) / (mse + 1e-10)) if mse > 0 else 99
                
                # Simplified SSIM
                C1 = (0.01 * 255)**2
                C2 = (0.03 * 255)**2
                
                mu_x = original.mean()
                mu_y = enhanced.mean()
                sigma_x = original.std()
                sigma_y = enhanced.std()
                sigma_xy = np.mean((original - mu_x) * (enhanced - mu_y))
                
                ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                       ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))
                
                return mse, psnr, ssim
            
            mse, psnr, ssim = calculate_metrics(img_np, enhanced)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("MSE (Error)", f"{mse:.1f}", 
                         delta_color="inverse",
                         help="Lower is better - measures difference from original")
            
            with col2:
                st.metric("PSNR (Quality)", f"{psnr:.1f} dB",
                         help="Higher is better - signal vs noise ratio")
            
            with col3:
                st.metric("SSIM (Similarity)", f"{ssim:.3f}",
                         help="Closer to 1 is better - structural similarity")
            
            # Problem-specific evaluation
            st.markdown("---")
            st.markdown(f"### 🎯 {problem}-Specific Evaluation")
            
            if problem == "Blurry":
                # Edge strength comparison
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
                
                def manual_conv(image, kernel):
                    h, w = image.shape
                    kh, kw = kernel.shape
                    pad_h, pad_w = kh//2, kw//2
                    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
                    output = np.zeros_like(image, dtype=float)
                    
                    for i in range(h):
                        for j in range(w):
                            region = padded[i:i+kh, j:j+kw]
                            output[i, j] = np.sum(region * kernel)
                    
                    return output
                
                before_gx = manual_conv(img_np, sobel_x)
                before_gy = manual_conv(img_np, sobel_y)
                before_edges = np.sqrt(before_gx**2 + before_gy**2)
                
                after_gx = manual_conv(enhanced.astype(float), sobel_x)
                after_gy = manual_conv(enhanced.astype(float), sobel_y)
                after_edges = np.sqrt(after_gx**2 + after_gy**2)
                
                edge_improvement = (after_edges.mean() - before_edges.mean()) / before_edges.mean() * 100
                
                st.info(f"""
                **Edge Enhancement Analysis:**
                - Average edge strength: {before_edges.mean():.1f} → {after_edges.mean():.1f}
                - Edge improvement: **{edge_improvement:.1f}%**
                - Strong edges (>50): {np.sum(before_edges > 50):,} → {np.sum(after_edges > 50):,} pixels
                
                **Verdict:** {'✅ Excellent sharpening!' if edge_improvement > 30 else '⚠️ Could be sharper' if edge_improvement > 10 else '❌ Needs stronger treatment'}
                """)
            
            elif problem == "Noisy":
                # Noise reduction evaluation
                # Simple noise estimation: difference from smoothed version
                smooth_kernel = np.ones((5,5), dtype=float) / 25
                
                def manual_conv(image, kernel):
                    h, w = image.shape
                    kh, kw = kernel.shape
                    pad_h, pad_w = kh//2, kw//2
                    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
                    output = np.zeros_like(image, dtype=float)
                    
                    for i in range(h):
                        for j in range(w):
                            region = padded[i:i+kh, j:j+kw]
                            output[i, j] = np.sum(region * kernel)
                    
                    return output
                
                before_smooth = manual_conv(img_np, smooth_kernel)
                before_noise = np.mean(np.abs(img_np - before_smooth))
                
                after_smooth = manual_conv(enhanced.astype(float), smooth_kernel)
                after_noise = np.mean(np.abs(enhanced.astype(float) - after_smooth))
                
                noise_reduction = (before_noise - after_noise) / before_noise * 100
                
                st.info(f"""
                **Noise Reduction Analysis:**
                - Estimated noise level: {before_noise:.1f} → {after_noise:.1f}
                - Noise reduction: **{noise_reduction:.1f}%**
                - Cleanliness: {'✅ Very clean' if after_noise < 5 else '⚠️ Acceptable' if after_noise < 15 else '❌ Still noisy'}
                
                **Verdict:** {'✅ Excellent denoising!' if noise_reduction > 50 else '⚠️ Good reduction' if noise_reduction > 25 else '❌ Needs stronger filtering'}
                """)
            
            else:  # Low Contrast
                # Contrast improvement
                contrast_improvement = (after_range - before_range) / before_range * 100
                std_improvement = (after_std - before_std) / before_std * 100
                
                # Histogram comparison
                before_hist, _ = np.histogram(img_np, 256, [0, 256])
                after_hist, _ = np.histogram(enhanced, 256, [0, 256])
                
                # Calculate histogram spread
                before_spread = np.sum(before_hist > 0)
                after_spread = np.sum(after_hist > 0)
                
                st.info(f"""
                **Contrast Enhancement Analysis:**
                - Dynamic range: {before_range:.0f} → {after_range:.0f} ({contrast_improvement:.1f}%)
                - Contrast (std): {before_std:.1f} → {after_std:.1f} ({std_improvement:.1f}%)
                - Histogram spread: {before_spread} → {after_spread} brightness levels used
                
                **Verdict:** {'✅ Excellent contrast!' if contrast_improvement > 50 else '⚠️ Good improvement' if contrast_improvement > 20 else '❌ Needs stronger enhancement'}
                """)
            
            # Zoom comparison
            st.markdown("---")
            st.markdown("### 🔍 Detailed Zoom Comparison")
            
            zoom_level = st.slider("Zoom Level", 2.0, 8.0, 4.0, step=0.5, key="zoom_compare")
            
            center_y, center_x = height // 2, width // 2
            zoom_size = int(100 / zoom_level)
            
            y_start = max(0, center_y - zoom_size)
            y_end = min(height, center_y + zoom_size)
            x_start = max(0, center_x - zoom_size)
            x_end = min(width, center_x + zoom_size)
            
            before_zoom = before_display[y_start:y_end, x_start:x_end]
            after_zoom = enhanced[y_start:y_end, x_start:x_end]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(before_zoom, caption=f"Before (Zoom {zoom_level:.1f}×)", 
                        use_container_width=True, channels="L")
            
            with col2:
                st.image(after_zoom, caption=f"After (Zoom {zoom_level:.1f}×)", 
                        use_container_width=True, channels="L")
            
            # Overall assessment
            st.markdown("---")
            st.markdown("### 🏆 Overall Assessment")
            
            # Simple scoring
            score = 0
            feedback = []
            
            if problem == "Blurry":
                if 'edge_improvement' in locals() and edge_improvement > 30:
                    score += 2
                    feedback.append("✅ Excellent edge restoration")
                elif 'edge_improvement' in locals() and edge_improvement > 10:
                    score += 1
                    feedback.append("⚠️ Moderate improvement")
                else:
                    feedback.append("❌ Needs more sharpening")
            
            elif problem == "Noisy":
                if 'noise_reduction' in locals() and noise_reduction > 50:
                    score += 2
                    feedback.append("✅ Excellent noise reduction")
                elif 'noise_reduction' in locals() and noise_reduction > 25:
                    score += 1
                    feedback.append("⚠️ Good noise reduction")
                else:
                    feedback.append("❌ Needs more smoothing")
            
            else:  # Low Contrast
                if 'contrast_improvement' in locals() and contrast_improvement > 50:
                    score += 2
                    feedback.append("✅ Excellent contrast boost")
                elif 'contrast_improvement' in locals() and contrast_improvement > 20:
                    score += 1
                    feedback.append("⚠️ Good contrast improvement")
                else:
                    feedback.append("❌ Needs more contrast")
            
            # Check for artifacts (oversharpening halos, etc.)
            if psnr > 30 and ssim > 0.8:
                score += 1
                feedback.append("✅ Good quality preservation")
            elif psnr > 25:
                feedback.append("⚠️ Some quality loss")
            else:
                score -= 1
                feedback.append("❌ Significant quality loss")
            
            # Display score
            st.markdown(f"**Treatment Score: {score}/3**")
            
            for item in feedback:
                st.write(item)
            
            if score >= 2:
                st.success("🎉 **Great job!** Your treatment was very effective!")
                st.balloons()
            elif score >= 1:
                st.warning("👍 **Good effort!** The treatment helped, but could be better.")
            else:
                st.error("😕 **Needs improvement.** Try a different treatment approach.")
    
    # ==================== TAB 5: BECOME AN EXPERT ====================
    with tab5:
        st.header("🎓 Become an Image Enhancement Expert")
        
        st.markdown("""
        ### Congratulations on Completing the Challenge! 🏆
        
        You've learned how to diagnose and treat common image problems.
        Now let's solidify your expertise!
        """)
        
        # Summary of what was learned
        st.markdown("---")
        st.markdown("### 📚 What You've Learned")
        
        learnings = {
            "🔍 Diagnosis": "How to identify blurry, noisy, and low-contrast images",
            "🩺 Assessment": "Using statistical analysis and visual tests",
            "💊 Treatment": "Applying appropriate enhancement techniques",
            "📊 Evaluation": "Measuring improvement with metrics",
            "🎯 Strategy": "Choosing the right tool for each problem"
        }
        
        for key, value in learnings.items():
            st.markdown(f"**{key}:** {value}")
        
        # Expert tips
        st.markdown("---")
        st.markdown("### 💡 Expert Tips & Tricks")
        
        tips = st.tabs(["General Principles", "Blurry Images", "Noisy Images", "Low Contrast"])
        
        with tips[0]:
            st.markdown("""
            **Golden Rules of Image Enhancement:**
            
            1. **Start mild, then adjust** - Better to under-enhance than over-enhance
            2. **Check at 100% zoom** - Always inspect details up close
            3. **Compare side-by-side** - Our eyes adapt quickly to changes
            4. **Use multiple methods** - Sometimes combination works best
            5. **Know when to stop** - Too much enhancement creates artifacts
            
            **Pro Workflow:**
            1. Diagnose the primary problem
            2. Apply targeted enhancement  
            3. Check for secondary issues
            4. Make fine adjustments
            5. Final quality check
            """)
        
        with tips[1]:
            st.markdown("""
            **Blurry Images - Pro Techniques:**
            
            **For Mild Blur:**
            - Unsharp masking with small blur size (3-5)
            - Amount: 0.8-1.2 for natural look
            
            **For Heavy Blur:**
            - Multiple passes of mild sharpening
            - Combine with edge enhancement
            - Consider "Smart Sharpening" algorithms
            
            **Avoid:**
            - Too much sharpening (creates halos)
            - Sharpening noisy images (amplifies noise)
            - Ignoring the cause of blur
            
            **Advanced Tip:** Sometimes blur is intentional (portrait mode). 
            Don't sharpen everything!
            """)
        
        with tips[2]:
            st.markdown("""
            **Noisy Images - Pro Techniques:**
            
            **For Mild Noise:**
            - Gaussian blur with sigma 1.0-1.5
            - Small kernel (3×3 or 5×5)
            
            **For Heavy Noise:**
            - Bilateral or edge-preserving filters
            - Consider wavelet-based denoising
            - Multiple mild passes instead of one strong
            
            **For Preserving Details:**
            - Use edge detection to protect edges
            - Apply stronger smoothing in flat areas
            - Keep some texture for natural look
            
            **Advanced Tip:** Noise isn't always bad! 
            Film grain and texture can add character.
            """)
        
        with tips[3]:
            st.markdown("""
            **Low Contrast - Pro Techniques:**
            
            **For Overall Low Contrast:**
            - Histogram equalization (automatic)
            - Contrast stretching (manual control)
            - Levels/curves adjustment
            
            **For Local Contrast Issues:**
            - Adaptive histogram equalization
            - Local contrast enhancement
            - Dodge & burn techniques
            
            **For Specific Tonal Ranges:**
            - Shadow/highlight recovery
            - Midtone contrast boost
            - Selective color adjustment
            
            **Advanced Tip:** Preserve highlight and shadow detail.
            Clipped whites/blacks lose information forever!
            """)
        
        # Practice challenge
        st.markdown("---")
        st.markdown("### 🏁 Final Expert Challenge")
        
        challenge = st.selectbox(
            "Test your expertise with this scenario:",
            ["No challenge selected", 
             "Old faded photo with both blur AND low contrast",
             "Night photo with noise AND blur", 
             "Backlit subject with dark face and bright background"]
        )
        
        if challenge != "No challenge selected":
            st.markdown(f"**Scenario:** {challenge}")
            
            if "Old faded photo" in challenge:
                st.markdown("""
                **Expert Analysis:**
                - Primary issue: Low contrast (faded)
                - Secondary issue: Blur (age/scanning)
                - Challenge: Sharpening might amplify scanning artifacts
                
                **Recommended Approach:**
                1. First: Contrast enhancement (histogram equalization)
                2. Then: Mild sharpening (unsharp masking, small amount)
                3. Finally: Noise reduction if needed
                4. Consider: Specialized "photo restoration" algorithms
                
                **Why this order?** 
                Enhancing contrast first reveals details, 
                then sharpening can work on those revealed details.
                """)
            
            elif "Night photo" in challenge:
                st.markdown("""
                **Expert Analysis:**
                - Primary issue: High noise (low light)
                - Secondary issue: Blur (camera shake)
                - Challenge: Denoising without losing details
                
                **Recommended Approach:**
                1. First: Edge-preserving noise reduction
                2. Then: Very mild sharpening
                3. Consider: Multiple exposure blending if available
                4. Alternative: AI-based low-light enhancement
                
                **Pro Tip:** In low light, some blur is inevitable.
                Focus on noise reduction and accept mild softness.
                """)
            
            else:  # Backlit subject
                st.markdown("""
                **Expert Analysis:**
                - Primary issue: Extreme dynamic range
                - Challenge: Brightening dark areas without overexposing bright areas
                
                **Recommended Approach:**
                1. First: Local contrast enhancement
                2. Then: Selective brightening of shadows
                3. Consider: HDR techniques if multiple exposures available
                4. Alternative: Fill light in post-processing
                
                **Advanced Technique:** Use masks to selectively enhance 
                different areas of the image.
                """)
            
            # Self-assessment
            st.markdown("---")
            st.markdown("### 🤔 How Would You Handle It?")
            
            user_approach = st.text_area("Describe your approach:", 
                                        placeholder="E.g., 'I would first apply... then use... because...'")
            
            if user_approach:
                st.success("✅ **Great thinking!** Every expert develops their own workflow.")
        
        # Next steps
        st.markdown("---")
        st.markdown("### 🚀 Continue Your Learning Journey")
        
        next_steps = st.tabs(["Advanced Topics", "Real Projects", "Career Paths"])
        
        with next_steps[0]:
            st.markdown("""
            **Advanced Image Processing Topics:**
            
            **1. Frequency Domain Processing**
            - Fourier transforms
            - Wavelet analysis
            - Frequency-based filtering
            
            **2. Machine Learning Approaches**
            - Deep learning for image enhancement
            - Neural networks for restoration
            - AI-based super-resolution
            
            **3. Color Science**
            - Color spaces and management
            - Color grading techniques
            - Color correction workflows
            
            **4. Computational Photography**
            - HDR imaging
            - Panorama stitching
            - Focus stacking
            """)
        
        with next_steps[1]:
            st.markdown("""
            **Real-World Projects to Try:**
            
            **Beginner:**
            - Restore old family photos
            - Enhance smartphone photos
            - Create artistic filters
            
            **Intermediate:**
            - Build a simple photo editor
            - Process batch images automatically
            - Develop custom filters
            
            **Advanced:**
            - Medical image enhancement
            - Satellite image analysis
            - Real-time video enhancement
            
            **Tools to Learn:**
            - OpenCV (computer vision library)
            - PIL/Pillow (Python imaging)
            - Photoshop/GIMP (professional editors)
            """)
        
        with next_steps[2]:
            st.markdown("""
            **Career Paths Using These Skills:**
            
            **1. Photo Editor/Retoucher**
            - Work with photographers
            - Magazine/advertising work
            - Freelance opportunities
            
            **2. Medical Imaging Specialist**
            - Hospital/clinical work
            - Research positions
            - Medical device companies
            
            **3. Computer Vision Engineer**
            - Tech companies (Google, Facebook, etc.)
            - Autonomous vehicles
            - Robotics companies
            
            **4. Forensic Image Analyst**
            - Law enforcement
            - Legal services
            - Security companies
            
            **5. Remote Sensing Analyst**
            - Satellite companies
            - Environmental monitoring
            - Agriculture technology
            """)
        
        # Final encouragement
        st.markdown("---")
        st.success("""
        🎉 **Congratulations! You've completed the Image Enhancement Challenge!**
        
        **You now have:**
        - ✅ Diagnostic skills for image problems
        - ✅ Practical enhancement techniques
        - ✅ Evaluation methods for results
        - ✅ Expert-level understanding
        
        **Remember:** The best enhancement is often invisible.
        Your goal should be to make images better while keeping them natural.
        
        Keep practicing, keep learning, and most importantly - keep creating! 🚀
        
        **Final Thought:** Every image tells a story. Your job as an enhancer 
        is to help that story shine through more clearly! ✨
        """)