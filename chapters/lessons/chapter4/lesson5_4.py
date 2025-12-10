import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import os

def app():
    st.title("🚀 Real Applications: Frequency Domain in Action")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Complete Lab Tasks", 
        "2️⃣ Medical Imaging", 
        "3️⃣ Satellite & Remote Sensing",
        "4️⃣ Industrial Inspection",
        "5️⃣ Creative Applications"
    ])
    
    # ==================== TAB 1: COMPLETE LAB TASKS ====================
    with tab1:
        st.header("✅ Complete Your Lab Tasks")
        
        st.markdown("""
        Let's complete all 5 tasks from your lab sheet using frequency domain filtering!
        This tab walks you through the complete process step by step.
        """)
        
        # Task 1: Load and Transform
        st.markdown("---")
        st.markdown("### 📋 **Task 1: Load and Transform**")
        
        # Image selection
        image_dir = "public/lab5"
        available_images = []
        
        if os.path.exists(image_dir):
            available_images = [f for f in os.listdir(image_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if available_images:
            st.success(f"✅ Found {len(available_images)} images in lab5 folder")
            
            # Let user select an image for the lab
            lab_image = st.selectbox("Select image for lab tasks:", available_images)
            img_path = os.path.join(image_dir, lab_image)
            img = Image.open(img_path).convert('L')
            img_np = np.array(img).astype(float)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Step 1.1: Load Image")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img_np, cmap='gray')
                ax.set_title(f"Loaded: {lab_image}")
                ax.axis('off')
                st.pyplot(fig)
                st.metric("Image Size", f"{img_np.shape[1]}×{img_np.shape[0]}")
                st.metric("Data Type", str(img_np.dtype))
            
            with col2:
                st.markdown("#### Step 1.2: Apply 2D DFT")
                # Calculate DFT
                dft = fft2(img_np)
                dft_shifted = fftshift(dft)
                magnitude = np.abs(dft_shifted)
                magnitude_log = 20 * np.log(magnitude + 1)
                
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(magnitude_log, cmap='gray')
                ax.set_title("Magnitude Spectrum")
                ax.axis('off')
                st.pyplot(fig)
                
                # Spectrum analysis
                center_strength = magnitude_log[magnitude_log.shape[0]//2, magnitude_log.shape[1]//2]
                edge_strength = magnitude_log[0, 0].mean()
                st.metric("Center Brightness", f"{center_strength:.1f}")
                st.metric("Edge Brightness", f"{edge_strength:.1f}")
            
            st.markdown("""
            **✅ Task 1 Complete:** Image loaded and transformed to frequency domain!
            
            **Interpretation:**
            - Bright center = strong low frequencies (average brightness)
            - Pattern from center = frequency distribution
            - Bright spots = strong frequencies in those directions
            """)
        
        else:
            st.error("❌ No images found in lab5 folder!")
            st.info("Please add your teacher's 8 images to `public/lab5/`")
            return
        
        # Task 2: Create Filters
        st.markdown("---")
        st.markdown("### 🎛️ **Task 2: Create Filters**")
        
        st.markdown("""
        Design filters to manipulate specific frequencies.
        Choose what you want to achieve with your image.
        """)
        
        task2_cols = st.columns(2)
        
        with task2_cols[0]:
            filter_purpose = st.selectbox(
                "What do you want to do with the image?",
                ["Remove noise", "Sharpen edges", "Smooth/blur", 
                 "Isolate textures", "Remove patterns", "Custom effect"]
            )
        
        with task2_cols[1]:
            if filter_purpose == "Remove noise":
                filter_type = "Notch Filter"
                filter_desc = "Removes specific noise frequencies"
            elif filter_purpose == "Sharpen edges":
                filter_type = "High-pass Filter"
                filter_desc = "Enhances edges and details"
            elif filter_purpose == "Smooth/blur":
                filter_type = "Low-pass Filter"
                filter_desc = "Smooths and blurs image"
            elif filter_purpose == "Isolate textures":
                filter_type = "Band-pass Filter"
                filter_desc = "Isolates texture frequencies"
            elif filter_purpose == "Remove patterns":
                filter_type = "Band-stop Filter"
                filter_desc = "Removes specific pattern frequencies"
            else:
                filter_type = st.selectbox("Choose filter type:", 
                                         ["Low-pass", "High-pass", "Band-pass", "Band-stop", "Notch"])
                filter_desc = "Custom filter application"
        
        # Create appropriate filter
        rows, cols = img_np.shape
        center_y, center_x = rows // 2, cols // 2
        
        y, x = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist = dist_from_center / max_dist
        
        if filter_type == "Low-pass Filter":
            cutoff = st.slider("Cutoff frequency", 0.01, 0.5, 0.2, 0.01)
            filter_mask = np.ones((rows, cols))
            filter_mask[normalized_dist > cutoff] = 0
        
        elif filter_type == "High-pass Filter":
            cutoff = st.slider("Cutoff frequency", 0.01, 0.5, 0.1, 0.01)
            strength = st.slider("Sharpening strength", 1.0, 5.0, 2.0, 0.5)
            filter_mask = np.zeros((rows, cols))
            filter_mask[normalized_dist > cutoff] = strength
        
        elif filter_type == "Band-pass Filter":
            low_cut = st.slider("Low cutoff", 0.01, 0.4, 0.1, 0.01)
            high_cut = st.slider("High cutoff", low_cut + 0.01, 0.5, 0.3, 0.01)
            filter_mask = np.zeros((rows, cols))
            filter_mask[(normalized_dist >= low_cut) & (normalized_dist <= high_cut)] = 1
        
        elif filter_type == "Band-stop Filter":
            low_cut = st.slider("Low cutoff", 0.01, 0.4, 0.15, 0.01)
            high_cut = st.slider("High cutoff", low_cut + 0.01, 0.5, 0.25, 0.01)
            filter_mask = np.ones((rows, cols))
            filter_mask[(normalized_dist >= low_cut) & (normalized_dist <= high_cut)] = 0
        
        else:  # Notch Filter
            notch_u = st.slider("Horizontal position", -50, 50, 30, 1)
            notch_v = st.slider("Vertical position", -50, 50, 30, 1)
            notch_size = st.slider("Notch size", 1, 20, 5, 1)
            
            filter_mask = np.ones((rows, cols))
            notch_positions = [
                (center_x + notch_u, center_y + notch_v),
                (center_x - notch_u, center_y - notch_v),
                (center_x + notch_u, center_y - notch_v),
                (center_x - notch_u, center_y + notch_v)
            ]
            for nx, ny in notch_positions:
                notch_dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                filter_mask[notch_dist < notch_size] = 0
        
        # Display filter
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Step 2.1: Analyze Spectrum")
            # Show spectrum with filter overlay
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(magnitude_log, cmap='gray', alpha=0.7)
            ax.imshow(filter_mask, cmap='Reds', alpha=0.3)
            ax.set_title(f"Filter: {filter_type}")
            ax.axis('off')
            st.pyplot(fig)
            st.caption("Red areas = frequencies kept, Gray = original spectrum")
        
        with col2:
            st.markdown("#### Step 2.2: Filter Mask")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(filter_mask, cmap='gray')
            ax.set_title("Filter Mask")
            ax.axis('off')
            st.pyplot(fig)
            
            # Filter statistics
            kept_percent = np.sum(filter_mask > 0.5) / filter_mask.size * 100
            st.metric("Frequencies Kept", f"{kept_percent:.1f}%")
            st.info(f"**Purpose:** {filter_desc}")
        
        st.markdown(f"""
        **✅ Task 2 Complete:** {filter_type} created for {filter_purpose}!
        
        **Filter Design:**
        - Type: {filter_type}
        - Purpose: {filter_purpose}
        - Frequencies kept: {kept_percent:.1f}%
        - Expected effect: {filter_desc}
        """)
        
        # Task 3: Apply Filters
        st.markdown("---")
        st.markdown("### ⚡ **Task 3: Apply Filters**")
        
        st.markdown("Multiply the frequency spectrum by the filter mask:")
        
        # Apply filter
        filtered_dft = dft_shifted * filter_mask
        
        # Calculate filtered spectrum
        filtered_magnitude = np.abs(filtered_dft)
        filtered_magnitude_log = 20 * np.log(filtered_magnitude + 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Step 3.1: Original Spectrum")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(magnitude_log, cmap='gray')
            ax.set_title("Before Filtering")
            ax.axis('off')
            st.pyplot(fig)
            
            orig_energy = np.sum(magnitude_log)
            st.metric("Spectral Energy", f"{orig_energy:.0f}")
        
        with col2:
            st.markdown("#### Step 3.2: Filtered Spectrum")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(filtered_magnitude_log, cmap='gray')
            ax.set_title("After Filtering")
            ax.axis('off')
            st.pyplot(fig)
            
            filtered_energy = np.sum(filtered_magnitude_log)
            energy_change = (filtered_energy - orig_energy) / orig_energy * 100
            st.metric("Spectral Energy", f"{filtered_energy:.0f}", delta=f"{energy_change:.1f}%")
        
        st.markdown("""
        **✅ Task 3 Complete:** Filter applied to frequency spectrum!
        
        **What happened:**
        - Spectrum multiplied by filter mask
        - Some frequencies removed (black in filter mask)
        - Some frequencies kept/boosted (white in filter mask)
        - Spectral energy changed by filtering
        """)
        
        # Task 4: Reconstruct Image
        st.markdown("---")
        st.markdown("### 🔄 **Task 4: Reconstruct Image**")
        
        st.markdown("Convert back to spatial domain using Inverse DFT:")
        
        # Perform inverse transform
        filtered_dft_ishift = ifftshift(filtered_dft)
        reconstructed = np.real(ifft2(filtered_dft_ishift))
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        # Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img_np, cmap='gray')
            ax.set_title("Before")
            ax.axis('off')
            st.pyplot(fig)
            
            # Original metrics
            orig_std = img_np.std()
            orig_mean = img_np.mean()
            st.metric("Contrast (std)", f"{orig_std:.1f}")
            st.metric("Brightness (mean)", f"{orig_mean:.1f}")
        
        with col2:
            st.markdown("#### Reconstructed Image")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(reconstructed, cmap='gray')
            ax.set_title("After Filtering")
            ax.axis('off')
            st.pyplot(fig)
            
            # Filtered metrics
            filtered_std = reconstructed.std()
            filtered_mean = reconstructed.mean()
            std_change = (filtered_std - orig_std) / orig_std * 100
            mean_change = (filtered_mean - orig_mean) / orig_mean * 100
            
            st.metric("Contrast (std)", f"{filtered_std:.1f}", delta=f"{std_change:.1f}%")
            st.metric("Brightness (mean)", f"{filtered_mean:.1f}", delta=f"{mean_change:.1f}%")
        
        # Comparison analysis
        st.markdown("#### 📊 Comparison Analysis")
        
        analysis_cols = st.columns(4)
        
        with analysis_cols[0]:
            diff_image = np.abs(img_np.astype(float) - reconstructed.astype(float))
            avg_change = diff_image.mean()
            st.metric("Avg Pixel Change", f"{avg_change:.1f}")
        
        with analysis_cols[1]:
            if filtered_std < orig_std * 0.7:
                effect = "✅ Smoother/Blurrier"
            elif filtered_std > orig_std * 1.3:
                effect = "✅ Sharper/More detailed"
            else:
                effect = "⚖️ Similar detail"
            st.metric("Effect", effect)
        
        with analysis_cols[2]:
            mse = np.mean((img_np - reconstructed) ** 2)
            st.metric("MSE", f"{mse:.1f}")
        
        with analysis_cols[3]:
            psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')
            st.metric("PSNR", f"{psnr:.1f} dB")
        
        # Side-by-side comparison
        st.markdown("#### 🔄 Side-by-Side Comparison")
        
        comparison_value = st.slider("Comparison slider", 0, 100, 50, 1,
                                   help="Slide to compare original (left) and filtered (right)")
        
        comparison_img = np.hstack([
            img_np[:, :img_np.shape[1]//2],
            reconstructed[:, reconstructed.shape[1]//2:]
        ])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        split_point = int(comparison_img.shape[1] * comparison_value / 100)
        
        # Add visual divider
        comparison_display = comparison_img.copy()
        if 0 < split_point < comparison_img.shape[1]:
            comparison_display[:, split_point-2:split_point+2] = 255
        
        ax.imshow(comparison_display, cmap='gray')
        ax.axvline(x=split_point, color='red', linestyle='--', linewidth=2)
        ax.text(split_point + 10, 30, "FILTERED", color='red', fontweight='bold')
        ax.text(split_point - 80, 30, "ORIGINAL", color='red', fontweight='bold')
        ax.axis('off')
        ax.set_title("Slide to compare: Original ← → Filtered")
        st.pyplot(fig)
        
        st.markdown(f"""
        **✅ Task 4 Complete:** Image reconstructed from filtered frequencies!
        
        **Results:**
        - Image successfully reconstructed using inverse DFT
        - Filter effect: {effect.lower()}
        - Average pixel change: {avg_change:.1f} units
        - {'✅ Improvement achieved!' if abs(std_change) > 10 else '⚖️ Minimal change'}
        """)
        
        # Task 5: Report & Interpretation
        st.markdown("---")
        st.markdown("### 📝 **Task 5: Report & Interpretation**")
        
        st.markdown("Complete your lab report by answering these questions:")
        
        with st.expander("📋 **Complete Experiment Report**", expanded=True):
            # Display all results
            st.markdown("#### Experimental Results")
            
            results_cols = st.columns(4)
            with results_cols[0]:
                st.image(img_np, caption="1. Original Image", use_container_width=True)
            with results_cols[1]:
                st.image(magnitude_log, caption="2. Magnitude Spectrum", use_container_width=True)
            with results_cols[2]:
                st.image(filter_mask, caption="3. Filter Mask", use_container_width=True)
            with results_cols[3]:
                st.image(reconstructed, caption="4. Filtered Image", use_container_width=True)
            
            # Questions
            st.markdown("---")
            st.markdown("#### ❓ **Interpretation Questions**")
            
            q1 = st.text_area(
                "1. **Interpret the spectrum:** What does the original spectrum tell you about the image's frequency content?",
                placeholder="Describe the bright spots, patterns, and what they mean...",
                height=100
            )
            
            q2 = st.text_area(
                f"2. **Filter choice:** Why did you choose the {filter_type}? What specific frequencies were you targeting?",
                placeholder=f"Explain why {filter_type} was appropriate for {filter_purpose}...",
                height=100
            )
            
            q3 = st.text_area(
                "3. **Effect analysis:** Did the filter achieve its purpose? What changed in the image?",
                placeholder="Describe the visual changes and whether they match expectations...",
                height=100
            )
            
            q4 = st.text_area(
                "4. **Success evaluation:** Did the filter improve the image or remove important details?",
                placeholder="Evaluate whether the trade-off was worthwhile...",
                height=100
            )
            
            # Auto-analysis helper
            if st.button("🔄 Generate Auto-Analysis"):
                # Analyze results
                if filtered_std < orig_std * 0.7:
                    effect_desc = "successfully smoothed/blurred the image"
                    detail_tradeoff = "some fine details were lost but noise was reduced"
                elif filtered_std > orig_std * 1.3:
                    effect_desc = "successfully sharpened/enhanced edges"
                    detail_tradeoff = "edges are clearer but noise may be amplified"
                else:
                    effect_desc = "had minimal effect on overall detail"
                    detail_tradeoff = "balance maintained between enhancement and preservation"
                
                # Center vs edge analysis
                center_before = magnitude_log[magnitude_log.shape[0]//2, magnitude_log.shape[1]//2]
                center_after = filtered_magnitude_log[filtered_magnitude_log.shape[0]//2, 
                                                      filtered_magnitude_log.shape[1]//2]
                
                if center_after < center_before * 0.8:
                    freq_effect = "Low frequencies were reduced (image may be darker/less smooth)"
                elif center_after > center_before * 1.2:
                    freq_effect = "Low frequencies were boosted (image may be brighter/smoother)"
                else:
                    freq_effect = "Low frequencies were mostly preserved"
                
                st.info(f"""
                **Auto-Analysis:**
                
                **Spectrum Interpretation:** {center_before:.1f} dB center brightness indicates {'strong' if center_before > 10 else 'moderate'} low frequencies.
                
                **Filter Effectiveness:** The {filter_type} {effect_desc}. {freq_effect}
                
                **Detail Trade-off:** {detail_tradeoff}. The average pixel changed by {avg_change:.1f} units.
                
                **Recommendation:** {'Good improvement for the intended purpose' if abs(std_change) > 15 else 'Consider adjusting filter parameters for stronger effect'}.
                """)
            
            # Save report
            st.markdown("---")
            st.markdown("#### 💾 **Save Your Report**")
            
            report_name = st.text_input("Report name:", f"lab_report_{lab_image.split('.')[0]}")
            
            if st.button("📥 Download Complete Report"):
                # Create a summary report
                report = f"""
                ===== FREQUENCY DOMAIN FILTERING LAB REPORT =====
                
                Image: {lab_image}
                Date: {st.session_state.get('report_date', 'Current date')}
                
                === TASK 1: LOAD AND TRANSFORM ===
                - Image size: {img_np.shape[1]}×{img_np.shape[0]}
                - Original contrast (std): {orig_std:.1f}
                - Original brightness (mean): {orig_mean:.1f}
                - Spectrum center brightness: {center_before:.1f} dB
                
                === TASK 2: CREATE FILTERS ===
                - Filter type: {filter_type}
                - Purpose: {filter_purpose}
                - Frequencies kept: {kept_percent:.1f}%
                - Cutoff parameters: {locals().get('cutoff', 'N/A')}
                
                === TASK 3: APPLY FILTERS ===
                - Original spectral energy: {orig_energy:.0f}
                - Filtered spectral energy: {filtered_energy:.0f}
                - Energy change: {energy_change:.1f}%
                
                === TASK 4: RECONSTRUCT IMAGE ===
                - Filtered contrast (std): {filtered_std:.1f}
                - Filtered brightness (mean): {filtered_mean:.1f}
                - Contrast change: {std_change:.1f}%
                - Brightness change: {mean_change:.1f}%
                - Average pixel change: {avg_change:.1f}
                - MSE: {mse:.1f}
                - PSNR: {psnr:.1f} dB
                
                === TASK 5: INTERPRETATION ===
                1. Spectrum interpretation: {q1[:50]}...
                2. Filter choice: {q2[:50]}...
                3. Effect analysis: {q3[:50]}...
                4. Success evaluation: {q4[:50]}...
                
                === CONCLUSION ===
                The {filter_type} {'successfully' if abs(std_change) > 10 else 'partially'} achieved {filter_purpose}.
                {'Significant improvement noted.' if abs(std_change) > 15 else 'Minimal visual change.'}
                """
                
                # Create download button
                st.download_button(
                    label="📥 Download Report as Text",
                    data=report,
                    file_name=f"{report_name}.txt",
                    mime="text/plain"
                )
        
        st.success("""
        🎉 **LAB COMPLETE!** All 5 tasks finished!
        
        **You have successfully:**
        1. ✅ Loaded image and computed DFT
        2. ✅ Designed frequency domain filter
        3. ✅ Applied filter to spectrum
        4. ✅ Reconstructed filtered image
        5. ✅ Analyzed and reported results
        
        **Next:** Explore real-world applications in other tabs! 🚀
        """)
    
    # ==================== TAB 2: MEDICAL IMAGING ====================
    with tab2:
        st.header("🏥 Medical Imaging Applications")
        
        st.markdown("""
        Frequency domain filtering is crucial in medical imaging for:
        - **Noise reduction** in low-signal images
        - **Feature enhancement** for diagnosis
        - **Artifact removal** from scanning
        - **Pattern recognition** in tissues
        """)
        
        # Create medical image examples
        st.markdown("### 🩻 Medical Image Processing Examples")
        
        # Create simulated medical images
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### X-ray Enhancement")
            xray_img = create_xray_image()
            st.image(xray_img, caption="Original X-ray", use_container_width=True)
            
            # Process X-ray
            xray_filtered = enhance_medical_image(xray_img, "xray")
            st.image(xray_filtered, caption="Enhanced (High-pass)", use_container_width=True)
            st.caption("**Effect:** Edge enhancement for fracture detection")
        
        with col2:
            st.markdown("#### MRI Noise Reduction")
            mri_img = create_mri_image()
            st.image(mri_img, caption="Noisy MRI", use_container_width=True)
            
            # Process MRI
            mri_filtered = enhance_medical_image(mri_img, "mri")
            st.image(mri_filtered, caption="Denoised (Low-pass)", use_container_width=True)
            st.caption("**Effect:** Noise reduction for clearer tissue boundaries")
        
        with col3:
            st.markdown("#### Ultrasound Artifact Removal")
            ultrasound_img = create_ultrasound_image()
            st.image(ultrasound_img, caption="Ultrasound with artifacts", use_container_width=True)
            
            # Process ultrasound
            ultrasound_filtered = enhance_medical_image(ultrasound_img, "ultrasound")
            st.image(ultrasound_filtered, caption="Cleaned (Notch filter)", use_container_width=True)
            st.caption("**Effect:** Speckle noise and artifact reduction")
        
        st.markdown("---")
        
        # Interactive medical image processing
        st.markdown("### 🔬 Interactive Medical Image Processor")
        
        medical_case = st.selectbox(
            "Select medical imaging case:",
            ["Brain MRI - Tumor Detection", 
             "Chest X-ray - Pneumonia Detection",
             "Retinal Scan - Blood Vessel Analysis",
             "Dental X-ray - Cavity Detection"]
        )
        
        # Load or create appropriate image
        if "Brain" in medical_case:
            medical_img = create_brain_mri()
            target = "tumor edges"
            filter_type = "High-pass for edge enhancement"
        elif "Chest" in medical_case:
            medical_img = create_chest_xray()
            target = "lung texture patterns"
            filter_type = "Band-pass for texture isolation"
        elif "Retinal" in medical_case:
            medical_img = create_retinal_scan()
            target = "blood vessel network"
            filter_type = "High-pass for vessel enhancement"
        else:
            medical_img = create_dental_xray()
            target = "tooth structure and cavities"
            filter_type = "Multi-band for structure enhancement"
        
        # Process with frequency filters
        processed_img = process_medical_image(medical_img, medical_case)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Medical Image")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(medical_img, cmap='gray')
            ax.set_title(f"{medical_case}")
            ax.axis('off')
            st.pyplot(fig)
            
            # Original metrics
            orig_snr = calculate_snr(medical_img)
            st.metric("Estimated SNR", f"{orig_snr:.1f} dB")
        
        with col2:
            st.markdown("#### Processed for Diagnosis")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(processed_img, cmap='gray')
            ax.set_title(f"Enhanced for {target}")
            ax.axis('off')
            st.pyplot(fig)
            
            # Processed metrics
            processed_snr = calculate_snr(processed_img)
            snr_improvement = processed_snr - orig_snr
            st.metric("Estimated SNR", f"{processed_snr:.1f} dB", delta=f"+{snr_improvement:.1f} dB")
        
        # Show frequency analysis
        st.markdown("#### 📊 Frequency Domain Analysis")
        
        # Calculate spectra
        fft_medical = fft2(medical_img.astype(float))
        fft_medical_shifted = fftshift(fft_medical)
        magnitude_medical = np.log(np.abs(fft_medical_shifted) + 1)
        
        fft_processed = fft2(processed_img.astype(float))
        fft_processed_shifted = fftshift(fft_processed)
        magnitude_processed = np.log(np.abs(fft_processed_shifted) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0, 0].imshow(medical_img, cmap='gray')
        axes[0, 0].set_title("Original")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(magnitude_medical, cmap='gray')
        axes[0, 1].set_title("Original Spectrum")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(processed_img, cmap='gray')
        axes[1, 0].set_title("Processed")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(magnitude_processed, cmap='gray')
        axes[1, 1].set_title("Processed Spectrum")
        axes[1, 1].axis('off')
        
        st.pyplot(fig)
        
        st.info(f"""
        **Medical Imaging Insight:**
        
        **Case:** {medical_case}
        **Target:** Enhancing {target}
        **Filter Strategy:** {filter_type}
        
        **Clinical Benefit:** 
        - Improved visibility of diagnostic features
        - Reduced interpretation time
        - Increased diagnostic confidence
        - Better treatment planning
        
        **Real-world example:** In MRI scans, frequency filters can improve tumor boundary definition by 30-50%!
        """)
    
    # ==================== TAB 3: SATELLITE & REMOTE SENSING ====================
    with tab3:
        st.header("🛰️ Satellite & Remote Sensing")
        
        st.markdown("""
        Frequency domain analysis is essential in remote sensing for:
        - **Image enhancement** of satellite/aerial photos
        - **Pattern detection** in geographical features
        - **Noise removal** from atmospheric interference
        - **Feature extraction** for land use analysis
        """)
        
        # Create remote sensing examples
        st.markdown("### 🌍 Remote Sensing Applications")
        
        rs_apps = st.selectbox(
            "Select remote sensing application:",
            ["Agricultural Monitoring", "Urban Planning", 
             "Disaster Assessment", "Environmental Monitoring",
             "Geological Survey"]
        )
        
        # Create appropriate imagery
        if "Agricultural" in rs_apps:
            rs_img = create_agricultural_image()
            analysis_target = "crop health patterns"
            filter_method = "Multi-band analysis for vegetation indices"
        elif "Urban" in rs_apps:
            rs_img = create_urban_image()
            analysis_target = "building patterns and road networks"
            filter_method = "Edge enhancement and pattern recognition"
        elif "Disaster" in rs_apps:
            rs_img = create_disaster_image()
            analysis_target = "damage assessment and change detection"
            filter_method = "Texture analysis and anomaly detection"
        elif "Environmental" in rs_apps:
            rs_img = create_environmental_image()
            analysis_target = "pollution patterns and ecosystem health"
            filter_method = "Spectral signature analysis"
        else:
            rs_img = create_geological_image()
            analysis_target = "mineral deposits and geological structures"
            filter_method = "Multi-frequency pattern extraction"
        
        # Process remote sensing image
        processed_rs = process_remote_sensing(rs_img, rs_apps)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Satellite/Aerial Image")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(rs_img, cmap='gray')
            ax.set_title(f"{rs_apps} - Raw Data")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Processed Analysis")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(processed_rs, cmap='gray')
            ax.set_title(f"Enhanced for {analysis_target}")
            ax.axis('off')
            st.pyplot(fig)
        
        # Multi-spectral analysis
        st.markdown("---")
        st.markdown("### 🌈 Multi-spectral Frequency Analysis")
        
        # Create multi-band demonstration
        bands = ["Visible", "Near Infrared", "Short-wave Infrared", "Thermal"]
        band_images = []
        
        for band in bands:
            band_img = create_spectral_band(rs_img, band)
            band_images.append(band_img)
        
        # Display bands
        band_cols = st.columns(len(bands))
        for idx, (band, img) in enumerate(zip(bands, band_images)):
            with band_cols[idx]:
                st.image(img, caption=band, use_container_width=True)
        
        # Frequency analysis of each band
        st.markdown("#### 📶 Band-specific Frequency Analysis")
        
        fig, axes = plt.subplots(2, len(bands), figsize=(15, 8))
        
        for idx, (band, img) in enumerate(zip(bands, band_images)):
            # Calculate spectrum
            fft_band = fft2(img.astype(float))
            fft_band_shifted = fftshift(fft_band)
            magnitude_band = np.log(np.abs(fft_band_shifted) + 1)
            
            axes[0, idx].imshow(img, cmap='gray')
            axes[0, idx].set_title(f"{band} Band")
            axes[0, idx].axis('off')
            
            axes[1, idx].imshow(magnitude_band, cmap='gray')
            axes[1, idx].set_title(f"{band} Spectrum")
            axes[1, idx].axis('off')
        
        st.pyplot(fig)
        
        # Application details
        st.markdown("---")
        st.markdown("### 📡 Remote Sensing Technology Details")
        
        tech_cols = st.columns(2)
        
        with tech_cols[0]:
            st.markdown("""
            **Common Satellite Sensors:**
            - **Landsat**: 30m resolution, 11 bands
            - **Sentinel-2**: 10-60m resolution, 13 bands  
            - **MODIS**: 250m-1km resolution, 36 bands
            - **WorldView**: 0.3m resolution, 8 bands
            
            **Frequency Applications:**
            - **Atmospheric correction**: Remove haze/scattering
            - **Pan-sharpening**: Enhance spatial resolution
            - **Change detection**: Monitor temporal changes
            - **Classification**: Land use/land cover mapping
            """)
        
        with tech_cols[1]:
            st.markdown(f"""
            **{rs_apps} Specifics:**
            
            **Analysis Target:** {analysis_target}
            
            **Filter Method:** {filter_method}
            
            **Key Frequencies:**
            - Low: Terrain and large-scale patterns
            - Medium: Field boundaries, roads
            - High: Individual plants, buildings
            
            **Data Sources:**
            - Satellite imagery
            - Aerial photography
            - Drone surveys
            - Historical archives
            """)
        
        st.success(f"""
        **Remote Sensing Impact:**
        
        Frequency domain processing in {rs_apps.lower()} enables:
        - **Early detection** of issues (crop disease, urban sprawl)
        - **Quantitative analysis** of changes over time
        - **Automated monitoring** of large areas
        - **Data fusion** from multiple sources
        
        **Example:** Agricultural monitoring can predict yield with 85-90% accuracy using frequency-based texture analysis!
        """)
    
    # ==================== TAB 4: INDUSTRIAL INSPECTION ====================
    with tab4:
        st.header("🏭 Industrial Inspection & Quality Control")
        
        st.markdown("""
        Frequency domain filtering revolutionizes industrial inspection by:
        - **Automating defect detection** with high accuracy
        - **Measuring surface quality** quantitatively
        - **Identifying patterns** invisible to human eye
        - **Real-time quality control** in production lines
        """)
        
        # Industrial applications
        st.markdown("### 🔧 Industrial Applications")
        
        industry = st.selectbox(
            "Select industry for inspection:",
            ["Manufacturing - Surface Defects",
             "Electronics - PCB Inspection", 
             "Textile - Fabric Quality",
             "Food - Quality Sorting",
             "Automotive - Part Inspection"]
        )
        
        # Create inspection images
        if "Manufacturing" in industry:
            inspect_img = create_manufacturing_defect()
            defect_type = "surface scratches and pits"
            inspection_method = "Texture analysis and anomaly detection"
        elif "Electronics" in industry:
            inspect_img = create_pcb_defect()
            defect_type = "solder bridges and missing components"
            inspection_method = "Pattern matching and edge analysis"
        elif "Textile" in industry:
            inspect_img = create_fabric_defect()
            defect_type = "weave irregularities and stains"
            inspection_method = "Periodic pattern analysis"
        elif "Food" in industry:
            inspect_img = create_food_defect()
            defect_type = "bruises, rot, and foreign objects"
            inspection_method = "Color and texture analysis"
        else:
            inspect_img = create_automotive_defect()
            defect_type = "cracks, dents, and finish issues"
            inspection_method = "Surface reflection analysis"
        
        # Process for defect detection
        processed_inspect = process_inspection_image(inspect_img, industry)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Raw Inspection Image")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(inspect_img, cmap='gray')
            ax.set_title(f"Raw {industry.split(' - ')[1]}")
            ax.axis('off')
            st.pyplot(fig)
            
            # Calculate quality metrics
            uniformity = calculate_uniformity(inspect_img)
            st.metric("Surface Uniformity", f"{uniformity:.1f}%")
        
        with col2:
            st.markdown("#### Defect Detection Result")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(processed_inspect, cmap='gray')
            ax.set_title(f"Defects Highlighted")
            ax.axis('off')
            st.pyplot(fig)
            
            # Defect metrics
            defect_area = calculate_defect_area(processed_inspect)
            st.metric("Defect Area", f"{defect_area:.1f}%")
        
        # Defect analysis in frequency domain
        st.markdown("---")
        st.markdown("### 🔍 Frequency-based Defect Analysis")
        
        # Calculate defect spectrum
        fft_defect = fft2(inspect_img.astype(float))
        fft_defect_shifted = fftshift(fft_defect)
        magnitude_defect = np.log(np.abs(fft_defect_shifted) + 1)
        
        # Create defect mask
        defect_mask = create_defect_mask(processed_inspect)
        
        # Analyze defect frequencies
        defect_freq_analysis = analyze_defect_frequencies(magnitude_defect, defect_mask)
        
        # Display frequency analysis
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0, 0].imshow(inspect_img, cmap='gray')
        axes[0, 0].set_title("Inspection Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(magnitude_defect, cmap='gray')
        axes[0, 1].set_title("Frequency Spectrum")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(defect_mask, cmap='hot')
        axes[1, 0].set_title("Defect Locations")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(defect_freq_analysis, cmap='hot')
        axes[1, 1].set_title("Defect Frequency Signature")
        axes[1, 1].axis('off')
        
        st.pyplot(fig)
        
        # Defect classification
        st.markdown("#### 🏷️ Defect Classification")
        
        defect_features = extract_defect_features(inspect_img, processed_inspect)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Defect Count", defect_features['count'])
        with col2:
            st.metric("Avg Defect Size", f"{defect_features['avg_size']:.1f} px")
        with col3:
            st.metric("Max Severity", f"{defect_features['max_severity']:.1f}")
        with col4:
            st.metric("Classification", defect_features['classification'])
        
        # Quality control decision
        st.markdown("#### ⚖️ Quality Control Decision")
        
        quality_score = calculate_quality_score(defect_features)
        
        if quality_score > 90:
            decision = "✅ ACCEPT - Excellent Quality"
            decision_color = "green"
        elif quality_score > 75:
            decision = "⚠️ ACCEPT - Minor Defects"
            decision_color = "orange"
        elif quality_score > 50:
            decision = "⚠️ REWORK - Moderate Defects"
            decision_color = "yellow"
        else:
            decision = "❌ REJECT - Major Defects"
            decision_color = "red"
        
        st.markdown(f"""
        <div style='background-color: {decision_color}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='color: white;'>{decision}</h2>
            <h3 style='color: white;'>Quality Score: {quality_score}/100</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Industrial benefits
        st.markdown("---")
        st.markdown("### 🏆 Industrial Benefits")
        
        benefit_cols = st.columns(3)
        
        with benefit_cols[0]:
            st.markdown("""
            **💰 Cost Reduction:**
            - Automated inspection: 70-90% labor cost reduction
            - Early defect detection: 40-60% material waste reduction
            - Reduced rework: 50-80% rework cost savings
            - Faster inspection: 3-10x speed increase
            """)
        
        with benefit_cols[1]:
            st.markdown("""
            **📈 Quality Improvement:**
            - Consistent standards: 99.9% consistency
            - Quantitative metrics: Precise quality measurements
            - Traceability: Complete defect history
            - Continuous improvement: Data-driven optimization
            """)
        
        with benefit_cols[2]:
            st.markdown(f"""
            **{industry} Specific:**
            
            **Defect Types:** {defect_type}
            
            **Inspection Method:** {inspection_method}
            
            **Accuracy:** 95-99% defect detection rate
            
            **Speed:** 100-1000 parts per minute
            
            **ROI:** 3-12 month payback period
            """)
        
        st.info(f"""
        **Real-world Example:** 
        
        In automotive manufacturing, frequency-based inspection systems:
        - Detect paint defects as small as 0.1mm
        - Inspect 60 cars per hour (vs. 6 manually)
        - Reduce warranty claims by 40%
        - Improve customer satisfaction by 25%
        
        **Technology Adoption:** Over 80% of major manufacturers now use frequency-domain inspection systems!
        """)
    
    # ==================== TAB 5: CREATIVE APPLICATIONS ====================
    with tab5:
        st.header("🎨 Creative & Artistic Applications")
        
        st.markdown("""
        Frequency domain filtering isn't just for technical applications!
        Artists and creatives use these techniques for:
        - **Digital art creation** with unique textures
        - **Photo manipulation** and special effects
        - **Generative art** from mathematical patterns
        - **Style transfer** and artistic filters
        """)
        
        # Creative tools
        st.markdown("### 🎭 Creative Frequency Tools")
        
        creative_tool = st.selectbox(
            "Select creative tool:",
            ["Frequency Painting", "Pattern Synthesis", 
             "Texture Transfer", "Artistic Filters",
             "Generative Art"]
        )
        
        if creative_tool == "Frequency Painting":
            st.markdown("#### 🖌️ Frequency Painting Studio")
            st.markdown("Paint in the frequency domain to create unique image effects!")
            
            # Create interactive painting
            painting_size = 256
            painting_canvas = np.ones((painting_size, painting_size))
            
            # Painting controls
            col1, col2, col3 = st.columns(3)
            with col1:
                paint_value = st.slider("Paint value", 0.0, 1.0, 0.5, 0.1)
            with col2:
                paint_radius = st.slider("Brush radius", 1, 50, 10, 1)
            with col3:
                paint_shape = st.selectbox("Brush shape", ["Circle", "Square", "Star", "Wave"])
            
            # Create painting pattern
            if paint_shape == "Circle":
                y, x = np.ogrid[:painting_size, :painting_size]
                center = painting_size // 2
                mask = np.sqrt((x - center)**2 + (y - center)**2) < paint_radius
                painting_canvas[mask] = paint_value
            
            elif paint_shape == "Square":
                center = painting_size // 2
                painting_canvas[center-paint_radius:center+paint_radius, 
                               center-paint_radius:center+paint_radius] = paint_value
            
            elif paint_shape == "Star":
                # Create star pattern
                painting_canvas = create_star_pattern(painting_size, paint_radius)
            
            else:  # Wave
                y, x = np.ogrid[:painting_size, :painting_size]
                painting_canvas = 0.5 + 0.5 * np.sin(0.1 * x) * np.cos(0.1 * y)
            
            # Apply to test image
            test_img = create_test_image()
            creative_result = apply_frequency_painting(test_img, painting_canvas)
            
            # Display
            col1, col2 = st.columns(2)
            with col1:
                st.image(painting_canvas, caption="Frequency Painting", use_container_width=True)
            with col2:
                st.image(creative_result, caption="Applied to Image", use_container_width=True)
            
            st.info("**Artistic Insight:** Painting in frequency domain creates global image effects!")
        
        elif creative_tool == "Pattern Synthesis":
            st.markdown("#### 🔄 Pattern Synthesis Generator")
            st.markdown("Create infinite patterns from frequency components!")
            
            # Pattern parameters
            col1, col2 = st.columns(2)
            with col1:
                pattern_type = st.selectbox("Pattern type", 
                                          ["Geometric", "Organic", "Fractal", "Chaotic"])
            with col2:
                complexity = st.slider("Pattern complexity", 1, 10, 3, 1)
            
            # Generate pattern
            pattern = generate_pattern(pattern_type, complexity)
            
            # Display
            st.image(pattern, caption=f"{pattern_type} Pattern", use_container_width=True)
            
            # Pattern variations
            st.markdown("##### Pattern Variations")
            variations = generate_pattern_variations(pattern, 4)
            var_cols = st.columns(4)
            for idx, var in enumerate(variations):
                with var_cols[idx]:
                    st.image(var, use_container_width=True)
        
        elif creative_tool == "Texture Transfer":
            st.markdown("#### 🎭 Texture Transfer Studio")
            st.markdown("Transfer textures between images using frequency domain!")
            
            # Create source and target images
            source_texture = create_texture_sample()
            target_image = create_target_image()
            
            # Transfer texture
            transferred = transfer_texture_frequency(source_texture, target_image)
            
            # Display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(source_texture, caption="Source Texture", use_container_width=True)
            with col2:
                st.image(target_image, caption="Target Image", use_container_width=True)
            with col3:
                st.image(transferred, caption="Texture Transferred", use_container_width=True)
        
        elif creative_tool == "Artistic Filters":
            st.markdown("#### 🎨 Artistic Filter Gallery")
            st.markdown("Apply artistic frequency filters to create unique styles!")
            
            # Filter gallery
            artistic_filters = ["Van Gogh", "Pointillism", "Cubism", "Impressionism", "Pop Art"]
            selected_filter = st.selectbox("Select artistic style:", artistic_filters)
            
            # Apply filter
            art_image = create_art_image()
            filtered_art = apply_artistic_filter(art_image, selected_filter)
            
            # Display
            col1, col2 = st.columns(2)
            with col1:
                st.image(art_image, caption="Original", use_container_width=True)
            with col2:
                st.image(filtered_art, caption=f"{selected_filter} Style", use_container_width=True)
            
            # Filter explanation
            filter_descriptions = {
                "Van Gogh": "Swirling patterns in mid-high frequencies",
                "Pointillism": "Isolated high frequency dots",
                "Cubism": "Geometric patterns at specific angles",
                "Impressionism": "Soft, blended mid frequencies",
                "Pop Art": "High contrast with specific color frequencies"
            }
            st.info(f"**{selected_filter} Technique:** {filter_descriptions[selected_filter]}")
        
        else:  # Generative Art
            st.markdown("#### 🌀 Generative Art Creator")
            st.markdown("Create art from mathematical frequency functions!")
            
            # Mathematical function input
            math_function = st.text_area(
                "Enter mathematical function f(x,y):",
                value="np.sin(0.1*x) * np.cos(0.1*y) + 0.5*np.sin(0.05*x*y)",
                height=100
            )
            
            # Generate art
            try:
                generative_art = generate_math_art(math_function)
                st.image(generative_art, caption="Generative Art", use_container_width=True)
                
                # Create variations
                st.markdown("##### Color Variations")
                color_variations = create_color_variations(generative_art, 4)
                color_cols = st.columns(4)
                for idx, var in enumerate(color_variations):
                    with color_cols[idx]:
                        st.image(var, use_container_width=True)
            
            except:
                st.error("Invalid mathematical function!")
        
        # Creative applications showcase
        st.markdown("---")
        st.markdown("### 🌟 Creative Applications Showcase")
        
        showcase = st.selectbox(
            "Select creative application showcase:",
            ["Digital Painting", "Photo Restoration", 
             "Game Texture Generation", "Fashion Design", "Architectural Visualization"]
        )
        
        # Create showcase example
        showcase_result = create_creative_showcase(showcase)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Before Processing")
            before_img = create_showcase_before(showcase)
            st.image(before_img, use_container_width=True)
        with col2:
            st.markdown("#### After Frequency Art")
            st.image(showcase_result, use_container_width=True)
        
        # Creative impact
        st.markdown("---")
        st.markdown("### 🎭 Impact on Creative Industries")
        
        impact_cols = st.columns(2)
        
        with impact_cols[0]:
            st.markdown("""
            **🎨 Artistic Impact:**
            - **New mediums**: Digital frequency art
            - **Unique styles**: Previously impossible effects
            - **Interactive art**: Real-time frequency manipulation
            - **Generative art**: Infinite variations from algorithms
            
            **Examples:**
            - Museum exhibitions of frequency art
            - Digital art sold as NFTs
            - Interactive installations
            - Algorithmic fashion design
            """)
        
        with impact_cols[1]:
            st.markdown("""
            **💼 Commercial Applications:**
            - **Advertising**: Eye-catching frequency effects
            - **Entertainment**: Movie/TV special effects
            - **Gaming**: Procedural texture generation
            - **Fashion**: Digital textile design
            - **Architecture**: Visualizing material properties
            
            **Market Size:**
            - Digital art market: $3+ billion
            - Game development: $200+ billion
            - Movie VFX: $10+ billion
            - Fashion tech: $5+ billion
            """)
        
        st.success("""
        **Creative Future:**
        
        Frequency domain techniques are opening new frontiers in creative expression:
        
        **🎯 Democratization:** Tools once available only to scientists are now accessible to artists
        
        **🚀 Innovation:** New art forms emerging from mathematical principles
        
        **🌈 Collaboration:** Artists and technologists working together
        
        **🌍 Impact:** Changing how we create, experience, and value art
        
        **The future of art is not just about what we see, but about understanding and manipulating the frequencies that make up what we see!**
        """)

# ==================== HELPER FUNCTIONS ====================

def create_test_image():
    """Create test image for demonstrations"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Gradient
    for i in range(size):
        for j in range(size):
            img[i, j] = 128 + 50 * np.sin(0.05*i) * np.cos(0.05*j)
    
    # Add features
    cv2.circle(img, (size//2, size//2), 60, 200, -1)
    cv2.rectangle(img, (50, 180), (150, 230), 255, -1)
    
    return img

def create_xray_image():
    """Create simulated X-ray image"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Bone structure
    cv2.line(img, (50, 50), (200, 50), 255, 10)  # Horizontal bone
    cv2.line(img, (125, 50), (125, 200), 255, 8)  # Vertical bone
    
    # Add noise and texture
    noise = np.random.normal(0, 20, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def create_mri_image():
    """Create simulated MRI image"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Brain-like structure
    cv2.circle(img, (size//2, size//2), 100, 150, -1)
    cv2.circle(img, (size//2, size//2), 60, 100, -1)
    
    # Add lots of noise
    noise = np.random.normal(0, 40, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    return img

def create_ultrasound_image():
    """Create simulated ultrasound image"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Ultrasound speckle pattern
    for i in range(size):
        for j in range(size):
            img[i, j] = 100 + 50 * np.random.rand()
    
    # Add some structure
    for i in range(0, size, 30):
        img[i:i+5, :] = 200
    
    # Add periodic artifacts
    for i in range(size):
        if i % 20 == 0:
            img[:, i:i+3] = np.clip(img[:, i:i+3] + 50, 0, 255)
    
    return img

def enhance_medical_image(img, modality):
    """Enhance medical image based on modality"""
    if modality == "xray":
        # High-pass for edge enhancement
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    
    elif modality == "mri":
        # Low-pass for noise reduction
        return cv2.GaussianBlur(img, (5, 5), 1.5)
    
    else:  # ultrasound
        # Remove periodic artifacts
        fft = fft2(img.astype(float))
        fft_shifted = fftshift(fft)
        
        # Create notch filter for periodic patterns
        rows, cols = img.shape
        center_y, center_x = rows//2, cols//2
        y, x = np.ogrid[:rows, :cols]
        
        mask = np.ones((rows, cols))
        for angle in [0, 90]:
            nx = center_x + int(100 * np.cos(np.radians(angle)))
            ny = center_y + int(100 * np.sin(np.radians(angle)))
            dist = np.sqrt((x - nx)**2 + (y - ny)**2)
            mask[dist < 5] = 0
        
        filtered = fft_shifted * mask
        filtered_ishift = ifftshift(filtered)
        result = np.real(ifft2(filtered_ishift))
        return np.clip(result, 0, 255).astype(np.uint8)

def calculate_snr(img):
    """Calculate signal-to-noise ratio estimate"""
    # Simple SNR estimation
    mean = img.mean()
    std = img.std()
    return 20 * np.log10(mean / (std + 1e-6))

def create_brain_mri():
    """Create brain MRI simulation"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Brain hemispheres
    cv2.ellipse(img, (size//2, size//2), (100, 80), 0, 0, 360, 180, -1)
    cv2.line(img, (size//2, 50), (size//2, 200), 100, 3)
    
    # Simulated tumor
    cv2.circle(img, (180, 120), 15, 255, -1)
    
    # Add noise
    noise = np.random.normal(0, 25, img.shape)
    img = np.clip(img + noise, 0, 255)
    
    return img.astype(np.uint8)

def create_chest_xray():
    """Create chest X-ray simulation"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Rib cage
    for i in range(5):
        y = 50 + i * 40
        cv2.ellipse(img, (size//2, y), (100, 15), 0, 0, 360, 200, 3)
    
    # Lungs
    cv2.ellipse(img, (size//3, 150), (40, 60), 0, 0, 360, 150, -1)
    cv2.ellipse(img, (2*size//3, 150), (40, 60), 0, 0, 360, 150, -1)
    
    # Add texture
    texture = np.random.rand(size, size) * 30
    img = np.clip(img + texture, 0, 255)
    
    return img.astype(np.uint8)

def create_retinal_scan():
    """Create retinal scan simulation"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Eye background
    cv2.circle(img, (size//2, size//2), 100, 100, -1)
    
    # Blood vessels
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        x1 = size//2 + int(30 * np.cos(angle))
        y1 = size//2 + int(30 * np.sin(angle))
        x2 = size//2 + int(90 * np.cos(angle))
        y2 = size//2 + int(90 * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), 255, 2)
    
    # Optic disc
    cv2.circle(img, (size//2, size//2), 20, 200, -1)
    
    return img

def create_dental_xray():
    """Create dental X-ray simulation"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Teeth
    for i in range(8):
        x = 30 + i * 25
        cv2.rectangle(img, (x, 50), (x+20, 150), 200, -1)
        # Cavity
        if i == 3:
            cv2.circle(img, (x+10, 100), 8, 100, -1)
    
    return img

def process_medical_image(img, case):
    """Process medical image based on case"""
    if "Brain" in case:
        # Edge enhancement for tumor detection
        fft_img = fft2(img.astype(float))
        fft_shifted = fftshift(fft_img)
        
        # High-pass filter
        rows, cols = img.shape
        center_y, center_x = rows//2, cols//2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        mask = np.zeros((rows, cols))
        mask[dist > 30] = 2.0  # Boost edges
        
        filtered = fft_shifted * mask
        filtered_ishift = ifftshift(filtered)
        result = np.real(ifft2(filtered_ishift))
        return np.clip(result, 0, 255).astype(np.uint8)
    
    elif "Chest" in case:
        # Texture enhancement
        return cv2.equalizeHist(img)
    
    elif "Retinal" in case:
        # Vessel enhancement
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)
    
    else:  # Dental
        # Multi-band enhancement
        low = cv2.GaussianBlur(img, (21, 21), 10)
        high = cv2.subtract(img, low)
        enhanced = cv2.addWeighted(img, 1.5, high, 0.5, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)

def create_agricultural_image():
    """Create agricultural field image"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Field patterns
    for i in range(0, size, 30):
        img[i:i+15, :] = 150  # Crop rows
    
    # Variable crop health
    for i in range(size):
        for j in range(size):
            if (i // 30) % 2 == 0:
                img[i, j] += int(50 * np.sin(0.02 * j))
    
    # Add some unhealthy patches
    cv2.circle(img, (100, 100), 20, 80, -1)
    cv2.circle(img, (180, 180), 15, 60, -1)
    
    return img

def create_urban_image():
    """Create urban area image"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Buildings
    buildings = [(50, 50, 60, 100), (120, 30, 40, 120), (180, 60, 50, 90)]
    for x, y, w, h in buildings:
        cv2.rectangle(img, (x, y), (x+w, y+h), 200, -1)
    
    # Roads
    cv2.line(img, (0, 150), (size, 150), 100, 10)
    cv2.line(img, (128, 0), (128, size), 100, 10)
    
    # Texture
    noise = np.random.rand(size, size) * 20
    img = np.clip(img + noise, 0, 255)
    
    return img.astype(np.uint8)

def create_disaster_image():
    """Create disaster area image"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Before (normal)
    for i in range(0, size, 20):
        img[i:i+10, :] = 150
    
    # Disaster damage
    cv2.circle(img, (100, 100), 40, 50, -1)  # Flooded area
    cv2.rectangle(img, (180, 50), (220, 150), 80, -1)  # Collapsed building
    
    # Debris (noise)
    debris = np.random.rand(size, size) * 100
    debris[debris < 0.9] = 0
    img = np.clip(img - debris, 0, 255)
    
    return img.astype(np.uint8)

def create_environmental_image():
    """Create environmental monitoring image"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Forest/vegetation
    for i in range(size):
        for j in range(size):
            img[i, j] = 100 + int(50 * np.sin(0.03*i) * np.cos(0.03*j))
    
    # Polluted river
    cv2.line(img, (50, 0), (50, size), 50, 15)
    
    # Industrial area
    cv2.rectangle(img, (180, 180), (230, 230), 200, -1)
    
    # Smoke plume
    for i in range(30):
        radius = 5 + i//3
        cv2.circle(img, (205, 180-i), radius, 150, -1)
    
    return img

def create_geological_image():
    """Create geological survey image"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Geological layers
    for i in range(5):
        y = i * 40 + 30
        thickness = 10 + i * 5
        cv2.line(img, (0, y), (size, y), 100 + i*20, thickness)
    
    # Fault line
    cv2.line(img, (100, 0), (150, size), 50, 3)
    
    # Mineral deposit
    cv2.circle(img, (180, 100), 25, 255, -1)
    
    return img

def process_remote_sensing(img, application):
    """Process remote sensing image"""
    # Common enhancement
    enhanced = cv2.equalizeHist(img)
    
    if "Agricultural" in application:
        # Texture analysis for crop health
        fft_img = fft2(img.astype(float))
        fft_shifted = fftshift(fft_img)
        
        # Band-pass for crop patterns
        rows, cols = img.shape
        center_y, center_x = rows//2, cols//2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        mask = np.zeros((rows, cols))
        mask[(dist > 20) & (dist < 60)] = 1.5  # Boost mid frequencies
        
        filtered = fft_shifted * mask
        filtered_ishift = ifftshift(filtered)
        result = np.real(ifft2(filtered_ishift))
        return np.clip(result, 0, 255).astype(np.uint8)
    
    elif "Urban" in application:
        # Edge enhancement
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(enhanced, -1, kernel)
    
    elif "Disaster" in application:
        # Change detection simulation
        return cv2.subtract(enhanced, cv2.GaussianBlur(enhanced, (31, 31), 15))
    
    else:
        return enhanced

def create_spectral_band(base_img, band):
    """Create different spectral bands"""
    if band == "Visible":
        return base_img
    elif band == "Near Infrared":
        # Simulate NIR response
        return cv2.add(base_img, 30)
    elif band == "Short-wave Infrared":
        # Simulate SWIR response
        return cv2.subtract(base_img, 30)
    else:  # Thermal
        # Simulate thermal band
        return cv2.GaussianBlur(base_img, (15, 15), 5)

def create_manufacturing_defect():
    """Create manufacturing surface with defects"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Metal surface texture
    for i in range(size):
        for j in range(size):
            img[i, j] = 150 + int(20 * np.sin(0.1*i) * np.cos(0.1*j))
    
    # Defects
    cv2.line(img, (50, 100), (150, 100), 80, 2)  # Scratch
    cv2.circle(img, (180, 80), 10, 100, -1)  # Pit
    cv2.rectangle(img, (100, 180), (120, 200), 200, -1)  # Burr
    
    # Add noise
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255)
    
    return img.astype(np.uint8)

def create_pcb_defect():
    """Create PCB with defects"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # PCB traces
    cv2.rectangle(img, (30, 30), (100, 100), 200, 3)
    cv2.rectangle(img, (120, 30), (190, 100), 200, 3)
    cv2.line(img, (100, 65), (120, 65), 200, 3)
    
    # Components
    cv2.circle(img, (65, 150), 15, 150, -1)  # Good capacitor
    cv2.circle(img, (160, 150), 15, 150, 2)  # Missing solder (hollow)
    
    # Solder bridge defect
    cv2.line(img, (65, 180), (160, 180), 255, 5)
    
    return img

def create_fabric_defect():
    """Create fabric with defects"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Weave pattern
    for i in range(0, size, 10):
        for j in range(0, size, 10):
            if (i//10 + j//10) % 2 == 0:
                img[i:i+5, j:j+5] = 200
            else:
                img[i:i+5, j:j+5] = 100
    
    # Defects
    img[100:120, 100:120] = 50  # Stain
    img[150:170, 80:100] = 255  # Hole
    # Broken weave pattern
    for i in range(180, 200, 10):
        img[i:i+5, 150:200] = 150
    
    return img

def create_food_defect():
    """Create food item with defects"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Apple/fruit
    cv2.ellipse(img, (size//2, size//2), (80, 100), 0, 0, 360, 150, -1)
    
    # Defects
    cv2.circle(img, (100, 100), 15, 100, -1)  # Bruise
    cv2.circle(img, (180, 120), 10, 50, -1)   # Rot spot
    
    # Stem
    cv2.line(img, (128, 30), (128, 70), 100, 5)
    
    # Texture
    noise = np.random.rand(size, size) * 30
    img = np.clip(img + noise, 0, 255)
    
    return img.astype(np.uint8)

def create_automotive_defect():
    """Create automotive part with defects"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Car panel
    for i in range(size):
        for j in range(size):
            img[i, j] = 180 + int(20 * np.sin(0.05*i))
    
    # Defects
    cv2.line(img, (50, 150), (100, 180), 100, 2)  # Scratch
    cv2.circle(img, (180, 100), 15, 220, -1)      # Dent (different reflection)
    
    # Paint defect
    cv2.rectangle(img, (120, 80), (140, 100), 150, -1)
    
    return img

def process_inspection_image(img, industry):
    """Process image for defect detection"""
    # Edge detection for defects
    edges = cv2.Canny(img, 50, 150)
    
    if "Manufacturing" in industry:
        # Emphasize linear defects
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        return dilated
    
    elif "Electronics" in industry:
        # Emphasize circular/rectangular features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(img)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # Small defects
                cv2.drawContours(result, [cnt], -1, 255, -1)
        return result
    
    elif "Textile" in industry:
        # Pattern deviation detection
        fft_img = fft2(img.astype(float))
        fft_shifted = fftshift(fft_img)
        
        rows, cols = img.shape
        center_y, center_x = rows//2, cols//2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Expected pattern frequencies
        expected_mask = np.zeros((rows, cols))
        expected_mask[(dist > 20) & (dist < 40)] = 1
        
        # Find deviations
        magnitude = np.abs(fft_shifted)
        deviation = cv2.absdiff(magnitude * expected_mask, magnitude)
        deviation = np.clip(deviation * 10, 0, 255).astype(np.uint8)
        
        return deviation
    
    else:
        return edges

def calculate_uniformity(img):
    """Calculate surface uniformity"""
    std = img.std()
    return max(0, 100 - std)

def calculate_defect_area(processed_img):
    """Calculate defect area percentage"""
    defect_pixels = np.sum(processed_img > 50)
    total_pixels = processed_img.size
    return defect_pixels / total_pixels * 100

def create_defect_mask(processed_img):
    """Create mask of defect locations"""
    mask = np.zeros_like(processed_img, dtype=np.uint8)
    mask[processed_img > 50] = 255
    return mask

def analyze_defect_frequencies(spectrum, defect_mask):
    """Analyze frequency content of defects"""
    # Simple analysis: defect areas in spectrum
    rows, cols = spectrum.shape
    center_y, center_x = rows//2, cols//2
    
    # Create frequency bands
    analysis = np.zeros((rows, cols))
    
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Mark different frequency regions
    analysis[dist < 20] = 1  # Low frequencies
    analysis[(dist >= 20) & (dist < 60)] = 2  # Mid frequencies
    analysis[dist >= 60] = 3  # High frequencies
    
    return analysis

def extract_defect_features(original, processed):
    """Extract defect features"""
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = {
        'count': len(contours),
        'avg_size': 0,
        'max_severity': 0,
        'classification': 'None'
    }
    
    if contours:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        features['avg_size'] = np.mean(areas)
        features['max_severity'] = np.max(areas) / 100
        
        # Classify based on features
        if features['count'] > 5:
            features['classification'] = 'Multiple Small'
        elif features['max_severity'] > 10:
            features['classification'] = 'Major Defect'
        else:
            features['classification'] = 'Minor Defect'
    
    return features

def calculate_quality_score(features):
    """Calculate quality score from features"""
    score = 100
    
    # Penalties
    score -= features['count'] * 5  # Each defect
    score -= features['max_severity'] * 2  # Severity
    
    # Cap at 0-100
    return max(0, min(100, score))

def create_star_pattern(size, radius):
    """Create star pattern for painting"""
    img = np.ones((size, size)) * 0.5
    
    center = size // 2
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        # Star points
        x = center + int(radius * 1.5 * np.cos(angle))
        y = center + int(radius * 1.5 * np.sin(angle))
        
        # Connect points
        next_angle = angle + np.pi/4
        x2 = center + int(radius * 0.5 * np.cos(next_angle))
        y2 = center + int(radius * 0.5 * np.sin(next_angle))
        
        cv2.line(img, (center, center), (x, y), 1.0, 2)
        cv2.line(img, (x, y), (x2, y2), 0.8, 2)
    
    return img

def apply_frequency_painting(img, painting):
    """Apply frequency painting to image"""
    fft_img = fft2(img.astype(float))
    fft_shifted = fftshift(fft_img)
    
    # Resize painting to match image
    if painting.shape != img.shape:
        painting = cv2.resize(painting, (img.shape[1], img.shape[0]))
    
    # Apply painting as filter
    filtered = fft_shifted * painting
    filtered_ishift = ifftshift(filtered)
    result = np.real(ifft2(filtered_ishift))
    
    return np.clip(result, 0, 255).astype(np.uint8)

def generate_pattern(pattern_type, complexity):
    """Generate different patterns"""
    size = 256
    
    if pattern_type == "Geometric":
        img = np.zeros((size, size))
        for i in range(complexity):
            angle = i * np.pi / complexity
            cv2.line(img, (0, 0), 
                    (int(size*np.cos(angle)), int(size*np.sin(angle))), 
                    1.0, 2)
    
    elif pattern_type == "Organic":
        img = np.random.rand(size, size)
        for _ in range(complexity):
            img = cv2.GaussianBlur(img, (5, 5), 1)
    
    elif pattern_type == "Fractal":
        img = np.zeros((size, size))
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        C = np.full_like(Z, -0.4 + 0.6j)
        
        for _ in range(complexity):
            Z = Z**2 + C
            mask = np.abs(Z) < 2
            img += mask.astype(float)
        
        img = img / img.max()
    
    else:  # Chaotic
        img = np.random.rand(size, size)
        img = np.sin(complexity * img * 2*np.pi)
    
    return (img * 255).astype(np.uint8)

def generate_pattern_variations(pattern, num_variations):
    """Generate variations of a pattern"""
    variations = []
    for i in range(num_variations):
        # Rotate and scale
        angle = i * 360 / num_variations
        scale = 0.8 + 0.4 * (i / num_variations)
        
        rows, cols = pattern.shape
        M = cv2.getRotationMatrix2D((cols//2, rows//2), angle, scale)
        variation = cv2.warpAffine(pattern, M, (cols, rows))
        
        variations.append(variation)
    
    return variations

def create_texture_sample():
    """Create texture sample"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Wood grain texture
    for i in range(size):
        for j in range(size):
            img[i, j] = 100 + int(50 * np.sin(0.02*j) * np.cos(0.005*i))
    
    return img

def create_target_image():
    """Create target image for texture transfer"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Simple shape
    cv2.circle(img, (size//2, size//2), 80, 200, -1)
    
    return img

def transfer_texture_frequency(source, target):
    """Transfer texture using frequency domain"""
    # Get source texture frequencies
    fft_source = fft2(source.astype(float))
    fft_source_shifted = fftshift(fft_source)
    magnitude_source = np.abs(fft_source_shifted)
    phase_source = np.angle(fft_source_shifted)
    
    # Get target structure frequencies
    fft_target = fft2(target.astype(float))
    fft_target_shifted = fftshift(fft_target)
    magnitude_target = np.abs(fft_target_shifted)
    phase_target = np.angle(fft_target_shifted)
    
    # Combine: source texture magnitude with target phase
    combined = magnitude_source * np.exp(1j * phase_target)
    combined_ishift = ifftshift(combined)
    result = np.real(ifft2(combined_ishift))
    
    return np.clip(result, 0, 255).astype(np.uint8)

def create_art_image():
    """Create image for artistic filters"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Simple composition
    cv2.rectangle(img, (50, 50), (200, 200), 150, -1)
    cv2.circle(img, (128, 128), 60, 200, -1)
    cv2.line(img, (0, 0), (size, size), 100, 3)
    
    return img

def apply_artistic_filter(img, style):
    """Apply artistic filter"""
    if style == "Van Gogh":
        # Swirling patterns
        fft_img = fft2(img.astype(float))
        fft_shifted = fftshift(fft_img)
        
        rows, cols = img.shape
        center_y, center_x = rows//2, cols//2
        y, x = np.ogrid[:rows, :cols]
        
        # Create swirl mask
        angle = np.arctan2(y - center_y, x - center_x)
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        swirl = np.exp(1j * dist * 0.01)  # Complex swirl
        
        filtered = fft_shifted * swirl
        filtered_ishift = ifftshift(filtered)
        result = np.real(ifft2(filtered_ishift))
    
    elif style == "Pointillism":
        # Isolated dots
        result = img.copy()
        for i in range(0, img.shape[0], 5):
            for j in range(0, img.shape[1], 5):
                if np.random.rand() > 0.7:
                    cv2.circle(result, (j, i), 2, img[i, j], -1)
        result = cv2.GaussianBlur(result, (3, 3), 1)
    
    elif style == "Cubism":
        # Geometric patterns
        result = cv2.Canny(img, 50, 150)
        result = cv2.dilate(result, np.ones((2, 2), np.uint8))
    
    elif style == "Impressionism":
        # Soft, blended
        result = cv2.GaussianBlur(img, (11, 11), 3)
    
    else:  # Pop Art
        # High contrast, posterized
        result = cv2.convertScaleAbs(img, alpha=2.0, beta=-100)
        _, result = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def generate_math_art(function_str):
    """Generate art from mathematical function"""
    size = 256
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    
    try:
        # Evaluate function
        art = eval(function_str, {"np": np, "X": X, "Y": Y, "x": X, "y": Y})
        # Normalize to 0-255
        art = (art - art.min()) / (art.max() - art.min()) * 255
        return art.astype(np.uint8)
    except:
        # Fallback pattern
        return generate_pattern("Fractal", 3)

def create_color_variations(img, num_variations):
    """Create color variations of grayscale image"""
    variations = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i in range(min(num_variations, len(colors))):
        # Create color version
        color_img = np.zeros((*img.shape, 3), dtype=np.uint8)
        color_img[:, :, 0] = img * colors[i][0] // 255
        color_img[:, :, 1] = img * colors[i][1] // 255
        color_img[:, :, 2] = img * colors[i][2] // 255
        variations.append(color_img)
    
    return variations

def create_creative_showcase(application):
    """Create showcase for creative application"""
    size = 256
    
    if application == "Digital Painting":
        # Create frequency-based painting
        img = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                r = np.sqrt((i-size//2)**2 + (j-size//2)**2)
                img[i, j] = 128 + int(50 * np.sin(0.05*r))
        
        # Add frequency patterns
        fft_img = fft2(img.astype(float))
        fft_shifted = fftshift(fft_img)
        
        # Artistic modulation
        rows, cols = img.shape
        center_y, center_x = rows//2, cols//2
        y, x = np.ogrid[:rows, :cols]
        angle = np.arctan2(y - center_y, x - center_x)
        
        modulation = np.exp(1j * 5 * angle)  # Star modulation
        artistic = fft_shifted * modulation
        
        artistic_ishift = ifftshift(artistic)
        result = np.real(ifft2(artistic_ishift))
    
    elif application == "Photo Restoration":
        # Simulate restoration
        img = create_test_image()
        # Add damage
        damaged = img.copy()
        damaged[100:120, 100:120] = 0
        damaged[50:70, 180:200] = 255
        
        # Restore using frequency inpainting
        fft_damaged = fft2(damaged.astype(float))
        fft_shifted = fftshift(fft_damaged)
        
        # Remove high frequency noise (damage)
        rows, cols = damaged.shape
        center_y, center_x = rows//2, cols//2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        mask = np.ones((rows, cols))
        mask[dist > 100] = 0  # Remove very high frequencies
        
        restored_fft = fft_shifted * mask
        restored_ishift = ifftshift(restored_fft)
        result = np.real(ifft2(restored_ishift))
    
    elif application == "Game Texture Generation":
        # Generate game texture
        base = generate_pattern("Fractal", 4)
        # Add some structure
        cv2.circle(base, (size//2, size//2), 60, 200, 3)
        result = base
    
    elif application == "Fashion Design":
        # Create fabric pattern
        result = np.zeros((size, size), dtype=np.uint8)
        for i in range(0, size, 20):
            for j in range(0, size, 20):
                if (i//20 + j//20) % 2 == 0:
                    result[i:i+10, j:j+10] = 200
                else:
                    result[i:i+10, j:j+10] = 100
        
        # Add fashionable pattern
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
            x = size//2 + int(80 * np.cos(angle))
            y = size//2 + int(80 * np.sin(angle))
            cv2.line(result, (size//2, size//2), (x, y), 255, 3)
    
    else:  # Architectural Visualization
        # Create architectural material
        result = np.zeros((size, size), dtype=np.uint8)
        # Concrete texture
        for i in range(size):
            for j in range(size):
                result[i, j] = 150 + int(30 * np.sin(0.03*i) * np.cos(0.03*j))
        
        # Add window reflections
        for i in range(0, size, 60):
            for j in range(0, size, 60):
                cv2.rectangle(result, (i+10, j+10), (i+40, j+40), 220, -1)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def create_showcase_before(application):
    """Create before image for showcase"""
    size = 256
    
    if application == "Digital Painting":
        return np.full((size, size), 128, dtype=np.uint8)
    
    elif application == "Photo Restoration":
        img = create_test_image()
        # Add damage
        damaged = img.copy()
        damaged[100:120, 100:120] = 0
        damaged[50:70, 180:200] = 255
        return damaged
    
    elif application == "Game Texture Generation":
        return np.zeros((size, size), dtype=np.uint8)
    
    elif application == "Fashion Design":
        return np.full((size, size), 150, dtype=np.uint8)
    
    else:  # Architectural Visualization
        return np.full((size, size), 100, dtype=np.uint8)

if __name__ == "__main__":
    app()