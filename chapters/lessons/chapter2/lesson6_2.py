import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def app():
    st.title("📊 Histogram & Histogram Equalization")
    
    # Simple analogy first
    st.markdown("""
    ### 🤔 The Core Idea (Simple Analogy)
    
    Imagine you're organizing **students by height** for a class photo:
    
    **❌ Bad Distribution (Original):**
    - Most students are 5'4" to 5'6" (clustered together)
    - Hard to see individual faces when everyone's the same height
    - Photo looks "flat" and boring
    
    **✅ Good Distribution (Equalized):**
    - Spread students out: 4'10", 5'2", 5'6", 5'10", 6'2"
    - Everyone is more visible and distinct
    - Photo has better "depth" and interest
    
    **Histogram Equalization does this with brightness:**
    - Takes clustered brightness values (all similar grays)
    - Spreads them across full range (dark blacks to bright whites)
    - Result: **Better contrast and more visible details!** ✨
    """)
    
    st.markdown("---")
    
    # Load and prepare images
    img = Image.open("public/ch1.1.jpg").convert('L')
    img_np = np.array(img).astype(float)
    
    # Create test images with different contrast levels
    low_contrast = np.clip(img_np * 0.4 + 80, 0, 255).astype(np.uint8)
    medium_contrast = img_np.astype(np.uint8)
    high_contrast = np.clip((img_np - 128) * 1.5 + 128, 0, 255).astype(np.uint8)
    
    # Controls
    st.markdown("### 🎛️ Choose Test Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        image_type = st.radio(
            "Select image to demonstrate on:",
            ["😴 Low Contrast (Washed Out)", "😐 Normal Contrast", "😎 High Contrast (Punchy)"],
            help="Different images show different improvements!"
        )
    
    with col2:
        show_process = st.checkbox("🔍 Show Step-by-Step Process", value=True)
    
    # Select working image
    if "Low Contrast" in image_type:
        working_img = low_contrast
        problem = "All pixels clustered in middle range (washed out)"
    elif "Normal" in image_type:
        working_img = medium_contrast
        problem = "Decent distribution but room for improvement"
    else:
        working_img = high_contrast
        problem = "Already good contrast (minimal improvement expected)"
    
    st.info(f"**Problem with this image:** {problem}")
    
    # Histogram equalization function
    def equalize_with_steps(image):
        # Step 1: Calculate histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        
        # Step 2: Calculate CDF
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        
        # Step 3: Map old intensities to new using CDF
        equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        equalized = equalized.reshape(image.shape).astype(np.uint8)
        
        return equalized, hist, cdf_normalized
    
    equalized, hist_orig, cdf = equalize_with_steps(working_img)
    
    # Main comparison
    st.markdown("---")
    st.markdown("### 📸 Before & After: The Transformation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(working_img, caption="⚫ BEFORE: Original Image", 
                use_container_width=True, channels="L")
        
        # Statistics
        range_orig = working_img.max() - working_img.min()
        st.markdown(f"""
        **Image Statistics:**  
        📉 Min Brightness: {working_img.min()}  
        📈 Max Brightness: {working_img.max()}  
        📏 **Range Used: {range_orig}/255** ({range_orig/255*100:.0f}%)  
        📊 Average: {working_img.mean():.0f}
        """)
        
        if range_orig < 180:
            st.warning(f"⚠️ Only using {range_orig/255*100:.0f}% of available brightness range!")
        else:
            st.success("✅ Using most of the brightness range")
    
    with col2:
        st.image(equalized, caption="⚪ AFTER: Equalized Image", 
                use_container_width=True, channels="L")
        
        # Statistics
        range_eq = equalized.max() - equalized.min()
        improvement = range_eq - range_orig
        st.markdown(f"""
        **Image Statistics:**  
        📉 Min Brightness: {equalized.min()}  
        📈 Max Brightness: {equalized.max()}  
        📏 **Range Used: {range_eq}/255** ({range_eq/255*100:.0f}%)  
        📊 Average: {equalized.mean():.0f}
        """)
        
        if improvement > 0:
            st.success(f"✨ Improved range by {improvement} levels! (+{improvement/255*100:.0f}%)")
        else:
            st.info("Already had good contrast - minimal change")
    
    # Histogram visualization
    st.markdown("---")
    st.markdown("### 📊 Histogram: Distribution of Brightness")
    st.markdown("**The histogram shows how many pixels have each brightness level (0=black, 255=white)**")
    
    # Calculate histograms
    hist_eq, _ = np.histogram(equalized.flatten(), 256, [0, 256])
    
    # Create histogram data for visualization
    bins_display = 32  # Use 32 bins for cleaner display
    hist_orig_grouped, bin_edges = np.histogram(working_img.flatten(), bins_display, [0, 256])
    hist_eq_grouped, _ = np.histogram(equalized.flatten(), bins_display, [0, 256])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 BEFORE: Original Histogram**")
        
        # Create DataFrame for bar chart
        hist_df_orig = pd.DataFrame({
            'Brightness': [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(hist_orig_grouped))],
            'Pixel Count': hist_orig_grouped
        })
        st.bar_chart(hist_df_orig.set_index('Brightness')['Pixel Count'])
        
        # Find concentration
        peak_bin = np.argmax(hist_orig_grouped)
        peak_range = f"{int(bin_edges[peak_bin])}-{int(bin_edges[peak_bin+1])}"
        st.caption(f"🎯 Most pixels concentrated at: {peak_range}")
    
    with col2:
        st.markdown("**📊 AFTER: Equalized Histogram**")
        
        # Create DataFrame for bar chart
        hist_df_eq = pd.DataFrame({
            'Brightness': [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(hist_eq_grouped))],
            'Pixel Count': hist_eq_grouped
        })
        st.bar_chart(hist_df_eq.set_index('Brightness')['Pixel Count'])
        
        st.caption("🎯 Pixels spread across full range!")
    
    # Key insight
    col1, col2 = st.columns(2)
    with col1:
        if range_orig < 180:
            st.error("❌ Histogram clustered (narrow peak)")
            st.caption("= Washed out, low contrast")
    with col2:
        st.success("✅ Histogram spread out (wider distribution)")
        st.caption("= Better contrast, more details visible")
    
    # Step-by-step process
    if show_process:
        st.markdown("---")
        st.markdown("### 🔬 How It Works: Step-by-Step")
        
        st.markdown("""
        **The Magic 3-Step Process:**
        
        1️⃣ **Count pixels at each brightness** (create histogram)  
        2️⃣ **Calculate cumulative sum** (how many pixels up to each brightness)  
        3️⃣ **Remap brightnesses** using the cumulative sum as a lookup table
        """)
        
        # Show example transformation
        st.markdown("#### 📋 Example: How Specific Brightnesses Get Transformed")
        
        sample_values = [32, 64, 96, 128, 160, 192, 224]
        transform_data = []
        
        for orig_val in sample_values:
            if orig_val < len(cdf):
                new_val = int(cdf[orig_val])
                change = new_val - orig_val
                
                # Determine arrow
                if change > 10:
                    arrow = "⬆️ Brightened"
                elif change < -10:
                    arrow = "⬇️ Darkened"
                else:
                    arrow = "➡️ Unchanged"
                
                transform_data.append({
                    "Original Brightness": orig_val,
                    "→ New Brightness": new_val,
                    "Change": change,
                    "Effect": arrow
                })
        
        df = pd.DataFrame(transform_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.caption("💡 Notice: Values get spread out to use the full 0-255 range!")
    
    # Side-by-side comparison
    st.markdown("---")
    st.markdown("### 👀 Direct Visual Comparison")
    
    composite = np.concatenate([working_img, equalized], axis=1)
    st.image(composite, caption="Left: Original (clustered histogram) | Right: Equalized (spread histogram)", 
            use_container_width=True, channels="L")
    
    # Interactive detail finder
    st.markdown("---")
    st.markdown("### 🔍 Detail Comparison: Find Hidden Details")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Select a region to examine:**")
        region = st.radio(
            "Region:",
            ["Top-Left", "Top-Right", "Center", "Bottom-Left", "Bottom-Right"],
            label_visibility="collapsed"
        )
    
    # Define regions
    h, w = working_img.shape
    regions = {
        "Top-Left": (0, h//3, 0, w//3),
        "Top-Right": (0, h//3, 2*w//3, w),
        "Center": (h//3, 2*h//3, w//3, 2*w//3),
        "Bottom-Left": (2*h//3, h, 0, w//3),
        "Bottom-Right": (2*h//3, h, 2*w//3, w)
    }
    
    y1, y2, x1, x2 = regions[region]
    
    with col2:
        col_a, col_b = st.columns(2)
        
        with col_a:
            crop_orig = working_img[y1:y2, x1:x2]
            st.image(crop_orig, caption=f"Original: {region}", use_container_width=True, channels="L")
            st.metric("Detail Visibility", f"{crop_orig.std():.0f}", help="Higher = more detail visible")
        
        with col_b:
            crop_eq = equalized[y1:y2, x1:x2]
            st.image(crop_eq, caption=f"Equalized: {region}", use_container_width=True, channels="L")
            improvement_pct = ((crop_eq.std() - crop_orig.std()) / crop_orig.std() * 100)
            st.metric("Detail Visibility", f"{crop_eq.std():.0f}", 
                     delta=f"{improvement_pct:+.0f}%",
                     help="Higher = more detail visible")
    
    # Real-world applications
    st.markdown("---")
    st.markdown("### 🌍 Real-World Magic: Where This Saves The Day")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🏥 Medical Imaging (Life-Saving!):**
        - **X-rays**: See hidden bone fractures
        - **CT scans**: Distinguish tumor from tissue
        - **MRI**: Reveal subtle brain abnormalities
        - **Mammograms**: Detect early cancer signs
        
        **Real Example:** A faint fracture invisible on original X-ray 
        becomes clearly visible after equalization, allowing proper treatment!
        """)
        
        st.info("""
        **📸 Photography & Video:**
        - **Underexposed photos**: Rescue dark photos
        - **Foggy images**: Cut through haze
        - **Backlit subjects**: See shadowed faces
        - **Night photography**: Reveal hidden details
        
        **Real Example:** Wedding photo with bride backlit (face too dark) 
        → equalization reveals facial details!
        """)
    
    with col2:
        st.warning("""
        **🛰️ Satellite & Aerial Imagery:**
        - **Terrain mapping**: See elevation changes
        - **Agriculture**: Identify crop health
        - **Disaster response**: Assess damage clearly
        - **Urban planning**: Map infrastructure
        
        **Real Example:** Satellite image of flooded area with poor contrast 
        → equalization clearly shows water vs land boundaries!
        """)
        
        st.error("""
        **🔒 Security & Forensics:**
        - **Surveillance footage**: Enhance dark videos
        - **License plates**: Read blurry plates
        - **Fingerprints**: Enhance partial prints
        - **Documents**: Recover faded text
        
        **Real Example:** Grainy security footage of robbery 
        → equalization reveals suspect's face clearly enough for ID!
        """)
    
    # Interactive challenges
    st.markdown("---")
    st.markdown("### 🎮 Try These Experiments!")
    
    challenges = [
        "**Compare Detail Levels**: Use the region selector above - which regions improve most? (Hint: shadowed areas!)",
        "**Histogram Hunt**: Look at the before/after histograms - can you see how the peak spreads out?",
        "**Contrast Test**: Try all three image types (low/normal/high contrast) - which improves most?",
        "**Range Check**: Compare the 'Range Used' numbers - equalized should be closer to 255!",
        "**Visual Challenge**: In the side-by-side view, can you spot details in the equalized version that were hidden before?"
    ]
    
    for i, challenge in enumerate(challenges, 1):
        st.markdown(f"{i}. {challenge}")
    
    # When NOT to use equalization
    st.markdown("---")
    st.markdown("### ⚠️ When Equalization DOESN'T Help")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("""
        **❌ Don't Use For:**
        
        1. **Already High Contrast Images**
           - Won't improve much
           - May over-process
        
        2. **Artistic Photos with Mood**
           - Dark/moody aesthetic gets destroyed
           - Removes intentional atmosphere
        
        3. **Color Photos (on RGB directly)**
           - Can cause color shifts
           - Better to process luminance channel only
        
        4. **Images with Important Shadows**
           - Loss of depth perception
           - "Flat" looking result
        """)
    
    with col2:
        st.success("""
        **✅ Best For:**
        
        1. **Low Contrast/Washed Out Images**
           - Foggy photos
           - Underexposed images
        
        2. **Medical/Scientific Images**
           - X-rays, CT, MRI
           - Microscopy images
        
        3. **Document Scanning**
           - Faded text
           - Poor lighting conditions
        
        4. **Surveillance/Security**
           - Dark footage
           - Low-quality cameras
        """)
    
    # Simple explanation
    with st.expander("🧮 The Simple Math (No Calculus!)"):
        st.markdown("""
        ### Breaking It Down:
        
        **Step 1: Count Pixels (Histogram)**
        ```
        How many pixels are brightness 0? → 50 pixels
        How many pixels are brightness 1? → 45 pixels
        How many pixels are brightness 2? → 60 pixels
        ... and so on for all 256 brightness levels
        ```
        
        **Step 2: Running Total (Cumulative Sum)**
        ```
        Up to brightness 0: 50 pixels
        Up to brightness 1: 50 + 45 = 95 pixels
        Up to brightness 2: 95 + 60 = 155 pixels
        ... and so on
        ```
        
        **Step 3: Remap Using the Running Total**
        ```
        Total pixels = 10,000
        
        Old brightness 0 had 50 pixels (0.5% of image)
        → Map to: 0.5% × 255 = 1.3 ≈ 1 (new brightness)
        
        Old brightness 100 had 5,000 pixels (50% of image)
        → Map to: 50% × 255 = 127.5 ≈ 128 (new brightness)
        
        Old brightness 200 had 9,500 pixels (95% of image)
        → Map to: 95% × 255 = 242 (new brightness)
        ```
        
        **The Result:**
        - Pixels that were rare (few in histogram) → stay extreme (very dark or very bright)
        - Pixels that were common (many in histogram) → get spread out across middle ranges
        - Overall effect: **Full use of 0-255 range! Better contrast!**
        
        ### In One Sentence:
        **"Give me what percentage of pixels are darker than this one, 
        then make that percentage the new brightness."**
        """)
    
    # Code example
    with st.expander("💻 Python Code: DIY Histogram Equalization"):
        st.code("""
import numpy as np
from PIL import Image

# Load grayscale image
img = np.array(Image.open('photo.jpg').convert('L'))

# Step 1: Calculate histogram (count pixels at each brightness)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Step 2: Calculate cumulative sum
cdf = hist.cumsum()

# Step 3: Normalize to 0-255 range
cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

# Step 4: Map old brightnesses to new using lookup table
equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized)
equalized = equalized.reshape(img.shape).astype(np.uint8)

# Save result
Image.fromarray(equalized).save('equalized.jpg')

# OR use OpenCV's built-in function (much faster!):
import cv2
equalized = cv2.equalizeHist(img)
        """, language="python")
    
    st.markdown("---")
    st.caption("💡 Fun Fact: Histogram equalization was invented in the 1970s for improving space probe images from Mars! NASA needed to see terrain details in poorly-lit photos. Now it's in every photo editor! 🚀")