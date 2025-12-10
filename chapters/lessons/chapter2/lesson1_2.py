import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def app():
    st.title("🧮 Image Arithmetic Operations")
    
    # Simple analogy first
    st.markdown("""
    ### 🤔 The Core Idea (Simple Analogy)
    
    Think of images as **spreadsheets of numbers**:
    - Each pixel = one cell in the spreadsheet
    - Each number represents brightness (0 = black, 255 = white)
    - Arithmetic operations work **pixel-by-pixel**, just like Excel formulas!
    
    **Example:**
    - Cell A1 = 100, Cell B1 = 50
    - Addition: C1 = A1 + B1 = 150 ✅
    - Subtraction: C1 = A1 - B1 = 50 ✅
    - **Same logic applies to every pixel in an image!**
    """)
    
    st.markdown("---")
    
    # Load images
    img1 = Image.open("public/nahida.jpg").convert('L')
    img2 = Image.open("public/raiden.jpg").convert('L')
    img2 = img2.resize(img1.size)
    
    img1_np = np.array(img1).astype(float)
    img2_np = np.array(img2).astype(float)
    
    # Controls
    st.markdown("### 🎛️ Experiment Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        operation = st.selectbox(
            "🧮 Select Operation:",
            ["Addition", "Subtraction", "Absolute Difference", "Multiplication", "Division", "Weighted Average"],
            help="Choose which mathematical operation to apply"
        )
    
    with col2:
        if operation == "Weighted Average":
            weight = st.slider("Weight for Image A", 0.0, 1.0, 0.5, 0.1,
                             help="0.0 = all Image B, 1.0 = all Image A")
        elif operation in ["Multiplication", "Division"]:
            scale_factor = st.slider("Result Scaling", 0.1, 5.0, 1.0, 0.1,
                                   help="Adjust brightness of result")
        else:
            scale_factor = 1.0
    
    with col3:
        show_histogram = st.checkbox("📊 Show Histograms", value=False)
        show_math = st.checkbox("🔢 Show Pixel Math Example", value=True)
    
    # Perform operation
    st.markdown("---")
    st.markdown("### 🎬 Operation in Action")
    
    epsilon = 1e-6
    
    if operation == "Addition":
        result = img1_np + img2_np
        formula = "Result = Image A + Image B"
        description = "Combines brightness from both images (creates brighter result)"
        
    elif operation == "Subtraction":
        result = img1_np - img2_np
        formula = "Result = Image A - Image B"
        description = "Shows what's brighter in Image A vs Image B (can be negative)"
        
    elif operation == "Absolute Difference":
        result = np.abs(img1_np - img2_np)
        formula = "Result = |Image A - Image B|"
        description = "Shows differences regardless of which is brighter (always positive)"
        
    elif operation == "Multiplication":
        result = (img1_np * img2_np) / 255.0  # Normalize
        result = result * scale_factor
        formula = "Result = (Image A × Image B) / 255"
        description = "Areas bright in BOTH images stay bright, others darken"
        
    elif operation == "Division":
        result = (img1_np / (img2_np + epsilon)) * 50  # Scale for visibility
        result = result * scale_factor
        formula = "Result = Image A ÷ Image B"
        description = "Shows ratio of brightness (highlights relative differences)"
        
    elif operation == "Weighted Average":
        result = weight * img1_np + (1 - weight) * img2_np
        formula = f"Result = {weight:.1f} × Image A + {1-weight:.1f} × Image B"
        description = "Blends the two images together with adjustable proportions"
    
    # Clip to valid range
    result_display = np.clip(result, 0, 255).astype(np.uint8)
    
    st.info(f"**{operation}:** {description}")
    st.code(formula, language=None)
    
    # Display images
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img1_np.astype(np.uint8), caption="📷 Image A", 
                use_container_width=True, channels="L")
        st.markdown(f"""
        **Range:** {int(img1_np.min())} - {int(img1_np.max())}  
        **Average:** {int(img1_np.mean())}
        """)
    
    with col2:
        st.image(img2_np.astype(np.uint8), caption="📷 Image B", 
                use_container_width=True, channels="L")
        st.markdown(f"""
        **Range:** {int(img2_np.min())} - {int(img2_np.max())}  
        **Average:** {int(img2_np.mean())}
        """)
    
    with col3:
        st.image(result_display, caption=f"✨ Result: {operation}", 
                use_container_width=True, channels="L")
        st.markdown(f"""
        **Range:** {int(result_display.min())} - {int(result_display.max())}  
        **Average:** {int(result_display.mean())}
        """)
    
    # Pixel-level math example
    if show_math:
        st.markdown("---")
        st.markdown("### 🔍 Pixel-Level Math Example")
        
        # Pick a few sample pixels
        h, w = img1_np.shape
        sample_points = [
            (h//4, w//4, "Top-Left Area"),
            (h//2, w//2, "Center"),
            (3*h//4, 3*w//4, "Bottom-Right Area")
        ]
        
        math_data = []
        for y, x, location in sample_points:
            val_a = img1_np[y, x]
            val_b = img2_np[y, x]
            
            if operation == "Addition":
                calc_result = val_a + val_b
            elif operation == "Subtraction":
                calc_result = val_a - val_b
            elif operation == "Absolute Difference":
                calc_result = abs(val_a - val_b)
            elif operation == "Multiplication":
                calc_result = (val_a * val_b) / 255.0 * scale_factor
            elif operation == "Division":
                calc_result = (val_a / (val_b + epsilon)) * 50 * scale_factor
            elif operation == "Weighted Average":
                calc_result = weight * val_a + (1 - weight) * val_b
            
            final_result = np.clip(calc_result, 0, 255)
            
            math_data.append({
                "Location": location,
                "Image A": f"{val_a:.0f}",
                "Image B": f"{val_b:.0f}",
                "Calculation": f"{calc_result:.1f}",
                "Final (clipped)": f"{final_result:.0f}"
            })
        
        df = pd.DataFrame(math_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.caption("💡 Notice how the operation is applied to each pixel independently!")
    
    # Histograms
    if show_histogram:
        st.markdown("---")
        st.markdown("### 📊 Brightness Distribution (Histograms)")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate histograms
        hist_a, bins = np.histogram(img1_np.flatten(), bins=50, range=(0, 255))
        hist_b, _ = np.histogram(img2_np.flatten(), bins=50, range=(0, 255))
        hist_result, _ = np.histogram(result_display.flatten(), bins=50, range=(0, 255))
        
        with col1:
            st.bar_chart(pd.DataFrame({'Count': hist_a}))
            st.caption("Image A Distribution")
        
        with col2:
            st.bar_chart(pd.DataFrame({'Count': hist_b}))
            st.caption("Image B Distribution")
        
        with col3:
            st.bar_chart(pd.DataFrame({'Count': hist_result}))
            st.caption("Result Distribution")
    
    # Operation-specific insights
    st.markdown("---")
    st.markdown("### 💡 What's Happening & Why It's Useful")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if operation == "Addition":
            st.success("""
            **✅ Addition Characteristics:**
            - Makes images brighter overall
            - Values can exceed 255 (get clipped)
            - Bright + Bright = Very Bright
            - Dark + Dark = Still Dark
            
            **⚠️ Watch Out:**
            Many pixels hit maximum (255) creating "washed out" areas
            """)
        elif operation == "Subtraction":
            st.success("""
            **✅ Subtraction Characteristics:**
            - Shows differences (positive and negative)
            - Negative values become black (0)
            - A > B = Bright, A < B = Dark
            - Highlights what's unique to Image A
            
            **⚠️ Watch Out:**
            Result can be very dark if images are similar
            """)
        elif operation == "Absolute Difference":
            st.success("""
            **✅ Absolute Difference Characteristics:**
            - Shows ALL differences (no negatives)
            - Symmetric: |A-B| = |B-A|
            - Similar areas = Dark
            - Different areas = Bright
            
            **⚠️ Watch Out:**
            Loses information about direction of change
            """)
        elif operation == "Multiplication":
            st.success("""
            **✅ Multiplication Characteristics:**
            - Acts like an AND operation
            - Both must be bright to stay bright
            - One dark pixel makes result dark
            - Good for masking/filtering
            
            **⚠️ Watch Out:**
            Usually produces very dark results (needs scaling)
            """)
        elif operation == "Division":
            st.success("""
            **✅ Division Characteristics:**
            - Shows relative brightness ratios
            - High ratio = A much brighter than B
            - Low ratio = B brighter than A
            - Normalizes lighting differences
            
            **⚠️ Watch Out:**
            Can create extreme values (needs careful scaling)
            """)
        elif operation == "Weighted Average":
            st.success("""
            **✅ Weighted Average Characteristics:**
            - Smoothly blends between images
            - Always produces valid range
            - No clipping issues
            - Natural-looking transitions
            
            **⚠️ Watch Out:**
            Equal weights (0.5) may look ghostly
            """)
    
    with col2:
        st.info("""
        **🌍 Real-World Applications:**
        
        **📸 Photography:**
        - HDR imaging (add exposures)
        - Panorama stitching (weighted average)
        - Ghost removal (absolute difference)
        
        **🏥 Medical Imaging:**
        - Compare scans over time (subtraction)
        - Highlight changes (absolute difference)
        - Combine modalities (weighted average)
        
        **🎥 Video Processing:**
        - Motion detection (subtraction)
        - Background removal (absolute difference)
        - Frame interpolation (weighted average)
        
        **🔬 Scientific Research:**
        - Compare experimental results
        - Normalize lighting conditions (division)
        - Combine multiple measurements
        """)
    
    # Interactive challenges
    st.markdown("---")
    st.markdown("### 🎮 Try These Experiments!")
    
    challenges = {
        "Addition": "Can you make the result completely white? What happens when both images are bright?",
        "Subtraction": "Try subtracting similar images - what do you see? Now try very different images!",
        "Absolute Difference": "Find areas where the two images are most different. What shows up bright?",
        "Multiplication": "Why is the result so dark? Adjust the scaling factor to see details!",
        "Division": "What happens in areas where Image B is very dark? See the bright spots?",
        "Weighted Average": "Slide the weight from 0 to 1 - watch the smooth transition between images!"
    }
    
    st.markdown(f"**🎯 Challenge for {operation}:**")
    st.markdown(f"_{challenges[operation]}_")
    
    # Common pitfalls
    st.markdown("---")
    st.markdown("### ⚠️ Common Pitfalls & Solutions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("""
        **❌ Common Mistakes:**
        1. **Overflow**: Adding bright images → values > 255
        2. **Underflow**: Subtracting dark images → values < 0
        3. **Dark Results**: Multiplying without scaling
        4. **Extreme Values**: Dividing by near-zero pixels
        5. **Wrong Data Types**: Using uint8 (loses negative values)
        """)
    
    with col2:
        st.success("""
        **✅ Best Practices:**
        1. **Use float32** for intermediate calculations
        2. **Clip results** to valid range [0, 255]
        3. **Scale appropriately** (especially for multiply/divide)
        4. **Add epsilon** to denominators (prevent ÷0)
        5. **Convert to uint8** only at the end
        """)
    
    # Code example
    with st.expander("💻 See the Python Code"):
        st.code("""
# Good practice for image arithmetic:

import numpy as np

# 1. Convert to float (allows negative values and values > 255)
img1 = img1.astype(float)
img2 = img2.astype(float)

# 2. Perform operation
if operation == "addition":
    result = img1 + img2
elif operation == "subtraction":
    result = img1 - img2
elif operation == "multiplication":
    result = (img1 * img2) / 255.0  # Normalize!
elif operation == "division":
    result = img1 / (img2 + 1e-6)  # Add epsilon!

# 3. Clip to valid range
result = np.clip(result, 0, 255)

# 4. Convert back to uint8
result = result.astype(np.uint8)
        """, language="python")
    
    st.markdown("---")
    st.caption("💡 Tip: Image arithmetic is the foundation of many advanced techniques like HDR, panoramas, and medical image analysis!")