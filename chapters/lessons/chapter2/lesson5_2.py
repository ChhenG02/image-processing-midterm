import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def app():
    st.title("🎞️ Negative Image Transformation")
    
    # Simple analogy first
    st.markdown("""
    ### 🤔 The Core Idea (Simple Analogy)
    
    Remember old film cameras? They produced **negatives** where:
    - ☀️ **Bright things** (like the sky) looked **dark** on film
    - 🌑 **Dark things** (like shadows) looked **bright** on film
    - 🎨 **Colors were reversed** (orange became blue, etc.)
    
    **The Math is Super Simple:**
    - Imagine a brightness scale from 0 (black) to 255 (white)
    - **Negative = 255 - Original**
    - If pixel is 0 (black) → becomes 255 (white)
    - If pixel is 255 (white) → becomes 0 (black)
    - If pixel is 100 (dark gray) → becomes 155 (light gray)
    
    **It's like flipping the brightness scale upside down!** ⬆️➡️⬇️
    """)
    
    # Visual scale diagram
    st.markdown("#### 📊 Brightness Scale Flip")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("**Original Scale:**")
        st.markdown("0 = ⬜ Black")
        st.markdown("128 = ◽ Gray")
        st.markdown("255 = ⬛ White")
    with col2:
        st.markdown("**→ FLIP →**")
        st.markdown("255 - value")
        st.markdown("🔄")
    with col3:
        st.markdown("**Negative Scale:**")
        st.markdown("255 = ⬛ White")
        st.markdown("127 = ◽ Gray")
        st.markdown("0 = ⬜ Black")
    
    st.markdown("---")
    
    # Load image
    img = Image.open("public/ch1.1.jpg")
    img_np = np.array(img)
    is_color = len(img_np.shape) == 3
    
    # Simple controls
    st.markdown("### 🎛️ Create Your Negative")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_color:
            mode = st.radio(
                "What to invert:",
                ["Everything (Full Negative)", "Only Red Channel", "Only Green Channel", 
                 "Only Blue Channel", "Custom Selection"],
                help="Choose which parts to flip"
            )
        else:
            mode = "Everything (Full Negative)"
            st.info("Grayscale image detected - will invert all brightness values")
    
    with col2:
        show_formula = st.checkbox("📐 Show Live Formula", value=True)
        show_examples = st.checkbox("👀 Show Before/After Pixels", value=True)
    
    # Custom channel selection
    if is_color and mode == "Custom Selection":
        st.markdown("#### 🎨 Select Channels to Invert:")
        col1, col2, col3 = st.columns(3)
        with col1:
            invert_red = st.checkbox("🔴 Red", value=True)
        with col2:
            invert_green = st.checkbox("🟢 Green", value=True)
        with col3:
            invert_blue = st.checkbox("🔵 Blue", value=True)
    
    # Create negative
    negative = img_np.copy().astype(float)
    
    if is_color:
        if mode == "Everything (Full Negative)":
            negative = 255 - negative
            formula = "Result = 255 - Original (all channels)"
        elif mode == "Only Red Channel":
            negative[:, :, 0] = 255 - negative[:, :, 0]
            formula = "Red = 255 - Red | Green & Blue unchanged"
        elif mode == "Only Green Channel":
            negative[:, :, 1] = 255 - negative[:, :, 1]
            formula = "Green = 255 - Green | Red & Blue unchanged"
        elif mode == "Only Blue Channel":
            negative[:, :, 2] = 255 - negative[:, :, 2]
            formula = "Blue = 255 - Blue | Red & Green unchanged"
        elif mode == "Custom Selection":
            channels = []
            if invert_red:
                negative[:, :, 0] = 255 - negative[:, :, 0]
                channels.append("Red")
            if invert_green:
                negative[:, :, 1] = 255 - negative[:, :, 1]
                channels.append("Green")
            if invert_blue:
                negative[:, :, 2] = 255 - negative[:, :, 2]
                channels.append("Blue")
            formula = f"Inverted: {', '.join(channels) if channels else 'None'}"
    else:
        negative = 255 - negative
        formula = "Result = 255 - Original"
    
    negative = negative.astype(np.uint8)
    
    if show_formula:
        st.code(formula, language=None)
    
    # Main comparison
    st.markdown("---")
    st.markdown("### 📸 Original vs Negative")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img_np, caption="🌟 Original Image", use_container_width=True)
        if is_color:
            avg_r, avg_g, avg_b = img_np[:,:,0].mean(), img_np[:,:,1].mean(), img_np[:,:,2].mean()
            st.markdown(f"""
            **Average Colors:**  
            🔴 Red: {avg_r:.0f}  
            🟢 Green: {avg_g:.0f}  
            🔵 Blue: {avg_b:.0f}
            """)
        else:
            avg_bright = img_np.mean()
            st.markdown(f"**Average Brightness:** {avg_bright:.0f}/255")
    
    with col2:
        st.image(negative, caption="🎞️ Negative Image", use_container_width=True)
        if is_color:
            avg_r_n, avg_g_n, avg_b_n = negative[:,:,0].mean(), negative[:,:,1].mean(), negative[:,:,2].mean()
            st.markdown(f"""
            **Average Colors:**  
            🔴 Red: {avg_r_n:.0f} ({255-avg_r:.0f} expected)  
            🟢 Green: {avg_g_n:.0f} ({255-avg_g:.0f} expected)  
            🔵 Blue: {avg_b_n:.0f} ({255-avg_b:.0f} expected)
            """)
        else:
            avg_bright_n = negative.mean()
            st.markdown(f"**Average Brightness:** {avg_bright_n:.0f}/255 ({255-avg_bright:.0f} expected)")
    
    # Side by side
    st.markdown("---")
    st.markdown("### 🔄 Side-by-Side Comparison")
    
    if is_color:
        composite = np.concatenate([img_np, negative], axis=1)
    else:
        composite = np.concatenate([img_np, negative], axis=1)
    
    st.image(composite, caption="Left: Original | Right: Negative", use_container_width=True)
    
    # Pixel examples
    if show_examples:
        st.markdown("---")
        st.markdown("### 🔬 How It Works: Pixel-by-Pixel Examples")
        
        # Pick sample pixels
        h, w = img_np.shape[:2]
        sample_points = [
            (h//4, w//4, "Dark Area", "🌑"),
            (h//2, w//2, "Medium Area", "◽"),
            (3*h//4, 3*w//4, "Bright Area", "☀️")
        ]
        
        pixel_data = []
        
        for y, x, area_name, emoji in sample_points:
            if is_color:
                orig = img_np[y, x]
                neg = negative[y, x]
                
                pixel_data.append({
                    "Area": f"{emoji} {area_name}",
                    "Original RGB": f"({orig[0]}, {orig[1]}, {orig[2]})",
                    "Math": f"255 - Original",
                    "Negative RGB": f"({neg[0]}, {neg[1]}, {neg[2]})",
                    "Check": f"({255-orig[0]}, {255-orig[1]}, {255-orig[2]})",
                    "✓": "✅" if np.array_equal(neg, 255-orig) else "❌"
                })
            else:
                orig = img_np[y, x]
                neg = negative[y, x]
                
                pixel_data.append({
                    "Area": f"{emoji} {area_name}",
                    "Original": f"{orig}",
                    "Math": "255 - Original",
                    "Negative": f"{neg}",
                    "Expected": f"{255-orig}",
                    "✓": "✅" if neg == 255-orig else "❌"
                })
        
        df = pd.DataFrame(pixel_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.caption("💡 Notice how dark pixels become bright and bright pixels become dark!")
    
    # Interactive pixel inspector
    st.markdown("---")
    st.markdown("### 🔍 Interactive Pixel Inspector")
    st.markdown("Click around the image to see how any pixel gets inverted:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_pos = st.slider("👈 Horizontal Position (X)", 0, w-1, w//2)
    with col2:
        y_pos = st.slider("👆 Vertical Position (Y)", 0, h-1, h//2)
    
    if is_color:
        orig_pixel = img_np[y_pos, x_pos]
        neg_pixel = negative[y_pos, x_pos]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Create visual swatch
            swatch = np.full((100, 100, 3), orig_pixel, dtype=np.uint8)
            st.image(swatch, caption="Original Pixel Color", use_container_width=True)
            st.markdown(f"""
            **RGB Values:**  
            🔴 R: {orig_pixel[0]}  
            🟢 G: {orig_pixel[1]}  
            🔵 B: {orig_pixel[2]}
            """)
        
        with col2:
            st.markdown("### ➡️")
            st.markdown("### 255 - RGB")
            st.markdown("### =")
        
        with col3:
            swatch_neg = np.full((100, 100, 3), neg_pixel, dtype=np.uint8)
            st.image(swatch_neg, caption="Negative Pixel Color", use_container_width=True)
            st.markdown(f"""
            **RGB Values:**  
            🔴 R: {neg_pixel[0]} (= 255-{orig_pixel[0]})  
            🟢 G: {neg_pixel[1]} (= 255-{orig_pixel[1]})  
            🔵 B: {neg_pixel[2]} (= 255-{orig_pixel[2]})
            """)
    else:
        orig_val = img_np[y_pos, x_pos]
        neg_val = negative[y_pos, x_pos]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            swatch = np.full((100, 100), orig_val, dtype=np.uint8)
            st.image(swatch, caption="Original Pixel", use_container_width=True, channels="L")
            st.metric("Brightness", f"{orig_val}/255")
        
        with col2:
            st.markdown("### ➡️")
            st.markdown(f"### 255 - {orig_val}")
            st.markdown(f"### = {255-orig_val}")
        
        with col3:
            swatch_neg = np.full((100, 100), neg_val, dtype=np.uint8)
            st.image(swatch_neg, caption="Negative Pixel", use_container_width=True, channels="L")
            st.metric("Brightness", f"{neg_val}/255")
    
    # Color channel showcase (for color images)
    if is_color:
        st.markdown("---")
        st.markdown("### 🎨 Channel-by-Channel Breakdown")
        st.markdown("See what happens when you invert only one color channel:")
        
        # Create all variations
        variations = {
            "Original": img_np,
            "All Inverted": 255 - img_np,
            "Red Only": img_np.copy(),
            "Green Only": img_np.copy(),
            "Blue Only": img_np.copy()
        }
        
        variations["Red Only"][:, :, 0] = 255 - variations["Red Only"][:, :, 0]
        variations["Green Only"][:, :, 1] = 255 - variations["Green Only"][:, :, 1]
        variations["Blue Only"][:, :, 2] = 255 - variations["Blue Only"][:, :, 2]
        
        cols = st.columns(5)
        for idx, (name, img_var) in enumerate(variations.items()):
            with cols[idx]:
                st.image(img_var.astype(np.uint8), caption=name, use_container_width=True)
                if name == "Original":
                    st.caption("No inversion")
                elif name == "All Inverted":
                    st.caption("All channels inverted")
                else:
                    st.caption(f"Only {name.split()[0]} inverted")
    
    # Real-world applications
    st.markdown("---")
    st.markdown("### 🌍 Why Invert Images? Real Applications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🏥 Medical Imaging:**
        - **X-rays**: Negative view often shows fractures better
        - **MRI scans**: Inverting can highlight different tissues
        - **Microscopy**: Reveals cellular structures more clearly
        - **CT scans**: Bone vs soft tissue contrast
        
        **Example:** Radiologists often toggle between positive and negative 
        X-rays to catch details they might miss in just one view!
        """)
        
        st.info("""
        **🔬 Scientific Research:**
        - **Astronomy**: Reveal faint stars and galaxies
        - **Microscopy**: Enhance specimen details
        - **Spectroscopy**: Analyze absorption patterns
        - **Material science**: Study surface features
        
        **Example:** The famous "Pillars of Creation" photo from Hubble 
        looks amazing in negative - shows different nebula structures!
        """)
    
    with col2:
        st.warning("""
        **📄 Document Processing:**
        - **Old documents**: Enhance faded text
        - **Photocopies**: Improve readability
        - **Blueprints**: Traditional architecture drawings
        - **OCR prep**: Better text recognition
        
        **Example:** Libraries often scan old manuscripts in negative 
        to make degraded ink more readable for digitization!
        """)
        
        st.error("""
        **🎨 Creative & Artistic:**
        - **Photo effects**: Surreal, otherworldly looks
        - **Film aesthetics**: Vintage film camera look
        - **Dark mode**: Reduce eye strain for viewing
        - **Contrast boost**: Make details pop
        
        **Example:** Many video editors use negative effects in 
        music videos and artistic films for dramatic impact!
        """)
    
    # Interactive challenges
    st.markdown("---")
    st.markdown("### 🎮 Try These Experiments!")
    
    challenges = [
        "**Find Hidden Details**: Compare original vs negative - do you see different details? Dark areas often reveal more in negative!",
        "**Color Opposites**: For color images, try inverting just ONE channel (red, green, or blue) - creates wild color effects!",
        "**Brightness Check**: Pick a very dark pixel (0-50) and see it become bright (205-255). Then try a bright pixel!",
        "**Half & Half**: Look at the side-by-side view - can you identify the same features in both versions?",
        "**Eye Strain Test**: If you're in a dark room, which version is easier on your eyes? (Hint: negative acts like dark mode!)"
    ]
    
    for i, challenge in enumerate(challenges, 1):
        st.markdown(f"{i}. {challenge}")
    
    # The math explained simply
    with st.expander("🧮 The Simple Math Behind It"):
        st.markdown("""
        ### Why 255?
        - Digital images store brightness as numbers from **0 to 255** (that's 256 values)
        - **0** = no light = black ⬛
        - **255** = full light = white ⬜
        - Everything between = shades of gray
        
        ### The Inversion Formula:
        ```
        Negative = 255 - Original
        ```
        
        ### Step-by-Step Examples:
        
        **Example 1: Black Pixel**
        - Original: 0 (black)
        - Calculation: 255 - 0 = 255
        - Result: 255 (white) ✅
        
        **Example 2: White Pixel**
        - Original: 255 (white)
        - Calculation: 255 - 255 = 0
        - Result: 0 (black) ✅
        
        **Example 3: Gray Pixel**
        - Original: 100 (dark gray)
        - Calculation: 255 - 100 = 155
        - Result: 155 (light gray) ✅
        
        **Example 4: Perfect Middle Gray**
        - Original: 128 (middle gray)
        - Calculation: 255 - 128 = 127
        - Result: 127 (almost same gray!) ✅
        
        ### For Color Images (RGB):
        Do it for each channel separately:
        - Negative Red = 255 - Original Red
        - Negative Green = 255 - Original Green  
        - Negative Blue = 255 - Original Blue
        
        That's it! Super simple math with powerful results! 🎉
        """)
    
    # Code example
    with st.expander("💻 Python Code: Create Negatives"):
        st.code("""
import numpy as np
from PIL import Image

# Load image
img = np.array(Image.open('photo.jpg'))

# Method 1: Full negative (super simple!)
negative = 255 - img

# Method 2: Only for grayscale
img_gray = np.array(Image.open('photo.jpg').convert('L'))
negative_gray = 255 - img_gray

# Method 3: Selective channel inversion (color only)
negative_custom = img.copy()
negative_custom[:, :, 0] = 255 - img[:, :, 0]  # Invert red only
# negative_custom[:, :, 1] = 255 - img[:, :, 1]  # Invert green
# negative_custom[:, :, 2] = 255 - img[:, :, 2]  # Invert blue

# Save result
Image.fromarray(negative).save('negative.jpg')

# That's it! One line of code: 255 - img 🎉
        """, language="python")
    
    st.markdown("---")
    st.caption("💡 Fun Fact: Before digital cameras, film photographers had to work with negatives every day! They'd develop film to get negatives, then print those negatives onto photo paper to get positive images. Your smartphone does this digitally in milliseconds! 📸")