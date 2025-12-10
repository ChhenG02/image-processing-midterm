import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import math

def app():
    st.title("🔍 Understanding Edge Detection: Finding Borders in Images")
    
    # Progressive learning tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ The Basic Idea", 
        "2️⃣ See It In Action", 
        "3️⃣ Control the Edges",
        "4️⃣ Why It Matters",
        "5️⃣ Advanced Play"
    ])
    
    # Load image
    try:
        img = Image.open("public/image1.png").convert('L')
    except:
        try:
            img = Image.open("public/image2.png").convert('L')
        except:
            try:
                img = Image.open("public/image3.png").convert('L')
            except:
                img = Image.open("public/ch1.1.jpg").convert('L')
    
    img_np = np.array(img).astype(float)
    
    # ==================== TAB 1: THE BASIC IDEA ====================
    with tab1:
        st.header("🤔 What is Edge Detection? (In Plain English)")
        
        st.markdown("""
        ### Imagine You're Blindfolded at a Table...
        
        You can't see, but you want to know: **"Where are the edges of this table?"**
        
        What do you do? You **feel for sudden changes**:
        - Move your hand slowly across the surface... 👉 smooth, smooth, smooth...
        - Suddenly you feel a **DROP** 📉 - that's an edge!
        - The bigger the drop, the stronger the edge!
        
        ---
        
        ### That's Exactly What Edge Detection Does! 🎉
        
        For **every pixel** in an image:
        1. 👀 Compare it with **neighbors** around it
        2. 📊 Look for **sudden brightness changes**
        3. ⚡ Big change = **Strong edge!**
        4. 😴 Small change = No edge, just smooth area
        
        **Result:** A map showing where all the edges/borders are! 🗺️
        """)
        
        st.markdown("---")
        
        # Simple visual demonstration
        st.markdown("### 📱 A Simple Example:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**BEFORE (Pixel values)**")
            example = np.array([
                [50, 50, 50],
                [50, 50, 50],
                [50, 200, 200]
            ])
            st.markdown("""
            ```
             50   50   50
             50   50   50
             50  200  200
            ```
            """)
            st.caption("Notice the sudden jump from 50 → 200!")
            
        with col2:
            st.markdown("**➡️ DIFFERENCE ➡️**")
            st.markdown("""
            Looking at center pixel (50):
            
            - **Left neighbor**: 50 (difference = 0)
            - **Right neighbor**: 200 (difference = 150!) 🔥
            - **Top neighbor**: 50 (difference = 0)
            - **Bottom neighbor**: 200 (difference = 150!) 🔥
            
            **Big differences = EDGE detected!**
            """)
            
        with col3:
            st.markdown("**AFTER (Edge strength)**")
            st.markdown("""
            ```
              0    0    0
              0  150  150
              0  150  150
            ```
            """)
            st.caption("High values = strong edges found!")
        
        st.success("""
        ✨ **The Magic:** Where pixels **differ a lot** from neighbors = **EDGE**!
        
        Do this for **every pixel** → find **ALL edges** in the image! 🎯
        """)
        
        st.markdown("---")
        
        # Key concept
        st.info("""
        ### 🎯 Key Concept to Remember:
        
        **Edge Detection = Finding Sudden Changes**
        
        - **Input:** Image with different brightness areas
        - **Process:** Compare each pixel with neighbors  
        - **Output:** Map of where changes happen (edges!)
        
        **Convolution finds similarity (blur), Edge Detection finds DIFFERENCE!** 🔄
        """)
        
        # Visual comparison
        st.markdown("---")
        st.markdown("### 🎨 Quick Visual: Blur vs Edge Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**😊 Blur (Convolution)**")
            st.markdown("""
            ```
            [1  1  1]     Average everything
            [1  1  1]  →  Makes it SMOOTH
            [1  1  1]     Removes differences
            ```
            """)
            st.caption("All positive → adds up → smooths")
            
        with col2:
            st.markdown("**🔍 Edge Detection (Sobel)**")
            st.markdown("""
            ```
            [-1  0  +1]    Subtract left from right
            [-2  0  +2] →  Finds DIFFERENCES
            [-1  0  +1]    Highlights changes
            ```
            """)
            st.caption("Negative & positive → finds changes")
    
    # ==================== TAB 2: SEE IT IN ACTION ====================
    with tab2:
        st.header("👀 See Edge Detection in Action")
        
        st.markdown("""
        Now let's detect edges on a **REAL pixel** from our image!
        
        We'll use **Sobel operators** - special filters designed to find edges.
        """)
        
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=float)
        
        sobel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=float)
        
        # Pick a pixel to demonstrate
        h, w = img_np.shape
        demo_y, demo_x = h // 2, w // 2
        
        st.markdown(f"### 🎯 Let's Process Pixel at Position ({demo_x}, {demo_y})")
        
        # Step 1: Show original neighborhood
        st.markdown("#### Step 1️⃣: Look at the 3×3 Neighborhood")
        
        pad = 1
        padded = np.pad(img_np, pad, mode='edge')
        neighborhood = padded[demo_y:demo_y+3, demo_x:demo_x+3]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            neighborhood_large = np.repeat(np.repeat(neighborhood, 30, axis=0), 30, axis=1)
            st.image(neighborhood_large.astype(np.uint8), caption="3×3 Neighborhood", 
                    use_container_width=True, channels="L")
        
        with col2:
            st.markdown("**The 9 pixel brightness values:**")
            
            neigh_df = pd.DataFrame(neighborhood, 
                                   columns=['Left', 'Center', 'Right'],
                                   index=['Top', 'Middle', 'Bottom'])
            st.dataframe(neigh_df.style.format("{:.0f}").background_gradient(cmap='Greys', vmin=0, vmax=255), 
                        use_container_width=True)
            
            st.caption("Darker = lower values, Lighter = higher values")
        
        # Step 2: Apply horizontal edge detection
        st.markdown("---")
        st.markdown("#### Step 2️⃣: Check for HORIZONTAL Edges (Left → Right changes)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Horizontal Sobel (Gx):**")
            st.code("""[-1  0  +1]
[-2  0  +2]
[-1  0  +1]""")
            st.caption("Compares LEFT vs RIGHT")
        
        with col2:
            st.markdown("**The Calculation:**")
            
            # Calculate Gx
            right_side = (neighborhood[0,2]*1 + neighborhood[1,2]*2 + neighborhood[2,2]*1)
            left_side = (neighborhood[0,0]*1 + neighborhood[1,0]*2 + neighborhood[2,0]*1)
            gx_value = right_side - left_side
            
            st.code(f"""
RIGHT side pixels:
  {neighborhood[0,2]:.0f}×(+1) + {neighborhood[1,2]:.0f}×(+2) + {neighborhood[2,2]:.0f}×(+1) = {right_side:.0f}

LEFT side pixels:
  {neighborhood[0,0]:.0f}×(-1) + {neighborhood[1,0]:.0f}×(-2) + {neighborhood[2,0]:.0f}×(-1) = {left_side:.0f}

Gx = {right_side:.0f} - {left_side:.0f} = {gx_value:.0f}
            """)
            
            if abs(gx_value) > 50:
                st.success(f"✅ **Strong horizontal edge!** (value = {abs(gx_value):.0f})")
            elif abs(gx_value) > 20:
                st.info(f"📊 Moderate horizontal edge (value = {abs(gx_value):.0f})")
            else:
                st.warning(f"😴 Weak/no horizontal edge (value = {abs(gx_value):.0f})")
        
        # Step 3: Apply vertical edge detection
        st.markdown("---")
        st.markdown("#### Step 3️⃣: Check for VERTICAL Edges (Top → Bottom changes)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Vertical Sobel (Gy):**")
            st.code("""[-1 -2 -1]
[ 0  0  0]
[+1 +2 +1]""")
            st.caption("Compares TOP vs BOTTOM")
        
        with col2:
            st.markdown("**The Calculation:**")
            
            # Calculate Gy
            bottom_side = (neighborhood[2,0]*1 + neighborhood[2,1]*2 + neighborhood[2,2]*1)
            top_side = (neighborhood[0,0]*1 + neighborhood[0,1]*2 + neighborhood[0,2]*1)
            gy_value = bottom_side - top_side
            
            st.code(f"""
BOTTOM pixels:
  {neighborhood[2,0]:.0f}×(+1) + {neighborhood[2,1]:.0f}×(+2) + {neighborhood[2,2]:.0f}×(+1) = {bottom_side:.0f}

TOP pixels:
  {neighborhood[0,0]:.0f}×(-1) + {neighborhood[0,1]:.0f}×(-2) + {neighborhood[0,2]:.0f}×(-1) = {top_side:.0f}

Gy = {bottom_side:.0f} - {top_side:.0f} = {gy_value:.0f}
            """)
            
            if abs(gy_value) > 50:
                st.success(f"✅ **Strong vertical edge!** (value = {abs(gy_value):.0f})")
            elif abs(gy_value) > 20:
                st.info(f"📊 Moderate vertical edge (value = {abs(gy_value):.0f})")
            else:
                st.warning(f"😴 Weak/no vertical edge (value = {abs(gy_value):.0f})")
        
        # Step 4: Combine into final edge strength
        st.markdown("---")
        st.markdown("#### Step 4️⃣: Combine to Get TOTAL Edge Strength")
        
        magnitude = math.sqrt(gx_value**2 + gy_value**2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**The Math (Pythagorean Theorem!):**")
            st.code(f"""
Gx = {gx_value:.0f}  (horizontal)
Gy = {gy_value:.0f}  (vertical)

Magnitude = √(Gx² + Gy²)
          = √({gx_value:.0f}² + {gy_value:.0f}²)
          = √({gx_value**2:.0f} + {gy_value**2:.0f})
          = √{gx_value**2 + gy_value**2:.0f}
          = {magnitude:.1f}
            """)
        
        with col2:
            st.markdown("**Edge Strength Result:**")
            
            # Visualize magnitude
            if magnitude > 100:
                strength_bar = "🔥🔥🔥🔥🔥"
                strength_text = "VERY STRONG EDGE"
                color = "red"
            elif magnitude > 50:
                strength_bar = "🔥🔥🔥"
                strength_text = "Strong edge"
                color = "orange"
            elif magnitude > 20:
                strength_bar = "🔥"
                strength_text = "Weak edge"
                color = "blue"
            else:
                strength_bar = "😴"
                strength_text = "No edge (smooth area)"
                color = "gray"
            
            st.metric("Edge Magnitude", f"{magnitude:.0f}", strength_bar)
            st.markdown(f"**Interpretation:** :{color}[{strength_text}]")
            
            # Direction
            if abs(gx_value) > abs(gy_value):
                direction = "Mostly VERTICAL edge (left-right change)"
            elif abs(gy_value) > abs(gx_value):
                direction = "Mostly HORIZONTAL edge (top-bottom change)"
            else:
                direction = "DIAGONAL edge (both directions)"
            
            st.info(f"**Direction:** {direction}")
        
        st.success("""
        ✅ **Done!** We found the edge strength for ONE pixel!
        
        Now imagine doing this for **EVERY pixel** → **Full edge map!** 🗺️
        """)
        
        st.info("""
        💡 **Key Insight:**
        - Gx finds edges that go UP-DOWN (vertical edges)
        - Gy finds edges that go LEFT-RIGHT (horizontal edges)  
        - Magnitude combines both → total edge strength
        - Direction tells us which way the edge points
        """)
    
    # ==================== TAB 3: CONTROL THE EDGES ====================
    with tab3:
        st.header("🎛️ Control Edge Detection")
        
        st.markdown("""
        ### Threshold = How Sensitive Should We Be? 🎚️
        
        Not all edges are equally important:
        - **Low threshold** = detect even tiny changes → see everything (noisy) 🌪️
        - **High threshold** = only detect big changes → see major edges only 🎯
        """)
        
        # Manual convolution function
        def manual_convolution(image, kernel):
            h, w = image.shape
            kh, kw = kernel.shape
            pad_h, pad_w = kh // 2, kw // 2
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            output = np.zeros_like(image, dtype=float)
            
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kh, j:j+kw]
                    output[i, j] = np.sum(region * kernel)
            
            return output
        
        # Apply Sobel
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
        
        with st.spinner("Detecting edges..."):
            gx = manual_convolution(img_np, sobel_x)
            gy = manual_convolution(img_np, sobel_y)
            magnitude = np.sqrt(gx**2 + gy**2)
            magnitude_display = np.clip(magnitude, 0, 255).astype(np.uint8)
        
        # Threshold control
        st.markdown("---")
        st.markdown("### 🎚️ Adjust Edge Sensitivity:")
        
        threshold = st.slider(
            "Threshold Value (higher = only strong edges)",
            min_value=0,
            max_value=255,
            value=50,
            help="Pixels with edge strength above this value are marked as edges"
        )
        
        edges_binary = (magnitude_display > threshold).astype(np.uint8) * 255
        
        # Show results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(img_np.astype(np.uint8), caption="🖼️ Original Image", 
                    use_container_width=True, channels="L")
            st.caption("Source image")
        
        with col2:
            st.image(magnitude_display, caption="📊 Edge Strength Map", 
                    use_container_width=True, channels="L")
            st.caption(f"Brighter = stronger edge")
            st.metric("Avg Edge Strength", f"{magnitude_display.mean():.0f}/255")
        
        with col3:
            st.image(edges_binary, caption=f"✅ Final Edges (T={threshold})", 
                    use_container_width=True, channels="L")
            edge_pixels = np.sum(edges_binary > 0)
            total_pixels = h * w
            st.metric("Edge Pixels", f"{edge_pixels/total_pixels*100:.1f}%")
        
        # Explanation
        if threshold < 30:
            st.warning("""
            ⚠️ **Low threshold** - Very sensitive!
            - Detects even small changes
            - More edges visible
            - Can be noisy/messy
            """)
        elif threshold < 80:
            st.success("""
            ✅ **Good balance** - Moderate sensitivity
            - Catches important edges
            - Filters out noise
            - Clean edge map
            """)
        else:
            st.info("""
            🎯 **High threshold** - Only strong edges
            - Only major boundaries
            - Very clean
            - May miss some details
            """)
        
        # Show individual gradients
        st.markdown("---")
        st.markdown("### 🧭 See Horizontal vs Vertical Edges Separately")
        
        if st.checkbox("Show Gx and Gy separately"):
            gx_display = np.clip(np.abs(gx), 0, 255).astype(np.uint8)
            gy_display = np.clip(np.abs(gy), 0, 255).astype(np.uint8)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(gx_display, caption="↔️ Gx (Vertical Edges)", 
                        use_container_width=True, channels="L")
                st.caption("Finds things like: | | |")
            
            with col2:
                st.image(gy_display, caption="↕️ Gy (Horizontal Edges)", 
                        use_container_width=True, channels="L")
                st.caption("Finds things like: ===")
            
            with col3:
                st.image(magnitude_display, caption="🔄 Combined (All Edges)", 
                        use_container_width=True, channels="L")
                st.caption("Finds ALL edge directions")
        
        # Gallery comparison
        st.markdown("---")
        st.markdown("### 📊 Threshold Comparison Gallery")
        
        gallery_thresholds = [0, 25, 50, 100, 150]
        cols = st.columns(len(gallery_thresholds))
        
        for idx, t in enumerate(gallery_thresholds):
            with cols[idx]:
                sample = (magnitude_display > t).astype(np.uint8) * 255
                st.image(sample, caption=f"T={t}", use_container_width=True, channels="L")
                edge_pct = np.sum(sample > 0) / (h * w) * 100
                st.caption(f"{edge_pct:.1f}% edges")
    
    # ==================== TAB 4: WHY IT MATTERS ====================
    with tab4:
        st.header("🌍 Why Do We Need Edge Detection?")
        
        st.markdown("""
        ### "Why not just look at the image myself?" 🤔
        
        Because computers can't "see" like we do! They need help finding what's important.
        """)
        
        # Use cases
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### ✅ Real-World Applications:
            
            **1. 🚗 Self-Driving Cars**
            - Find lane markings on roads
            - Detect pedestrians and obstacles
            - Recognize traffic signs
            - **Why edges?** Lanes = edges on pavement!
            
            **2. 📱 Face Recognition (FaceID)**
            - Find face outline
            - Detect eyes, nose, mouth boundaries
            - Match face shape
            - **Why edges?** Features have distinct edges!
            
            **3. 🏥 Medical Imaging**
            - Find tumor boundaries in MRI scans
            - Detect broken bones in X-rays
            - Trace blood vessels
            - **Why edges?** Organs have clear borders!
            
            **4. 📄 Document Scanning**
            - Detect page edges to crop
            - Find text boundaries (OCR)
            - Separate handwriting from background
            - **Why edges?** Text is just dark/light edges!
            """)
        
        with col2:
            st.info("""
            ### 🎯 Why Edges Are So Important:
            
            **Edges = Boundaries = Objects!**
            
            Think about it:
            - A face = collection of edges (eyes, nose, mouth)
            - A car = rectangular edges + wheel circles
            - A cat = fuzzy outline edges
            - Text = letter-shaped edges
            
            **Humans recognize objects by their SHAPE.**  
            **Shape = EDGES!** 🎨
            
            ---
            
            ### 💡 Cool Examples:
            
            **Instagram Filters:**
            - "Sketch" filter = edge detection!
            - Makes photos look like drawings
            - Just shows the edges!
            
            **QR Code Readers:**
            - Find the square edges first
            - Then read the pattern inside
            - Edge detection = first step!
            
            **Video Games:**
            - "Toon shader" / cel-shading
            - Draws black lines around objects
            - Those lines = detected edges!
            
            **Google Lens:**
            - Point camera at text → translates it
            - First: detect text edges
            - Then: recognize letters
            - Edge detection = foundation!
            """)
        
        st.markdown("---")
        
        # Visual example of why edges matter
        st.markdown("### 🎨 See For Yourself: Humans Recognize Edges Too!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Full Image:**")
            st.image(img_np.astype(np.uint8), use_container_width=True, channels="L")
            st.caption("You can see everything clearly")
        
        with col2:
            st.markdown("**ONLY Edges:**")
            edges_only = (magnitude_display > 50).astype(np.uint8) * 255
            st.image(edges_only, use_container_width=True, channels="L")
            st.caption("Can you still recognize what it is? 🤯")
        
        st.success("""
        🧠 **Mind-Blowing Fact:** Your brain can recognize objects from JUST the edges!
        
        Even though we removed all the colors, textures, and details - you can probably still 
        tell what the image shows. That's because **edges contain most of the information** 
        about shape and structure!
        
        That's why edge detection is so powerful for computers! 🚀
        """)
    
    # ==================== TAB 5: ADVANCED PLAY ====================
    with tab5:
        st.header("🧪 Advanced: Explore Edge Detection Parameters!")
        
        st.markdown("""
        ### Ready to experiment? 🚀
        
        Let's play with different edge detection techniques and settings!
        """)
        
        # Edge detector selector
        detector_type = st.radio(
            "Choose edge detection method:",
            ["Sobel (Standard)", "Prewitt (Alternative)", "Scharr (Enhanced)", "Simple Gradient"]
        )
        
        # Define kernels
        if detector_type == "Sobel (Standard)":
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
            explanation = "Standard Sobel - good balance of noise reduction and accuracy"
            
        elif detector_type == "Prewitt (Alternative)":
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)
            explanation = "Prewitt - simpler, gives equal weight to all neighbors"
            
        elif detector_type == "Scharr (Enhanced)":
            kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=float)
            kernel_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=float)
            explanation = "Scharr - more accurate for angle/direction detection"
            
        else:  # Simple Gradient
            kernel_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=float)
            kernel_y = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=float)
            explanation = "Simple gradient - fastest, but noisier results"
        
        # Show kernels
        st.markdown("---")
        st.markdown("### 🔢 The Kernels Being Used:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Horizontal (Gx):**")
            kernel_x_df = pd.DataFrame(kernel_x)
            st.dataframe(kernel_x_df.style.format("{:.0f}").background_gradient(cmap='RdYlGn', axis=None),
                        use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Vertical (Gy):**")
            kernel_y_df = pd.DataFrame(kernel_y)
            st.dataframe(kernel_y_df.style.format("{:.0f}").background_gradient(cmap='RdYlGn', axis=None),
                        use_container_width=True, hide_index=True)
        
        with col3:
            st.info(f"**Method:** {explanation}")
        
        # Manual convolution function
        def manual_convolution(image, kernel):
            h, w = image.shape
            kh, kw = kernel.shape
            pad_h, pad_w = kh // 2, kw // 2
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            output = np.zeros_like(image, dtype=float)
            
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kh, j:j+kw]
                    output[i, j] = np.sum(region * kernel)
            
            return output
        
        # Apply selected detector
        if st.button("🎬 Apply This Detector!", type="primary"):
            with st.spinner("Detecting edges..."):
                gx = manual_convolution(img_np, kernel_x)
                gy = manual_convolution(img_np, kernel_y)
                magnitude = np.sqrt(gx**2 + gy**2)
                result = np.clip(magnitude, 0, 255).astype(np.uint8)
            
            # Show result
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_np.astype(np.uint8), caption="Original", 
                        use_container_width=True, channels="L")
            
            with col2:
                st.image(result, caption=f"Edges - {detector_type}", 
                        use_container_width=True, channels="L")
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Edge Strength", f"{result.mean():.1f}")
            
            with col2:
                st.metric("Max Edge Strength", f"{result.max():.0f}")
            
            with col3:
                strong_edges = np.sum(result > 100)
                st.metric("Strong Edges", f"{strong_edges:,} pixels")
            
            st.success("✅ Edge detection complete! Try different methods to compare!")
        
        # Edge direction visualization
        st.markdown("---")
        st.markdown("### 🧭 Bonus: Edge Direction Visualization")
        
        if st.checkbox("Show edge directions (color-coded)"):
            st.info("""
            **What this shows:** Each edge is colored based on its DIRECTION:
            - 🔴 Red = horizontal edges (→)
            - 🟡 Yellow = diagonal edges (↗)
            - 🟢 Green = vertical edges (↑)
            - 🔵 Blue = diagonal edges (↖)
            
            **How to read it:** The color tells you which way the brightness is changing!
            """)
            
            # Calculate direction using Sobel
            gx = manual_convolution(img_np, sobel_x)
            gy = manual_convolution(img_np, sobel_y)
            magnitude = np.sqrt(gx**2 + gy**2)
            direction = np.arctan2(gy, gx)  # Angle in radians
            
            # Create color image
            h, w = img_np.shape
            direction_display = np.zeros((h, w, 3), dtype=np.uint8)
            threshold_dir = 50  # Only show directions for strong edges
            
            for i in range(h):
                for j in range(w):
                    if magnitude[i, j] > threshold_dir:
                        angle = direction[i, j]  # -π to π
                        # Convert angle to hue (0-180 for OpenCV/Matplotlib compatibility)
                        hue = ((angle + math.pi) / (2 * math.pi)) * 180
                        
                        # Create HSL color
                        if hue < 45 or hue >= 135:  # Red for horizontal edges
                            direction_display[i, j] = [255, 100, 100]  # Red
                        elif 45 <= hue < 90:  # Yellow for diagonal
                            direction_display[i, j] = [255, 255, 100]  # Yellow
                        elif 90 <= hue < 135:  # Green for vertical
                            direction_display[i, j] = [100, 255, 100]  # Green
                        else:  # Blue for other diagonal
                            direction_display[i, j] = [100, 100, 255]  # Blue
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(direction_display, caption="Edge Directions (Color-coded)", 
                        use_container_width=True)
            
            with col2:
                st.markdown("""
                **Color Guide:**
                
                - **🔴 Red** - Mostly horizontal edges
                  *Examples:* Horizon lines, table edges
                  
                - **🟡 Yellow** - Diagonal edges (45°)
                  *Examples:* Roofs, mountain slopes
                  
                - **🟢 Green** - Mostly vertical edges  
                  *Examples:* Tree trunks, building sides
                  
                - **🔵 Blue** - Diagonal edges (135°)
                  *Examples:* Shadows, perspective lines
                  
                ---
                
                **Fun Experiment:**
                Rotate your head sideways - what was red (horizontal) becomes green (vertical)! 
                Edge direction is RELATIVE to the image orientation! 🔄
                """)
        
        # Compare all methods
        st.markdown("---")
        st.markdown("### 📊 Compare All Edge Detection Methods")
        
        if st.button("🔄 Run All Detectors Comparison"):
            st.markdown("Let's compare how different edge detectors perform on the same image:")
            
            # Define all detectors
            detectors = {
                "Sobel": (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
                         np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)),
                "Prewitt": (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
                           np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)),
                "Scharr": (np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=float),
                          np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=float)),
                "Simple": (np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=float),
                          np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=float))
            }
            
            # Calculate results
            results = {}
            stats = []
            
            for name, (kx, ky) in detectors.items():
                gx = manual_convolution(img_np, kx)
                gy = manual_convolution(img_np, ky)
                magnitude = np.sqrt(gx**2 + gy**2)
                result = np.clip(magnitude, 0, 255).astype(np.uint8)
                results[name] = result
                
                # Calculate statistics
                stats.append({
                    "Method": name,
                    "Mean": f"{result.mean():.1f}",
                    "Std Dev": f"{result.std():.1f}",
                    "Max": f"{result.max():.0f}",
                    "Strong Edges": f"{np.sum(result > 100):,}"
                })
            
            # Display results in grid
            cols = st.columns(4)
            for idx, (name, result) in enumerate(results.items()):
                with cols[idx]:
                    st.image(result, caption=name, use_container_width=True, channels="L")
            
            # Show statistics table
            st.markdown("**📈 Performance Comparison:**")
            stats_df = pd.DataFrame(stats)
            st.dataframe(stats_df, use_container_width=True)
            
            st.info("""
            **Which detector is "best"?** It depends!
            - **Sobel:** Good all-rounder (most popular)
            - **Prewitt:** Simpler, less accurate  
            - **Scharr:** More accurate for angles
            - **Simple:** Fastest, but noisy
            
            Different applications need different detectors! 🤔
            """)
        
        # Edge detection challenge
        st.markdown("---")
        st.markdown("### 🎯 Edge Detection Challenge")
        
        st.markdown("""
        **Test Your Understanding!**
        
        Try to predict what will happen when we:
        1. Apply edge detection to a completely uniform image (all pixels same value)
        2. Apply edge detection to a noisy image
        3. Rotate the image 90° before edge detection
        """)
        
        challenge_option = st.selectbox(
            "Choose a challenge:",
            ["No Challenge", "Uniform Image", "Noisy Image", "Rotated Image"]
        )
        
        if challenge_option != "No Challenge":
            with st.spinner("Creating challenge..."):
                if challenge_option == "Uniform Image":
                    # Create uniform image
                    challenge_img = np.ones_like(img_np) * 128
                    title = "Completely Uniform Gray Image"
                    prediction = "What edges will we find in an image with NO brightness changes?"
                    
                elif challenge_option == "Noisy Image":
                    # Add noise
                    challenge_img = img_np + np.random.normal(0, 30, img_np.shape)
                    challenge_img = np.clip(challenge_img, 0, 255)
                    title = "Very Noisy Image"
                    prediction = "Will edge detection work well with lots of noise?"
                    
                else:  # Rotated Image
                    # Rotate 90°
                    challenge_img = np.rot90(img_np)
                    title = "Image Rotated 90°"
                    prediction = "How will edge directions change when we rotate?"
                
                # Apply Sobel
                kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
                ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
                
                gx = manual_convolution(challenge_img, kx)
                gy = manual_convolution(challenge_img, ky)
                magnitude = np.sqrt(gx**2 + gy**2)
                result = np.clip(magnitude, 0, 255).astype(np.uint8)
                
                # Display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(challenge_img.astype(np.uint8), caption=title, 
                            use_container_width=True, channels="L")
                
                with col2:
                    st.image(result, caption="Edge Detection Result", 
                            use_container_width=True, channels="L")
                
                st.markdown(f"**🤔 Prediction:** {prediction}")
                
                # Reveal answer after a moment
                if st.button("🔍 Reveal Answer"):
                    if challenge_option == "Uniform Image":
                        st.success("""
                        **Answer:** NO EDGES will be detected!
                        
                        Edge detection looks for **differences**. If all pixels are exactly the same:
                        - Every pixel has same value as neighbors
                        - Differences = 0 everywhere
                        - Result = completely black image (no edges)
                        
                        **Lesson:** Edges require brightness CHANGES! ⚡
                        """)
                    elif challenge_option == "Noisy Image":
                        st.warning("""
                        **Answer:** LOTS of fake edges will be detected!
                        
                        Noise creates random brightness changes everywhere:
                        - Each noisy pixel differs from neighbors
                        - Many small "edges" detected
                        - Result = noisy, hard to find real edges
                        
                        **Lesson:** Edge detectors are sensitive to noise! 
                        Often need to blur first to reduce noise. 🌪️
                        """)
                    else:  # Rotated Image
                        st.info("""
                        **Answer:** Edge directions swap!
                        
                        When you rotate an image 90°:
                        - What was horizontal becomes vertical
                        - What was vertical becomes horizontal
                        - Gx and Gy swap roles
                        
                        **Lesson:** Edge direction is RELATIVE to image orientation! 
                        Gx always finds horizontal changes (left-right), 
                        Gy always finds vertical changes (top-bottom) in the CURRENT orientation. 🔄
                        """)
        
        # Summary
        st.markdown("---")
        st.markdown("### 🎓 What You've Learned")
        
        st.success("""
        **🏆 Congratulations! You've mastered edge detection!**
        
        **Key Takeaways:**
        1. **Edges = sudden brightness changes** 📈
        2. **Sobel operators** find edges in X and Y directions
        3. **Gradient magnitude** combines them for total edge strength
        4. **Threshold** controls sensitivity
        5. **Different detectors** have different strengths
        
        **Remember:** Edge detection is often the FIRST STEP in computer vision!
        After finding edges, computers can then:
        - Connect edges to find shapes 🟦
        - Recognize objects from those shapes 👤
        - Make decisions based on what they "see" 🤖
        
        **You're now ready to explore more advanced topics like:**
        - Canny Edge Detector (even better!)
        - Hough Transform (find lines/circles in edges)
        - Feature Detection (find corners/interesting points)
        - Object Recognition (identify what objects are)
        
        Keep exploring! The world of computer vision is fascinating! 🌟
        """)
        
        # Next steps
        with st.expander("📚 Want to Learn More?"):
            st.markdown("""
            **Recommended Next Topics:**
            
            **1. Canny Edge Detector** - Even better edge detection
            - Uses Sobel internally
            - Adds non-maximum suppression (thins edges)
            - Uses hysteresis thresholding (connects edges)
            
            **2. Laplacian Edge Detector** - Finds edges differently
            - Looks for zero-crossings instead of gradients
            - Responds to intensity changes in any direction
            - Often used for image sharpening too!
            
            **3. Hough Transform** - Find lines and circles in edges
            - Takes edge map as input
            - Finds geometric shapes in the edges
            - Used for lane detection in self-driving cars
            
            **4. Corner Detection** (Harris, Shi-Tomasi)
            - Finds "interesting points" where edges meet
            - Used for feature matching and tracking
            - Foundation for many computer vision algorithms
            
            **Fun Project Ideas:**
            - Build a simple document scanner
            - Create a sketch filter for photos
            - Make a lane detection demo
            - Build a simple face finder
            """)