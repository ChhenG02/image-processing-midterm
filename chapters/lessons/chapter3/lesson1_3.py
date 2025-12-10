import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def app():
    st.title("🎓 Understanding Convolution: From Zero to Hero")
    
    # Progressive learning tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ The Basic Idea", 
        "2️⃣ See It In Action", 
        "3️⃣ Control the Blur",
        "4️⃣ Why It Matters",
        "5️⃣ Advanced Play"
    ])
    
    # Load image
    try:
        img = Image.open("public/image1.png").convert('L')
    except:
        try:
            img = Image.open("public/image2.jpg").convert('L')
        except:
            img = Image.open("public/ch1.1.jpg").convert('L')
    
    img_np = np.array(img).astype(float)
    
    # ==================== TAB 1: THE BASIC IDEA ====================
    with tab1:
        st.header("🤔 What is Convolution? (In Plain English)")
        
        st.markdown("""
        ### Imagine You're at a Party...
        
        You're standing in a crowded room. Someone asks: **"What's the vibe here?"**
        
        You can't just look at ONE person - you look around at your **neighbors**:
        - Guy on left: 😊 happy (8/10)
        - Girl on right: 😄 very happy (9/10)  
        - Person behind: 😐 neutral (5/10)
        - You: 😃 happy (7/10)
        
        **Your answer:** "The average mood is about **7.25/10** - pretty good!"
        
        ---
        
        ### That's Exactly What Convolution Does! 🎉
        
        For **every pixel** in an image:
        1. 👀 Look at its **neighbors** (the pixels around it)
        2. 📊 Calculate their **average** value
        3. ✨ Replace the pixel with that average
        4. 🔄 Move to the next pixel and **repeat**
        
        **Result:** Smooth, blurred image - just like averaging out the party vibes! 🎊
        """)
        
        st.markdown("---")
        
        # Simple visual demonstration
        st.markdown("### 📱 A Simple Example:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**BEFORE (Sharp pixels)**")
            # Create example grid
            example = np.array([
                [100, 100, 100],
                [100, 200, 100],
                [100, 100, 100]
            ])
            st.markdown("""
            ```
            100  100  100
            100  200  100
            100  100  100
            ```
            """)
            st.caption("Center pixel = 200 (very different from neighbors)")
            
        with col2:
            st.markdown("**↓ AVERAGING ↓**")
            st.markdown("""
            Take the **9 pixels**:
            
            100+100+100+100+**200**+100+100+100+100
            
            = 1000
            
            **1000 ÷ 9 = 111**
            """)
            st.caption("Divide by 9 to get average")
            
        with col3:
            st.markdown("**AFTER (Smooth)**")
            st.markdown("""
            ```
            100  100  100
            100  111  100
            100  100  100
            ```
            """)
            st.caption("Center pixel = 111 (now closer to neighbors!)")
        
        st.success("""
        ✨ **The Magic:** The "sharp" pixel (200) became smoother (111) by averaging with neighbors!
        
        Do this for **every pixel** → entire image becomes **smooth/blurred**! 🌫️
        """)
        
        st.markdown("---")
        
        # Key concept
        st.info("""
        ### 🎯 Key Concept to Remember:
        
        **Convolution = Neighborhood Averaging**
        
        - **Input:** Sharp image with different pixel values
        - **Process:** Average each pixel with its neighbors  
        - **Output:** Smooth/blurred image
        
        **That's it!** Everything else is just details about HOW BIG the neighborhood is! 📏
        """)
    
    # ==================== TAB 2: SEE IT IN ACTION ====================
    with tab2:
        st.header("👀 See Convolution in Action")
        
        st.markdown("""
        Now let's see it work on a **REAL pixel** from our actual image!
        """)
        
        # Pick a pixel to demonstrate
        h, w = img_np.shape
        demo_y, demo_x = h // 2, w // 2
        
        st.markdown(f"### 🎯 Let's Process Pixel at Position ({demo_x}, {demo_y})")
        
        # Step 1: Show original pixel
        st.markdown("#### Step 1️⃣: Look at the Original Pixel")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Show single pixel
            pixel_view = np.full((100, 100), img_np[demo_y, demo_x], dtype=np.uint8)
            st.image(pixel_view, caption=f"Original Pixel", use_container_width=True, channels="L")
        
        with col2:
            st.metric("Brightness Value", f"{img_np[demo_y, demo_x]:.0f} / 255")
            st.markdown(f"""
            This is just **one pixel** at position ({demo_x}, {demo_y}).
            
            Its brightness is **{img_np[demo_y, demo_x]:.0f}** out of 255.
            """)
        
        # Step 2: Show neighborhood
        st.markdown("---")
        st.markdown("#### Step 2️⃣: Look at Its 3×3 Neighborhood (9 pixels total)")
        
        # Get 3x3 neighborhood
        pad = 1
        padded = np.pad(img_np, pad, mode='edge')
        neighborhood = padded[demo_y:demo_y+3, demo_x:demo_x+3]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Show 3x3 region
            neighborhood_large = np.repeat(np.repeat(neighborhood, 30, axis=0), 30, axis=1)
            st.image(neighborhood_large.astype(np.uint8), caption="3×3 Neighborhood", 
                    use_container_width=True, channels="L")
        
        with col2:
            st.markdown("**The 9 pixel values:**")
            
            # Show as a nice table
            neigh_df = pd.DataFrame(neighborhood, 
                                   columns=['Left', 'Center', 'Right'],
                                   index=['Top', 'Middle', 'Bottom'])
            st.dataframe(neigh_df.style.format("{:.0f}").highlight_max(axis=None, color='lightgreen')
                        .highlight_min(axis=None, color='lightcoral'), 
                        use_container_width=True)
            
            st.caption("🟢 Green = brightest, 🔴 Red = darkest")
        
        # Step 3: Calculate average
        st.markdown("---")
        st.markdown("#### Step 3️⃣: Calculate the Average")
        
        avg = neighborhood.mean()
        total = neighborhood.sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**The Math:**")
            st.code(f"""
Sum of all 9 pixels:
{neighborhood[0,0]:.0f} + {neighborhood[0,1]:.0f} + {neighborhood[0,2]:.0f} +
{neighborhood[1,0]:.0f} + {neighborhood[1,1]:.0f} + {neighborhood[1,2]:.0f} +
{neighborhood[2,0]:.0f} + {neighborhood[2,1]:.0f} + {neighborhood[2,2]:.0f}
= {total:.0f}

Average = {total:.0f} ÷ 9 = {avg:.1f}
            """)
        
        with col2:
            st.markdown("**The Result:**")
            st.metric("Original Center Pixel", f"{neighborhood[1,1]:.0f}")
            st.metric("New Averaged Pixel", f"{avg:.0f}", 
                     delta=f"{avg - neighborhood[1,1]:.0f} change")
            
            if abs(avg - neighborhood[1,1]) < 5:
                st.info("Small change - neighbors were similar!")
            else:
                st.success("Noticeable smoothing - neighbors were different!")
        
        # Step 4: Show result
        st.markdown("---")
        st.markdown("#### Step 4️⃣: The Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pixel_before = np.full((100, 100), neighborhood[1,1], dtype=np.uint8)
            st.image(pixel_before, caption="Before", use_container_width=True, channels="L")
            st.metric("Value", f"{neighborhood[1,1]:.0f}")
        
        with col2:
            st.markdown("### →")
            st.markdown("### 🔄")
            st.markdown("### Average")
        
        with col3:
            pixel_after = np.full((100, 100), avg, dtype=np.uint8)
            st.image(pixel_after, caption="After", use_container_width=True, channels="L")
            st.metric("Value", f"{avg:.0f}")
        
        st.success("""
        ✅ **Done!** We just blurred ONE pixel by averaging it with its neighbors!
        
        Now imagine doing this for **EVERY pixel** in the image → **Full image blur!** 🌫️
        """)
        
        st.error("""
        ⚠️ **Important:** This happens to ALL pixels simultaneously:
        - Pixel (0,0) gets averaged with its neighbors
        - Pixel (1,0) gets averaged with its neighbors  
        - Pixel (2,0) gets averaged with its neighbors
        - ... and so on for ALL pixels! 🔄
        """)
    
    # ==================== TAB 3: CONTROL THE BLUR ====================
    with tab3:
        st.header("🎛️ Control the Blur Strength")
        
        st.markdown("""
        ### The SIZE of the neighborhood matters! 📏
        
        - **Small neighborhood (3×3)** = average with 9 pixels → **light blur** 🌤️
        - **Medium neighborhood (7×7)** = average with 49 pixels → **medium blur** ☁️
        - **Large neighborhood (11×11)** = average with 121 pixels → **heavy blur** 🌫️
        """)
        
        # Blur control
        st.markdown("---")
        st.markdown("### Try Different Blur Strengths:")
        
        blur_option = st.select_slider(
            "Neighborhood Size:",
            options=[
                "No Blur (1×1 = 1 pixel)",
                "Tiny (3×3 = 9 pixels)",
                "Small (5×5 = 25 pixels)",
                "Medium (7×7 = 49 pixels)",
                "Large (9×9 = 81 pixels)",
                "Huge (11×11 = 121 pixels)",
                "Extreme (15×15 = 225 pixels)"
            ],
            value="Small (5×5 = 25 pixels)"
        )
        
        # Parse kernel size
        size_map = {
            "No Blur (1×1 = 1 pixel)": 1,
            "Tiny (3×3 = 9 pixels)": 3,
            "Small (5×5 = 25 pixels)": 5,
            "Medium (7×7 = 49 pixels)": 7,
            "Large (9×9 = 81 pixels)": 9,
            "Huge (11×11 = 121 pixels)": 11,
            "Extreme (15×15 = 225 pixels)": 15
        }
        
        k_size = size_map[blur_option]
        
        # Apply blur function
        def apply_blur(image, size):
            if size == 1:
                return image.astype(np.uint8)
            
            kernel = np.ones((size, size)) / (size * size)
            pad = size // 2
            padded = np.pad(image, pad, mode='edge')
            
            h, w = image.shape
            output = np.zeros_like(image)
            
            for i in range(h):
                for j in range(w):
                    output[i, j] = np.sum(padded[i:i+size, j:j+size] * kernel)
            
            return np.clip(output, 0, 255).astype(np.uint8)
        
        # Apply the blur
        with st.spinner(f"Applying {k_size}×{k_size} blur..."):
            blurred = apply_blur(img_np, k_size)
        
        # Show results
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_np.astype(np.uint8), caption="🔍 Original (Sharp)", 
                    use_container_width=True, channels="L")
            st.metric("Detail Level", f"{img_np.std():.0f}", help="Higher = more detail")
        
        with col2:
            st.image(blurred, caption=f"🌫️ Blurred ({k_size}×{k_size})", 
                    use_container_width=True, channels="L")
            detail_loss = ((img_np.std() - blurred.std()) / img_np.std() * 100)
            st.metric("Detail Level", f"{blurred.std():.0f}", 
                     delta=f"-{detail_loss:.0f}%", delta_color="inverse")
        
        # Explanation
        if k_size == 1:
            st.info("ℹ️ No blur applied - image unchanged!")
        elif k_size <= 5:
            st.success(f"✅ Light blur - averaging {k_size*k_size} pixels. Details mostly preserved!")
        elif k_size <= 9:
            st.warning(f"⚠️ Medium blur - averaging {k_size*k_size} pixels. Noticeable smoothing!")
        else:
            st.error(f"🔥 Heavy blur - averaging {k_size*k_size} pixels. Lots of detail lost!")
        
        # Gallery comparison
        st.markdown("---")
        st.markdown("### 📊 Comparison Gallery")
        st.markdown("See all blur levels at once:")
        
        gallery_sizes = [1, 3, 5, 7, 11]
        cols = st.columns(len(gallery_sizes))
        
        for idx, size in enumerate(gallery_sizes):
            with cols[idx]:
                sample = apply_blur(img_np, size)
                st.image(sample, caption=f"{size}×{size}", use_container_width=True, channels="L")
                if size == 1:
                    st.caption("Original")
                else:
                    loss = ((img_np.std() - sample.std()) / img_np.std() * 100)
                    st.caption(f"-{loss:.0f}% detail")
    
    # ==================== TAB 4: WHY IT MATTERS ====================
    with tab4:
        st.header("🌍 Why Do We Blur Images?")
        
        st.markdown("""
        ### "Wait... why would I WANT to make my image blurry?" 🤔
        
        Great question! Here are real reasons:
        """)
        
        # Use cases
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### ✅ Good Reasons to Blur:
            
            **1. 🔇 Remove Noise**
            - Your phone photos in dark = grainy/noisy
            - Blur smooths out the grain
            - "Noise reduction" in photo apps = blur!
            
            **2. 🎨 Artistic Effects**
            - Bokeh (blurred background in portraits)
            - Frosted glass effects
            - Dreamy/soft focus looks
            
            **3. 🔒 Privacy Protection**
            - Blur faces in crowd photos
            - Blur license plates in videos
            - Blur sensitive documents
            
            **4. 🤖 Prepare for AI**
            - Computer vision needs smooth images
            - Too much detail = AI gets confused
            - Blur first → better AI results!
            """)
        
        with col2:
            st.info("""
            ### 🏥 Real-World Examples:
            
            **Medical Imaging:**
            - MRI scans are noisy → blur to see tumors clearly
            - X-rays have grain → blur to see fractures
            
            **Smartphone Cameras:**
            - "Portrait mode" = blur background artificially
            - "Night mode" = blur to reduce noise
            - "Beauty mode" = blur skin (controversial!)
            
            **Video Conferencing:**
            - Zoom/Teams blur background
            - Hides messy room behind you!
            - Same blur technique!
            
            **Self-Driving Cars:**
            - Camera feeds are noisy
            - Blur first to find lanes clearly
            - Smooth = safer driving!
            """)
        
        st.warning("""
        ### ⚠️ When NOT to Blur:
        
        ❌ Don't blur if you need:
        - **Read text** (blur makes text unreadable!)
        - **See fine details** (blur removes small features)
        - **Measure precisely** (blur changes edges)
        - **Recognize faces** (AI needs sharp features)
        
        **Remember:** Blur = permanent detail loss! You can't "un-blur" accurately! 🚫
        """)
    
    # ==================== TAB 5: ADVANCED PLAY ====================
    with tab5:
        st.header("🧪 Advanced: Build Your Own Filter!")
        
        st.markdown("""
        ### Ready to experiment? 🚀
        
        So far we've been using "box blur" where **every neighbor has equal weight**.
        
        But what if we could **weight some neighbors more than others**? 🤔
        """)
        
        # Filter type selector
        filter_type = st.radio(
            "Choose a filter type:",
            ["Box Blur (all equal)", "Center-Weighted", "Edge Detection", "Sharpen", "Custom"]
        )
        
        # Generate kernel based on type
        k_size = 3
        
        if filter_type == "Box Blur (all equal)":
            kernel = np.ones((3, 3)) / 9
            explanation = "Every pixel weighted equally (1/9 each) → smooth blur"
            
        elif filter_type == "Center-Weighted":
            kernel = np.array([
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
            ]) / 16
            explanation = "Center weighted more (4/16) than corners (1/16) → gentler blur"
            
        elif filter_type == "Edge Detection":
            kernel = np.array([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ])
            explanation = "Highlights differences = finds edges! (not a blur!)"
            
        elif filter_type == "Sharpen":
            kernel = np.array([
                [ 0, -1,  0],
                [-1,  5, -1],
                [ 0, -1,  0]
            ])
            explanation = "Opposite of blur - enhances edges and details!"
            
        else:  # Custom
            st.markdown("**Design your own 3×3 filter:**")
            kernel = np.zeros((3, 3))
            
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    for j in range(3):
                        kernel[j, i] = st.number_input(
                            f"[{j},{i}]",
                            value=1.0 if j==1 and i==1 else 0.0,
                            step=0.1,
                            key=f"custom_{j}_{i}"
                        )
            
            if st.checkbox("Auto-normalize (make sum = 1)"):
                if kernel.sum() != 0:
                    kernel = kernel / kernel.sum()
            
            explanation = "Your custom filter - experiment and see what happens!"
        
        # Show kernel
        st.markdown("---")
        st.markdown("### 🔢 The Filter (Kernel):")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            kernel_df = pd.DataFrame(kernel)
            st.dataframe(kernel_df.style.format("{:.3f}").background_gradient(cmap='RdYlGn', axis=None),
                        use_container_width=True, hide_index=True)
        
        with col2:
            st.info(f"**What this does:** {explanation}")
            st.markdown(f"**Sum of weights:** {kernel.sum():.3f}")
            if abs(kernel.sum() - 1.0) < 0.01:
                st.success("✅ Normalized (sum = 1) - brightness preserved!")
            elif kernel.sum() > 1:
                st.warning("⚠️ Sum > 1 - result will be brighter!")
            elif kernel.sum() < 1:
                st.warning("⚠️ Sum < 1 - result will be darker!")
        
        # Apply filter
        if st.button("🎬 Apply This Filter!", type="primary"):
            # Manual convolution
            pad = 1
            padded = np.pad(img_np, pad, mode='edge')
            h, w = img_np.shape
            output = np.zeros_like(img_np)
            
            for i in range(h):
                for j in range(w):
                    output[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
            
            result = np.clip(output, 0, 255).astype(np.uint8)
            
            # Show result
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_np.astype(np.uint8), caption="Original", 
                        use_container_width=True, channels="L")
            
            with col2:
                st.image(result, caption="After Filter", 
                        use_container_width=True, channels="L")
            
            st.success("✅ Filter applied! Try different filters to see different effects!")
    
    # Quick reference at bottom
    st.markdown("---")
    st.info("""
    ### 📚 Quick Reference:
    
    **Convolution** = Slide a window over image and calculate weighted average
    
    **Box Blur** = Special case where all weights are equal (simple average)
    
    **Kernel/Filter** = The weights used (3×3, 5×5, etc.)
    
    **Larger kernel** = More neighbors averaged = More blur
    
    **Different weights** = Different effects (blur, sharpen, edges, etc.)
    """)