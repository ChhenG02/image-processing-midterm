import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def app():
    st.title("⚡ Direct Sharpening Kernels: The Fast Way to Crisp Images")
    
    # Progressive learning tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ The Basic Idea", 
        "2️⃣ See It In Action", 
        "3️⃣ Control Sharpening",
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
    height, width = img_np.shape
    
    # ==================== TAB 1: THE BASIC IDEA ====================
    with tab1:
        st.header("🤔 What Are Sharpening Kernels? (In Plain English)")
        
        st.markdown("""
        ### The "One-Step Magic" vs "Three-Step Process" 🪄
        
        Remember unsharp masking? It had 3 steps:
        1. Blur image 🌫️
        2. Find difference 📊
        3. Add back 🔄
        
        **What if we could do all that in ONE STEP?** 🚀
        
        That's what **sharpening kernels** do!
        
        ---
        
        ### The "Center of Attention" Analogy 👑
        
        Imagine you're at a party:
        - **You** = center pixel (most important!) 👑
        - **Your friends around you** = neighbor pixels 👥
        
        **Regular photo:** You and friends blend together 😊
        **Sharpened photo:** You STAND OUT from the crowd! ✨
        
        How? The kernel says:
        - "Make YOU (center) look BRIGHTER!" ⬆️
        - "Make your friends (neighbors) look DARKER!" ⬇️
        - Result: You pop out more! 🎯
        """)
        
        st.markdown("---")
        
        # Visual metaphor
        st.markdown("### 🎨 How Kernels Work:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Pixels**")
            st.markdown("""
            ```
            [50, 50, 50]
            [50, 50, 50]  
            [50, 50, 50]
            ```
            """)
            st.caption("Everyone equal - no one stands out")
            
        with col2:
            st.markdown("**➡️ Apply Kernel ➡️**")
            st.markdown("""
            ```
            [-1, -1, -1]   Subtract neighbors
            [-1,  9, -1] → Boost center
            [-1, -1, -1]   Make center pop!
            ```
            """)
            st.caption("Center gets boost, neighbors reduced")
            
        with col3:
            st.markdown("**Result**")
            st.markdown("""
            ```
            [25, 25, 25]
            [25, 450, 25]  WOW! Center = 450!
            [25, 25, 25]
            ```
            (Clipped to 255)
            """)
            st.caption("Center pixel now stands out! ✨")
        
        st.success("""
        ✨ **The Magic Formula:**
        
        **New Pixel = (Center × Big Number) - (Neighbors × Small Numbers)**
        
        - Center gets multiplied by **5, 9, or higher**
        - Neighbors get multiplied by **-1, -2**
        - Result: **Center pops, edges sharpen!** 🎯
        """)
        
        st.markdown("---")
        
        # Key concept
        st.info("""
        ### 🎯 Key Concept to Remember:
        
        **Sharpening Kernels = One-Step Edge Enhancement**
        
        - **All in one:** No separate blur/difference steps
        - **Super fast:** Single convolution operation
        - **Same math:** Mathematically equals unsharp masking
        - **Center heavy:** Big positive weight in middle
        
        **Think of it like:** Baking a cake 🍰
        - Unsharp masking = Mix ingredients separately, then combine
        - Sharpening kernel = All ingredients mixed at once!
        """)
        
        # Simple example
        st.markdown("---")
        st.markdown("### 📱 Meet the Sharpening Family:")
        
        kernels = {
            "Basic (3×3)": "[[0,-1,0],[-1,5,-1],[0,-1,0]]",
            "Strong (3×3)": "[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]", 
            "Edge (5×5)": "5×5 with ring of -1, boost in center",
            "Laplacian": "[[0,1,0],[1,-4,1],[0,1,0]] (edge detection)"
        }
        
        for name, desc in kernels.items():
            st.markdown(f"- **{name}:** `{desc}`")
        
        st.success("""
        ✅ **Different kernels = different sharpening styles!**
        - **Basic:** Gentle, natural enhancement
        - **Strong:** Aggressive sharpening  
        - **Edge:** Smart, preserves textures
        - **Laplacian:** Pure edge enhancement
        """)
    
    # ==================== TAB 2: SEE IT IN ACTION ====================
    with tab2:
        st.header("👀 See Kernel Sharpening in Action")
        
        st.markdown("""
        Let's see exactly how a sharpening kernel works on real pixels!
        
        We'll pick a small area and watch the magic happen. ✨
        """)
        
        # Manual convolution function
        def manual_convolution(image, kernel):
            h, w = image.shape
            kh, kw = kernel.shape
            pad_h, pad_w = kh // 2, kw // 2
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
            output = np.zeros_like(image, dtype=float)
            
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kh, j:j+kw]
                    output[i, j] = np.sum(region * kernel)
            
            return np.clip(output, 0, 255)
        
        # Pick a region to demonstrate
        h, w = img_np.shape
        demo_y, demo_x = h // 2 - 10, w // 2 - 10
        
        st.markdown(f"### 🎯 Let's Process a 3×3 Region")
        
        # Step 1: Show original region
        st.markdown("#### Step 1️⃣: Look at Original Pixels")
        
        region = img_np[demo_y:demo_y+3, demo_x:demo_x+3]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            region_large = np.repeat(np.repeat(region, 40, axis=0), 40, axis=1)
            st.image(region_large.astype(np.uint8), caption="3×3 Region", 
                    use_container_width=True, channels="L")
        
        with col2:
            st.markdown("**The 9 pixel brightness values:**")
            
            region_df = pd.DataFrame(region, 
                                   columns=['Left', 'Center', 'Right'],
                                   index=['Top', 'Middle', 'Bottom'])
            st.dataframe(region_df.style.format("{:.0f}").background_gradient(cmap='Greys', vmin=0, vmax=255), 
                        use_container_width=True)
            
            st.caption("Look at center pixel (row 1, col 1) = Middle-Center")
        
        # Step 2: Choose a kernel
        st.markdown("---")
        st.markdown("#### Step 2️⃣: Choose a Sharpening Kernel")
        
        kernel_choice = st.radio(
            "Pick a kernel to apply:",
            ["Basic Sharpening (5-center)", "Strong Sharpening (9-center)", "Custom Kernel"]
        )
        
        if kernel_choice == "Basic Sharpening (5-center)":
            kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=float)
            kernel_name = "Basic (Center ×5)"
            
        elif kernel_choice == "Strong Sharpening (9-center)":
            kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]], dtype=float)
            kernel_name = "Strong (Center ×9)"
            
        else:  # Custom
            st.markdown("**Design Your Own Kernel:**")
            center_weight = st.slider("Center Weight", 1.0, 15.0, 5.0, step=0.5)
            neighbor_weight = st.slider("Neighbor Weight", -2.0, 0.0, -1.0, step=0.1)
            
            kernel = np.array([[neighbor_weight, neighbor_weight, neighbor_weight],
                              [neighbor_weight, center_weight, neighbor_weight],
                              [neighbor_weight, neighbor_weight, neighbor_weight]], dtype=float)
            kernel_name = f"Custom (Center ×{center_weight})"
        
        # Show kernel
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"**{kernel_name} Kernel:**")
            kernel_df = pd.DataFrame(kernel)
            st.dataframe(kernel_df.style.format("{:+.1f}").background_gradient(cmap='RdYlGn', vmin=-2, vmax=10), 
                        use_container_width=True, hide_index=True)
            
            st.caption(f"**Sum:** {np.sum(kernel):.1f}")
        
        with col2:
            st.markdown("**What This Kernel Does:**")
            
            center_effect = kernel[1, 1]
            neighbor_effect = kernel[0, 0]
            
            st.markdown(f"""
            - **Center pixel** gets multiplied by **{center_effect:+.1f}** ⬆️
            - **Each neighbor** gets multiplied by **{neighbor_effect:+.1f}** ⬇️
            - **Total effect:** Center boosted relative to neighbors
            
            **In plain English:**
            > "Make the center {abs(center_effect)}× stronger, 
            > while reducing neighbors {abs(neighbor_effect)}×"
            """)
        
        # Step 3: Apply kernel to center pixel
        st.markdown("---")
        st.markdown("#### Step 3️⃣: Calculate New Center Pixel")
        
        center_value = region[1, 1]
        
        # Calculate manually
        padded_region = np.pad(region, 1, mode='edge')
        kernel_region = padded_region[1:4, 1:4]  # Center 3×3
        
        st.markdown("**The Calculation:**")
        
        calculation = ""
        total = 0.0
        
        for i in range(3):
            for j in range(3):
                pixel_val = kernel_region[i, j]
                kernel_val = kernel[i, j]
                contribution = pixel_val * kernel_val
                total += contribution
                
                sign = "+" if contribution >= 0 else ""
                calculation += f"{pixel_val:.0f} × {kernel_val:+.1f} = {sign}{contribution:.1f}\n"
        
        st.code(f"""
Region multiplied by kernel:
{kernel_region[0,0]:.0f}×{kernel[0,0]:+.1f} + {kernel_region[0,1]:.0f}×{kernel[0,1]:+.1f} + {kernel_region[0,2]:.0f}×{kernel[0,2]:+.1f} = {kernel_region[0,0]*kernel[0,0] + kernel_region[0,1]*kernel[0,1] + kernel_region[0,2]*kernel[0,2]:.1f}
{kernel_region[1,0]:.0f}×{kernel[1,0]:+.1f} + {kernel_region[1,1]:.0f}×{kernel[1,1]:+.1f} + {kernel_region[1,2]:.0f}×{kernel[1,2]:+.1f} = {kernel_region[1,0]*kernel[1,0] + kernel_region[1,1]*kernel[1,1] + kernel_region[1,2]*kernel[1,2]:.1f}
{kernel_region[2,0]:.0f}×{kernel[2,0]:+.1f} + {kernel_region[2,1]:.0f}×{kernel[1,1]:+.1f} + {kernel_region[2,2]:.0f}×{kernel[2,2]:+.1f} = {kernel_region[2,0]*kernel[2,0] + kernel_region[2,1]*kernel[2,1] + kernel_region[2,2]*kernel[2,2]:.1f}

SUM = {total:.1f}
        """)
        
        new_center = min(255, max(0, total))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Original Center", f"{center_value:.0f}")
        
        with col2:
            change = new_center - center_value
            st.metric("New Center", f"{new_center:.0f}", f"{change:+.0f}")
        
        if change > 0:
            st.success(f"✅ Center pixel BRIGHTENED by {change:.0f}! That's sharpening! ✨")
        else:
            st.warning(f"⚠️ Center pixel DARKENED by {abs(change):.0f}. Try a stronger center weight!")
        
        # Step 4: Apply to entire region
        st.markdown("---")
        st.markdown("#### Step 4️⃣: Apply to All Pixels in Region")
        
        sharpened_region = manual_convolution(region, kernel)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Region:**")
            st.dataframe(pd.DataFrame(region).style.format("{:.0f}"), 
                        use_container_width=True)
        
        with col2:
            st.markdown("**Sharpened Region:**")
            st.dataframe(pd.DataFrame(sharpened_region).style.format("{:.0f}"), 
                        use_container_width=True)
        
        with col3:
            # Calculate enhancement
            orig_contrast = np.max(region) - np.min(region)
            sharp_contrast = np.max(sharpened_region) - np.min(sharpened_region)
            
            if orig_contrast > 0:
                improvement = (sharp_contrast - orig_contrast) / orig_contrast * 100
                st.metric("Contrast Boost", f"{improvement:.0f}%")
            else:
                st.info("No edges to enhance")
            
            # Find strongest enhancement
            max_change = np.max(np.abs(sharpened_region - region))
            st.metric("Max Change", f"{max_change:.0f}")
        
        # Visual comparison
        st.markdown("---")
        st.markdown("### 🔍 Visual Comparison")
        
        orig_display = np.clip(region, 0, 255).astype(np.uint8)
        sharp_display = np.clip(sharpened_region, 0, 255).astype(np.uint8)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(orig_display, caption="Original Region", 
                    use_container_width=True, channels="L")
        
        with col2:
            st.image(sharp_display, caption=f"Sharpened ({kernel_name})", 
                    use_container_width=True, channels="L")
        
        st.success("""
        ✅ **Success!** We sharpened a small region.
        
        Now imagine doing this for **EVERY 3×3 block** in the image → 
        **Whole image gets sharpened in one pass!** 🚀
        """)
        
        st.info("""
        💡 **Key Insight:**
        - Each pixel becomes: **Center × Big - Neighbors × Small**
        - Where edges exist: Center and neighbors differ → Big change!
        - Where areas are smooth: Center ≈ neighbors → Little change!
        
        **Smart:** Enhances edges without affecting smooth areas much! 🧠
        """)
    
    # ==================== TAB 3: CONTROL SHARPENING ====================
    with tab3:
        st.header("🎛️ Control Your Kernel Sharpening")
        
        st.markdown("""
        ### The Two Magic Numbers: 🎚️🎛️
        
        1. **Center Weight** = How much to boost the center pixel
           - Low (3-5) → Gentle enhancement 😊
           - Medium (5-9) → Strong sharpening 🔥
           - High (9-15) → Very aggressive ⚡
        
        2. **Neighbor Weight** = How much to reduce neighbors  
           - Near 0 (-0.5) → Mild neighbor reduction 😴
           - Medium (-1) → Standard reduction 📉
           - Strong (-2) → Heavy reduction 💥
        """)
        
        # Controls
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            center_weight = st.slider(
                "Center Weight (Boost Strength)",
                min_value=1.0,
                max_value=15.0,
                value=5.0,
                step=0.5,
                help="Bigger = stronger sharpening effect"
            )
            
            if center_weight < 3:
                st.info("😴 Very gentle - subtle enhancement")
            elif center_weight < 7:
                st.success("✅ Good balance - natural looking")
            elif center_weight < 10:
                st.warning("🔥 Strong - clear sharpening")
            else:
                st.error("⚡ Very strong - may create artifacts!")
        
        with col2:
            neighbor_weight = st.slider(
                "Neighbor Weight (Reduction)",
                min_value=-2.0,
                max_value=0.0,
                value=-1.0,
                step=0.1,
                help="More negative = stronger neighbor reduction"
            )
            
            st.markdown(f"""
            **What {neighbor_weight:.1f} means:**
            - Each neighbor multiplied by {neighbor_weight:.1f}
            - **Example:** Pixel value 100 becomes {100 * neighbor_weight:.0f}
            - Negative = reduces neighbor's influence
            """)
        
        # Create and apply kernel
        kernel = np.array([[neighbor_weight, neighbor_weight, neighbor_weight],
                          [neighbor_weight, center_weight, neighbor_weight],
                          [neighbor_weight, neighbor_weight, neighbor_weight]], dtype=float)
        
        st.markdown("---")
        st.markdown(f"### 🧮 Your Custom Kernel:")
        
        kernel_df = pd.DataFrame(kernel)
        st.dataframe(kernel_df.style.format("{:+.1f}").background_gradient(cmap='RdYlGn', vmin=-2, vmax=15), 
                    use_container_width=True, hide_index=True)
        
        kernel_sum = np.sum(kernel)
        st.caption(f"**Kernel Sum:** {kernel_sum:.2f} (Ideal: 1.0 for brightness preservation)")
        
        if abs(kernel_sum - 1.0) > 0.5:
            st.warning("⚠️ Kernel sum far from 1.0 - may change overall brightness!")
        
        # Apply to whole image
        def apply_kernel_sharpening(image, kernel):
            h, w = image.shape
            kh, kw = kernel.shape
            pad_h, pad_w = kh // 2, kw // 2
            
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
            output = np.zeros_like(image, dtype=float)
            
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kh, j:j+kw]
                    output[i, j] = np.sum(region * kernel)
            
            return np.clip(output, 0, 255).astype(np.uint8)
        
        with st.spinner(f"Sharpening with Center={center_weight}, Neighbor={neighbor_weight}..."):
            sharpened = apply_kernel_sharpening(img_np, kernel)
        
        # Show results
        st.markdown("---")
        st.markdown(f"### 📊 Sharpening Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_np.astype(np.uint8), caption="🖼️ Original Image", 
                    use_container_width=True, channels="L")
            
            orig_mean = img_np.mean()
            st.metric("Original Mean", f"{orig_mean:.1f}")
        
        with col2:
            st.image(sharpened, caption=f"✨ Sharpened Image", 
                    use_container_width=True, channels="L")
            
            sharp_mean = sharpened.mean()
            brightness_change = sharp_mean - orig_mean
            st.metric("Sharpened Mean", f"{sharp_mean:.1f}", f"{brightness_change:+.1f}")
        
        # Side-by-side comparison
        st.markdown("---")
        st.markdown("### 🔍 Direct Comparison")
        
        composite = np.concatenate([img_np.astype(np.uint8), sharpened], axis=1)
        st.image(composite, caption="Left: Original | Right: Sharpened", 
                use_container_width=True, channels="L")
        
        # Difference analysis
        st.markdown("---")
        st.markdown("### 📈 Difference Analysis")
        
        difference = sharpened.astype(float) - img_np.astype(float)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            enhanced = np.sum(difference > 0)
            st.metric("Brighter Pixels", f"{enhanced:,}", 
                     f"{enhanced/(h*w)*100:.1f}%")
        
        with col2:
            reduced = np.sum(difference < 0)
            st.metric("Darker Pixels", f"{reduced:,}", 
                     f"{reduced/(h*w)*100:.1f}%")
        
        with col3:
            avg_change = np.mean(np.abs(difference))
            st.metric("Avg Change", f"{avg_change:.1f}")
        
        with col4:
            max_change = np.max(np.abs(difference))
            st.metric("Max Change", f"{max_change:.0f}")
        
        # Kernel gallery
        st.markdown("---")
        st.markdown("### 🎨 Kernel Gallery (Common Presets)")
        
        presets = [
            {"name": "Subtle", "center": 3.0, "neighbor": -0.5, "desc": "Very gentle"},
            {"name": "Natural", "center": 5.0, "neighbor": -1.0, "desc": "Standard photo"},
            {"name": "Strong", "center": 7.0, "neighbor": -1.0, "desc": "Clear sharpening"},
            {"name": "Aggressive", "center": 9.0, "neighbor": -1.0, "desc": "Maximum effect"},
            {"name": "Edge Focus", "center": 5.0, "neighbor": -2.0, "desc": "Emphasizes edges"}
        ]
        
        cols = st.columns(len(presets))
        
        for idx, preset in enumerate(presets):
            with cols[idx]:
                if st.button(preset["name"], key=f"kernel_{idx}"):
                    st.session_state.center_weight = preset["center"]
                    st.session_state.neighbor_weight = preset["neighbor"]
                    st.rerun()
                
                # Show kernel visualization
                preset_kernel = np.array([
                    [preset["neighbor"], preset["neighbor"], preset["neighbor"]],
                    [preset["neighbor"], preset["center"], preset["neighbor"]],
                    [preset["neighbor"], preset["neighbor"], preset["neighbor"]]
                ])
                
                # Simple text representation
                center_str = f"{preset['center']:+.1f}"
                neighbor_str = f"{preset['neighbor']:+.1f}"
                
                st.markdown(f"""
                ```
                [{neighbor_str} {neighbor_str} {neighbor_str}]
                [{neighbor_str} {center_str} {neighbor_str}]
                [{neighbor_str} {neighbor_str} {neighbor_str}]
                ```
                """)
                
                st.caption(preset["desc"])
        
        # Comparison with unsharp masking
        st.markdown("---")
        st.markdown("### 🔄 Compare: Kernel vs Unsharp Masking")
        
        if st.checkbox("Show unsharp masking comparison"):
            # Create unsharp masking result
            blur_kernel = np.ones((3,3), dtype=float) / 9
            padded = np.pad(img_np, 1, mode='reflect')
            blurred = np.zeros_like(img_np, dtype=float)
            
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+3, j:j+3]
                    blurred[i, j] = np.sum(region * blur_kernel)
            
            unsharp = img_np + (img_np - blurred) * 1.0
            unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
            
            # Display comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(img_np.astype(np.uint8), caption="Original", 
                        use_container_width=True, channels="L")
            
            with col2:
                st.image(sharpened, caption=f"Kernel Method\nCenter={center_weight}", 
                        use_container_width=True, channels="L")
            
            with col3:
                st.image(unsharp, caption="Unsharp Masking\n(Amount=1.0)", 
                        use_container_width=True, channels="L")
            
            # Mathematical relationship
            st.markdown("---")
            st.markdown("### 📐 Mathematical Relationship")
            
            st.info("""
            **They're related!** For 3×3 box blur, unsharp masking with amount=1 gives:
            
            ```
            Unsharp = Original + (Original - Blurred)
            Blurred = average of 9 neighbors
            ```
            
            This is equivalent to a kernel:
            ```
            [-1/9, -1/9, -1/9]
            [-1/9, 2 - 8/9, -1/9]
            [-1/9, -1/9, -1/9]
            ```
            
            Multiply by 9 to get whole numbers:
            ```
            [-1, -1, -1]
            [-1, 10, -1]
            [-1, -1, -1]
            ```
            
            **That's very close to our kernel with center=10, neighbor=-1!** 🔍
            """)
    
    # ==================== TAB 4: WHY IT MATTERS ====================
    with tab4:
        st.header("🌍 Why Direct Kernel Sharpening Matters")
        
        st.markdown("""
        ### "Why not just use unsharp masking?" 🤔
        
        Because sometimes **SPEED MATTERS** more than flexibility! 🏃‍♂️💨
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### ⚡ When Kernel Method Wins:
            
            **1. 🎥 Real-Time Video Processing**
            - Security cameras (24/7 monitoring)
            - Live streaming (YouTube/Twitch)
            - Video conferencing (Zoom/Teams)
            - **Why kernels?** Single operation = faster!
            
            **2. 📱 Smartphone Cameras**
            - Instant photo processing
            - Live preview while shooting
            - Battery efficiency matters
            - **Why kernels?** Less computation = less power!
            
            **3. 🎮 Video Games & VR**
            - Need 60+ frames per second
            - Real-time graphics enhancement
            - VR can't have lag
            - **Why kernels?** Minimal processing delay!
            
            **4. 🚗 Automotive Systems**
            - Lane detection cameras
            - Parking assist systems
            - Dashcams
            - **Why kernels?** Hardware-friendly design!
            """)
        
        with col2:
            st.info("""
            ### 🎯 The Technical Advantage:
            
            **Kernel Method = 1 Operation**
            ```
            For each pixel:
            Load 9 neighbors
            Multiply by kernel
            Sum results
            Store back
            ```
            
            **Unsharp Masking = 3 Operations**
            ```
            1. Blur entire image
            2. Subtract blurred from original
            3. Add difference back
            ```
            
            **Result:** Kernels are ~3× faster! ⏱️
            
            ---
            
            ### 💡 Hardware Optimization:
            
            **Convolution is hardware-friendly!**
            - Easy to implement in silicon (ASIC/FPGA)
            - Parallel processing possible
            - GPUs have special convolution units
            - Mobile chips have image signal processors (ISP)
            
            **Example:** Your phone's camera chip has 
            dedicated hardware for 3×3 convolutions!
            
            ---
            
            ### ⚖️ Trade-offs:
            
            **Kernel Advantages:**
            - ✅ Super fast
            - ✅ Hardware friendly  
            - ✅ Simple to implement
            - ✅ Consistent results
            
            **Unsharp Masking Advantages:**
            - ✅ More flexible
            - ✅ Easier to tune
            - ✅ Can use better blur methods
            - ✅ Standard in photo editing
            """)
        
        # Speed demonstration
        st.markdown("---")
        st.markdown("### ⏱️ Speed Comparison Demo")
        
        if st.button("🏃 Run Speed Test"):
            import time
            
            # Time kernel method
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=float)
            
            start = time.time()
            for _ in range(10):  # Run 10 times for better measurement
                kernel_result = apply_kernel_sharpening(img_np, kernel)
            kernel_time = (time.time() - start) / 10
            
            # Time unsharp masking
            start = time.time()
            for _ in range(10):
                # Blur
                blur_kernel = np.ones((3,3), dtype=float) / 9
                padded = np.pad(img_np, 1, mode='reflect')
                blurred = np.zeros_like(img_np, dtype=float)
                
                for i in range(h):
                    for j in range(w):
                        region = padded[i:i+3, j:j+3]
                        blurred[i, j] = np.sum(region * blur_kernel)
                
                # Unsharp
                unsharp_result = img_np + (img_np - blurred)
                unsharp_result = np.clip(unsharp_result, 0, 255).astype(np.uint8)
            unsharp_time = (time.time() - start) / 10
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Kernel Method", f"{kernel_time*1000:.1f} ms", 
                         "Single convolution")
            
            with col2:
                st.metric("Unsharp Masking", f"{unsharp_time*1000:.1f} ms", 
                         f"{unsharp_time/kernel_time:.1f}× slower")
            
            st.success(f"✅ **Kernel method is {unsharp_time/kernel_time:.1f}× faster!**")
            
            # Visual check they're similar
            diff = np.mean(np.abs(kernel_result.astype(float) - unsharp_result.astype(float)))
            st.info(f"**Result similarity:** Average difference = {diff:.1f} pixels (very close!)")
        
        # Real-world example
        st.markdown("---")
        st.markdown("### 📱 Your Phone Uses This!")
        
        st.markdown("""
        **Fun Fact:** When you take a photo with your smartphone:
        
        1. 📸 Sensor captures raw image
        2. ⚡ ISP chip applies sharpening kernel
        3. 🎨 Other enhancements added
        4. 💾 Saves to photo library
        
        **The sharpening happens in REAL-TIME** as you press the shutter!
        
        **Why?** Because users expect instant results, not waiting for processing! ⌛
        """)
        
        with st.expander("🔧 How Hardware Implements This"):
            st.markdown("""
            **Specialized Hardware (ISP - Image Signal Processor):**
            
            ```
            Input: Raw pixel stream from camera
            ↓
            Hardware Convolution Unit:
            - Loads 3×3 window
            - 9 parallel multipliers
            - Fast adder tree
            - Outputs result
            ↓
            Next processing stage
            ```
            
            **This happens for EVERY PIXEL at 30+ frames per second!** 🚀
            
            **Key Features:**
            - **Pipelined:** While processing pixel N, loading pixel N+1
            - **Parallel:** All 9 multiplications at once
            - **Optimized:** Minimal memory access
            - **Low Power:** Special low-power circuits
            
            **Result:** Professional-quality sharpening in your pocket! 📱
            """)
    
    # ==================== TAB 5: ADVANCED PLAY ====================
    with tab5:
        st.header("🧪 Advanced: Experiment with Sharpening Kernels!")
        
        st.markdown("""
        Ready to become a kernel expert? Let's explore advanced concepts! 🔬
        """)
        
        # Different kernel patterns
        st.markdown("---")
        st.markdown("### 🎨 Try Different Kernel Patterns")
        
        pattern_choice = st.radio(
            "Choose a kernel pattern:",
            ["Standard 3×3", "Cross Pattern", "Diagonal Emphasis", "Large 5×5"]
        )
        
        if pattern_choice == "Standard 3×3":
            pattern_kernel = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]], dtype=float)
            pattern_name = "Full 3×3"
            
        elif pattern_choice == "Cross Pattern":
            pattern_kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]], dtype=float)
            pattern_name = "Cross (4 neighbors)"
            
        elif pattern_choice == "Diagonal Emphasis":
            pattern_kernel = np.array([[-1, 0, -1],
                                      [0, 6, 0],
                                      [-1, 0, -1]], dtype=float)
            pattern_name = "Diagonals Only"
            
        else:  # Large 5×5
            pattern_kernel = np.array([
                [0, 0, -1, 0, 0],
                [0, -1, -1, -1, 0],
                [-1, -1, 13, -1, -1],
                [0, -1, -1, -1, 0],
                [0, 0, -1, 0, 0]
            ], dtype=float)
            pattern_name = "Large 5×5"
        
        # Show pattern
        st.markdown(f"**{pattern_name} Pattern:**")
        
        pattern_df = pd.DataFrame(pattern_kernel)
        st.dataframe(pattern_df.style.format("{:+.1f}").background_gradient(cmap='RdYlGn'), 
                    use_container_width=True, hide_index=True)
        
        st.caption(f"**Sum:** {np.sum(pattern_kernel):.2f}")
        
        # Apply pattern
        if st.button("🔬 Apply This Pattern", type="primary"):
            result = apply_kernel_sharpening(img_np, pattern_kernel)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_np.astype(np.uint8), caption="Original", 
                        use_container_width=True, channels="L")
            
            with col2:
                st.image(result, caption=f"{pattern_name} Result", 
                        use_container_width=True, channels="L")
            
            # Pattern analysis
            st.markdown("---")
            st.markdown("### 🔍 Pattern Analysis")
            
            if pattern_choice == "Cross Pattern":
                st.info("""
                **Cross Pattern Insight:**
                - Only affects 4 immediate neighbors (up, down, left, right)
                - Ignores diagonal neighbors
                - Result: More natural, less aggressive
                - Good for portrait photos
                """)
            elif pattern_choice == "Diagonal Emphasis":
                st.info("""
                **Diagonal Pattern Insight:**
                - Only affects 4 diagonal neighbors
                - Preserves horizontal/vertical structures
                - Good for architectural photos
                - Enhances diagonal lines specifically
                """)
            elif pattern_choice == "Large 5×5":
                st.info("""
                **Large 5×5 Pattern Insight:**
                - Looks at wider neighborhood (25 pixels)
                - More gradual transitions
                - Less prone to artifacts
                - Better for high-resolution images
                """)
        
        # Edge case experiments
        st.markdown("---")
        st.markdown("### 🎯 Edge Case Experiments")
        
        experiment = st.selectbox(
            "Choose an experiment:",
            ["No Experiment", "What if Center = 1?", "What if All Weights Positive?", 
             "Zero-Sum Kernel?", "Extreme Kernel"]
        )
        
        if experiment != "No Experiment":
            if experiment == "What if Center = 1?":
                test_kernel = np.array([[-1, -1, -1],
                                       [-1, 1, -1],
                                       [-1, -1, -1]], dtype=float)
                explanation = """
                **Center = 1 means:** Center barely boosted, neighbors strongly reduced
                
                **Result:** Image gets DARKER overall!
                
                **Why?** Sum of kernel = 1 - 8 = -7 (negative!)
                Negative sum = overall brightness reduction
                """
                
            elif experiment == "What if All Weights Positive?":
                test_kernel = np.array([[1, 1, 1],
                                       [1, 5, 1],
                                       [1, 1, 1]], dtype=float)
                explanation = """
                **All positive means:** Center AND neighbors get boosted
                
                **Result:** Blurring instead of sharpening!
                
                **Why?** When everything gets brighter together,
                differences get reduced = less contrast = blurrier!
                """
                
            elif experiment == "Zero-Sum Kernel?":
                test_kernel = np.array([[1, -2, 1],
                                       [-2, 4, -2],
                                       [1, -2, 1]], dtype=float)
                explanation = """
                **Zero-sum means:** Positive and negative cancel exactly
                
                **Result:** Pure edge detection (like Laplacian)!
                
                **Why?** Sum = 0, so overall brightness unchanged
                Only edges (where pixels differ) get enhanced
                This is actually an edge detection kernel!
                """
                
            else:  # Extreme Kernel
                test_kernel = np.array([[-5, -5, -5],
                                       [-5, 45, -5],
                                       [-5, -5, -5]], dtype=float)
                explanation = """
                **Extreme weights:** Center ×45, neighbors ×-5
                
                **Result:** Crazy over-sharpening with halos!
                
                **Why?** Too much amplification creates:
                - Bright halos around dark edges
                - Dark halos around bright edges  
                - Overall noisy, artificial look
                - Example of what NOT to do!
                """
            
            # Apply and show
            test_result = apply_kernel_sharpening(img_np, test_kernel)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_np.astype(np.uint8), caption="Original", 
                        use_container_width=True, channels="L")
            
            with col2:
                st.image(test_result, caption=experiment, 
                        use_container_width=True, channels="L")
            
            st.info(explanation)
        
        # Interactive design challenge
        st.markdown("---")
        st.markdown("### 🎨 Design Your Perfect Kernel")
        
        st.markdown("""
        **Challenge:** Create a kernel that achieves specific goals:
        
        1. **Natural Enhancement** - Sharpen but keep natural look
        2. **Edge Emphasis** - Make edges very clear
        3. **Portrait Friendly** - Enhance features without skin texture
        """)
        
        design_cols = st.columns(3)
        
        with design_cols[0]:
            design_center = st.number_input("Center Weight", 1.0, 20.0, 5.0, 0.5)
        
        with design_cols[1]:
            design_cross = st.number_input("Cross Neighbors", -3.0, 0.0, -1.0, 0.1)
        
        with design_cols[2]:
            design_diag = st.number_input("Diagonal Neighbors", -3.0, 0.0, -1.0, 0.1)
        
        # Create asymmetric kernel
        design_kernel = np.array([
            [design_diag, design_cross, design_diag],
            [design_cross, design_center, design_cross],
            [design_diag, design_cross, design_diag]
        ], dtype=float)
        
        if st.button("✨ Test My Design"):
            design_result = apply_kernel_sharpening(img_np, design_kernel)
            
            # Evaluate design
            kernel_sum = design_kernel.sum()
            brightness_ok = abs(kernel_sum - 1.0) < 0.5
            
            # Check if it's a sharpening kernel
            is_sharpening = design_center > abs(design_cross) * 4
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Your Kernel:**")
                st.dataframe(pd.DataFrame(design_kernel).style.format("{:+.1f}"), 
                            use_container_width=True)
                
                st.markdown(f"""
                **Analysis:**
                - Sum: {kernel_sum:.2f} {'✅' if brightness_ok else '⚠️'}
                - Center vs Neighbors: {'✅ Sharpening' if is_sharpening else '⚠️ Not sharpening'}
                - Cross vs Diagonal: {'Equal' if design_cross == design_diag else 'Different'}
                """)
            
            with col2:
                st.image(design_result, caption="Your Design Result", 
                        use_container_width=True, channels="L")
            
            # Provide feedback
            if brightness_ok and is_sharpening:
                if design_cross == design_diag:
                    st.success("✅ **Great balanced kernel!** Good for general use.")
                elif abs(design_cross) > abs(design_diag):
                    st.success("✅ **Edge-focused kernel!** Good for text/architecture.")
                else:
                    st.success("✅ **Texture-aware kernel!** Good for portraits.")
            else:
                st.warning("⚠️ **Adjust your weights:** Center should be much larger than neighbors!")
        
        # Summary
        st.markdown("---")
        st.markdown("### 🎓 What You've Learned")
        
        st.success("""
        **🏆 Congratulations! You've mastered Direct Kernel Sharpening!**
        
        **Key Takeaways:**
        1. **Kernels = one-step sharpening** 🚀
        2. **Center weight controls strength** 🎚️
        3. **Neighbor weight controls edge emphasis** 🔍
        4. **Sum ≈ 1.0 preserves brightness** ⚖️
        5. **Different patterns = different effects** 🎨
        
        **You now understand:**
        - Why kernels are faster than unsharp masking ⏱️
        - How hardware implements this efficiently 🔧
        - How to design custom kernels 🎨
        - When to use different patterns 🤔
        
        **Next Steps:**
        - Try sharpening your own photos!
        - Experiment with other kernel operations (edge detection, blur)
        - Learn about separable kernels (even faster!)
        
        Remember: The best sharpening is invisible! 🎯
        """)