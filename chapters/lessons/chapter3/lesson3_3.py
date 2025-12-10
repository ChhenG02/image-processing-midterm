import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def app():
    st.title("✨ Understanding Unsharp Masking: The Magic of Image Sharpening")
    
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
        st.header("🤔 What is Unsharp Masking? (In Plain English)")
        
        st.markdown("""
        ### Think of a Photo That's "Almost Perfect" 📸
        
        It's a nice picture, but... **it's a little bit soft/blurry.** 😕
        
        What if you could:
        - Keep all the good parts ✅
        - Just **boost the edges** to make them clearer 🎯
        - Make the whole image look **sharper and crisper**! ✨
        
        That's exactly what **Unsharp Masking** does!
        
        ---
        
        ### The "Chef's Secret" Analogy 👨‍🍳
        
        Imagine you're making soup 🍲:
        1. **Start with your soup** = Original image
        2. **Make it super bland** = Blur the image (remove details)
        3. **Taste the difference** = Original - Blurred = "Flavor extract"
        4. **Add extra flavor extract** back in = Sharpen!
        
        **Result:** Same soup, but **more flavorful** (sharper)! 🎉
        """)
        
        st.markdown("---")
        
        # Visual metaphor
        st.markdown("### 🎨 Visual Metaphor:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Photo**")
            st.markdown("""
            ```
            📸 Normal photo
            👁️ Can see everything
            😊 Looks good
            🤏 But a bit soft
            ```
            """)
            st.caption("Like normal vision")
            
        with col2:
            st.markdown("**➡️ BLUR IT ➡️**")
            st.markdown("""
            Make it "out of focus":
            
            ```
            🌫️ Apply blur
            😵 Remove details
            📉 Lose sharpness
            🔍 Can't see edges
            ```
            """)
            st.caption("Like being nearsighted")
            
        with col3:
            st.markdown("**Sharpened Result**")
            st.markdown("""
            ```
            ✨ Add back details
            🔥 Boost edges
            🎯 Make crisp
            👁️ Clearer than original!
            ```
            """)
            st.caption("Like putting on glasses!")
        
        st.success("""
        ✨ **The Magic Formula:**
        
        **Sharpened = Original + (Original - Blurred) × Amount**
        
        - Take what was **lost in blurring**
        - **Amplify it** (multiply by amount)
        - **Add it back** to original
        
        **Result:** Enhanced edges without changing overall image! 🎯
        """)
        
        st.markdown("---")
        
        # Key concept
        st.info("""
        ### 🎯 Key Concept to Remember:
        
        **Unsharp Masking = Boosting What Was Lost**
        
        - **Step 1:** Create a "bad" (blurry) version  
        - **Step 2:** Find the difference (what was lost)
        - **Step 3:** Add that difference BACK, but stronger!
        
        **Why "Unsharp"?** Because we start by making it LESS sharp (blurry), 
        then enhance the difference to make it MORE sharp! 🔄
        """)
        
        # Simple example
        st.markdown("---")
        st.markdown("### 📱 A Simple Example:")
        
        example_original = np.array([50, 50, 100, 100], dtype=float)
        example_blurred = np.array([50, 67, 83, 100], dtype=float)  # Smoothed
        example_mask = example_original - example_blurred
        example_sharpened = example_original + example_mask * 1.0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Sharpening:**")
            st.code(f"""
Original: {example_original}
Blurred:  {example_blurred}
Difference: {example_mask}
Edge: 50 → 100 (contrast = 50)
            """)
            
        with col2:
            st.markdown("**After Sharpening:**")
            st.code(f"""
Sharpened: {example_sharpened}
New edge: {example_sharpened[1]:.0f} → {example_sharpened[2]:.0f}
New contrast = {abs(example_sharpened[2]-example_sharpened[1]):.0f}
Improvement: +{(abs(example_sharpened[2]-example_sharpened[1]) - 50)/50*100:.0f}%!
            """)
        
        st.success("""
        ✅ **See the difference?** The edge got stronger!  
        From 50→100 to 33→117 → Much more visible! 👁️
        """)
    
    # ==================== TAB 2: SEE IT IN ACTION ====================
    with tab2:
        st.header("👀 See Unsharp Masking in Action")
        
        st.markdown("""
        Let's walk through unsharp masking **step by step** on a real image!
        
        We'll pick a small area to see exactly what happens. 🔍
        """)
        
        # Pick a region to demonstrate
        h, w = img_np.shape
        demo_y, demo_x = h // 2 - 10, w // 2 - 10  # Center region
        
        st.markdown(f"### 🎯 Let's Process a 5×5 Region Starting at ({demo_x}, {demo_y})")
        
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
        
        # Create box blur kernel
        def create_box_kernel(size):
            kernel = np.ones((size, size), dtype=float)
            kernel /= (size * size)
            return kernel
        
        # Step 1: Show original region
        st.markdown("#### Step 1️⃣: Look at Original Pixels")
        
        region_size = 5
        region = img_np[demo_y:demo_y+region_size, demo_x:demo_x+region_size]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            region_large = np.repeat(np.repeat(region, 20, axis=0), 20, axis=1)
            st.image(region_large.astype(np.uint8), caption="5×5 Region", 
                    use_container_width=True, channels="L")
        
        with col2:
            st.markdown("**The 25 pixel brightness values:**")
            
            region_df = pd.DataFrame(region)
            st.dataframe(region_df.style.format("{:.0f}").background_gradient(cmap='Greys', vmin=0, vmax=255), 
                        use_container_width=True)
            
            st.caption("Look for edges where values change quickly!")
        
        # Step 2: Apply blur
        st.markdown("---")
        st.markdown("#### Step 2️⃣: Create Blurred Version (3×3 Box Blur)")
        
        blur_size = 3
        blur_kernel = create_box_kernel(blur_size)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Blur Kernel (3×3):**")
            st.code("""[0.111, 0.111, 0.111]
[0.111, 0.111, 0.111]
[0.111, 0.111, 0.111]""")
            st.caption("Averages 9 pixels")
        
        with col2:
            st.markdown("**Blur Calculation for Center Pixel:**")
            
            # Calculate blurred value for center pixel
            center_y, center_x = region_size//2, region_size//2
            padded_region = np.pad(region, 1, mode='edge')
            blur_window = padded_region[center_y:center_y+3, center_x:center_x+3]
            
            blurred_center = np.sum(blur_window * blur_kernel)
            
            st.code(f"""
3×3 window around center:
{blur_window[0,0]:5.0f} {blur_window[0,1]:5.0f} {blur_window[0,2]:5.0f}
{blur_window[1,0]:5.0f} {blur_window[1,1]:5.0f} {blur_window[1,2]:5.0f}
{blur_window[2,0]:5.0f} {blur_window[2,1]:5.0f} {blur_window[2,2]:5.0f}

Sum × (1/9) = {np.sum(blur_window):.0f} × 0.111
            """)
            
            st.info(f"**Blurred center pixel:** {blurred_center:.0f} (was {region[center_y, center_x]:.0f})")
        
        # Apply blur to entire region
        blurred_region = manual_convolution(region, blur_kernel)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Region:**")
            st.dataframe(pd.DataFrame(region).style.format("{:.0f}"), 
                        use_container_width=True)
        
        with col2:
            st.markdown("**Blurred Region:**")
            st.dataframe(pd.DataFrame(blurred_region).style.format("{:.0f}"), 
                        use_container_width=True)
            
            avg_change = np.mean(np.abs(region - blurred_region))
            st.caption(f"Average change per pixel: {avg_change:.1f}")
        
        # Step 3: Create mask
        st.markdown("---")
        st.markdown('#### Step 3️⃣: Create the "Unsharp Mask" (Difference)')

        
        mask_region = region - blurred_region
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**The Formula:**")
            st.code("""Mask = Original - Blurred
            
What this gives us:
- Positive values: Areas that were REDUCED by blur
- Negative values: Areas that were INCREASED by blur
- Zero: Areas that didn't change
            """)
            st.caption("The mask contains the LOST DETAILS!")
        
        with col2:
            st.markdown("**The Mask Values:**")
            st.dataframe(pd.DataFrame(mask_region).style.format("{:+.1f}").background_gradient(cmap='RdYlBu', vmin=-50, vmax=50), 
                        use_container_width=True)
            
            # Find strongest edge in mask
            max_mask = np.max(np.abs(mask_region))
            st.success(f"**Strongest detail found:** {max_mask:.1f}")
            
            st.markdown("""
            **Notice:** 
            - Biggest values = biggest differences
            - Where original and blurred differ most
            - These are the edges/details we'll enhance!
            """)
        
        # Step 4: Apply sharpening
        st.markdown("---")
        st.markdown("#### Step 4️⃣: Add Mask Back (with Boost!)")
        
        amount = 1.5  # Default amount
        sharpened_region = region + mask_region * amount
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sharpening Formula:**")
            st.code(f"""Sharpened = Original + Mask × Amount
                 = Original + (Original - Blurred) × {amount}
                 
Why multiply by amount?
- Amount = 1: Restore original edges
- Amount > 1: Enhance beyond original!
- Amount < 1: Mild enhancement
            """)
        
        with col2:
            st.markdown("**Sharpened Region:**")
            st.dataframe(pd.DataFrame(sharpened_region).style.format("{:.0f}"), 
                        use_container_width=True)
            
            # Compare edge enhancement
            # Find an edge in original region
            edge_orig = abs(region[2,2] - region[2,3])
            edge_sharp = abs(sharpened_region[2,2] - sharpened_region[2,3])
            
            if edge_orig > 0:
                improvement = (edge_sharp - edge_orig) / edge_orig * 100
                st.success(f"**Edge enhanced by {improvement:.0f}%!**")
        
        # Visual comparison
        st.markdown("---")
        st.markdown("### 🔍 Visual Comparison")
        
        # Create composite images
        col1, col2, col3 = st.columns(3)
        
        with col1:
            orig_display = np.clip(region, 0, 255).astype(np.uint8)
            st.image(orig_display, caption="Original", 
                    use_container_width=True, channels="L")
        
        with col2:
            mask_display = np.clip(mask_region - mask_region.min(), 0, 255).astype(np.uint8)
            st.image(mask_display, caption="Mask (Details)", 
                    use_container_width=True, channels="L")
        
        with col3:
            sharp_display = np.clip(sharpened_region, 0, 255).astype(np.uint8)
            st.image(sharp_display, caption=f"Sharpened (×{amount})", 
                    use_container_width=True, channels="L")
        
        st.success("""
        ✅ **Success!** We enhanced a small region.
        
        Now imagine doing this for **EVERY pixel** → **Whole image gets sharper!** 🔥
        """)
        
        st.info("""
        💡 **Key Insight:**
        - Blur removes details (makes edges softer)
        - Mask = what was removed
        - Adding back amplified mask = sharper edges
        - Control with: **Blur size** (what details to enhance) and **Amount** (how much)
        """)
    
    # ==================== TAB 3: CONTROL SHARPENING ====================
    with tab3:
        st.header("🎛️ Control Your Sharpening")
        
        st.markdown("""
        ### Two Knobs to Tweak: 🎚️🎛️
        
        1. **Blur Size** = How much to blur first
           - Small blur (3×3) → Enhance fine details (hair, texture) 🔬
           - Large blur (11×11) → Enhance coarse features (edges, shapes) 🏔️
        
        2. **Amount** = How much to boost
           - Low amount (0.5) → Subtle enhancement 😊
           - High amount (2.0) → Strong enhancement 🔥
           - Too high (>2.5) → Creates halos/artifacts ⚠️
        """)
        
        # Controls
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            blur_size = st.slider(
                "Blur Size (Kernel Dimensions)",
                min_value=3,
                max_value=15,
                value=5,
                step=2,
                help="Bigger = affect larger features, smaller = affect fine details"
            )
            
            # Show what blur size means
            st.markdown(f"""
            **What {blur_size}×{blur_size} blur does:**
            - Looks at {blur_size*blur_size} neighbors per pixel
            - Smooths over {blur_size} pixels wide
            - Affects features about {blur_size//2} pixels in size
            """)
        
        with col2:
            amount = st.slider(
                "Sharpening Amount (Boost Strength)",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="0 = no sharpening, 1 = restore lost details, >1 = enhance beyond original"
            )
            
            if amount == 0:
                st.info("😴 No sharpening - original image")
            elif amount < 0.5:
                st.info("😊 Very subtle enhancement")
            elif amount < 1.5:
                st.success("✅ Good balance - natural looking")
            elif amount < 2.5:
                st.warning("🔥 Strong enhancement - may create halos")
            else:
                st.error("⚡ Very strong - likely artifacts!")
        
        # Apply unsharp masking to whole image
        def apply_unsharp_masking(image, blur_size, amount):
            # Create blur kernel
            kernel = np.ones((blur_size, blur_size), dtype=float)
            kernel /= (blur_size * blur_size)
            
            # Apply blur
            padded = np.pad(image, blur_size//2, mode='reflect')
            blurred = np.zeros_like(image, dtype=float)
            
            h, w = image.shape
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+blur_size, j:j+blur_size]
                    blurred[i, j] = np.sum(region * kernel)
            
            # Create mask and sharpen
            mask = image - blurred
            sharpened = image + mask * amount
            sharpened = np.clip(sharpened, 0, 255)
            
            return sharpened.astype(np.uint8), blurred.astype(np.uint8), mask
        
        with st.spinner(f"Sharpening with {blur_size}×{blur_size} blur, amount={amount}..."):
            sharpened, blurred, mask = apply_unsharp_masking(img_np, blur_size, amount)
            
            # Normalize mask for display
            mask_display = mask - mask.min()
            mask_display = (mask_display / (mask_display.max() + 1e-6) * 255).astype(np.uint8)
        
        # Show results
        st.markdown("---")
        st.markdown(f"### 📊 Sharpening Results: Blur={blur_size}×{blur_size}, Amount={amount}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image(img_np.astype(np.uint8), caption="🖼️ Original", 
                    use_container_width=True, channels="L")
            st.metric("Mean", f"{img_np.mean():.1f}")
        
        with col2:
            st.image(blurred, caption=f"🌫️ Blurred ({blur_size}×{blur_size})", 
                    use_container_width=True, channels="L")
            st.metric("Mean", f"{blurred.mean():.1f}")
        
        with col3:
            st.image(mask_display, caption="🎭 Unsharp Mask", 
                    use_container_width=True, channels="L")
            st.metric("Range", f"{mask.min():.1f} to {mask.max():.1f}")
        
        with col4:
            st.image(sharpened, caption=f"✨ Sharpened (×{amount})", 
                    use_container_width=True, channels="L")
            st.metric("Mean", f"{sharpened.mean():.1f}")
        
        # Side-by-side comparison
        st.markdown("---")
        st.markdown("### 🔍 Direct Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            composite_orig_blur = np.concatenate([img_np.astype(np.uint8), blurred], axis=1)
            st.image(composite_orig_blur, caption="Left: Original | Right: Blurred", 
                    use_container_width=True, channels="L")
        
        with col2:
            composite_orig_sharp = np.concatenate([img_np.astype(np.uint8), sharpened], axis=1)
            st.image(composite_orig_sharp, caption="Left: Original | Right: Sharpened", 
                    use_container_width=True, channels="L")
        
        # Difference analysis
        st.markdown("---")
        st.markdown("### 📈 Difference Analysis")
        
        difference = sharpened.astype(float) - img_np.astype(float)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enhanced = np.sum(difference > 0)
            st.metric("Enhanced Pixels", f"{enhanced:,}", 
                     f"{enhanced/(h*w)*100:.1f}%")
        
        with col2:
            reduced = np.sum(difference < 0)
            st.metric("Reduced Pixels", f"{reduced:,}", 
                     f"{reduced/(h*w)*100:.1f}%")
        
        with col3:
            avg_change = np.mean(np.abs(difference))
            st.metric("Avg Change", f"{avg_change:.1f}")
        
        # Parameter gallery
        st.markdown("---")
        st.markdown("### 🎨 Sharpening Gallery (Try These Presets)")
        
        presets = [
            {"name": "Subtle", "blur": 3, "amount": 0.5},
            {"name": "Standard", "blur": 5, "amount": 1.0},
            {"name": "Strong", "blur": 7, "amount": 1.5},
            {"name": "Fine Details", "blur": 3, "amount": 1.5},
            {"name": "Coarse Features", "blur": 11, "amount": 1.0}
        ]
        
        cols = st.columns(len(presets))
        
        for idx, preset in enumerate(presets):
            with cols[idx]:
                if st.button(preset["name"], key=f"preset_{idx}"):
                    st.session_state.blur_size = preset["blur"]
                    st.session_state.amount = preset["amount"]
                    st.rerun()
                
                # Create preview (simplified calculation)
                preview_kernel = np.ones((preset["blur"], preset["blur"]), dtype=float)
                preview_kernel /= (preset["blur"] * preset["blur"])
                
                # Small preview region
                preview_region = img_np[h//4:h//4+50, w//4:w//4+50]
                preview_padded = np.pad(preview_region, preset["blur"]//2, mode='reflect')
                preview_blurred = np.zeros_like(preview_region, dtype=float)
                
                ph, pw = preview_region.shape
                for i in range(ph):
                    for j in range(pw):
                        region = preview_padded[i:i+preset["blur"], j:j+preset["blur"]]
                        preview_blurred[i, j] = np.sum(region * preview_kernel)
                
                preview_mask = preview_region - preview_blurred
                preview_sharp = preview_region + preview_mask * preset["amount"]
                preview_sharp = np.clip(preview_sharp, 0, 255).astype(np.uint8)
                
                st.image(preview_sharp, caption=preset["name"], 
                        use_container_width=True, channels="L")
                st.caption(f"Blur={preset['blur']}, Amount={preset['amount']}")
    
    # ==================== TAB 4: WHY IT MATTERS ====================
    with tab4:
        st.header("🌍 Why Do We Need Image Sharpening?")
        
        st.markdown("""
        ### "Why not take sharp photos in the first place?" 🤔
        
        Sometimes we CAN'T control the original quality! Many images need help:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            ### ✅ When Sharpening Saves the Day:
            
            **1. 📸 Old Photos & Scans**
            - Grandpa's faded photo album
            - Historical document scans
            - Film negatives being digitized
            - **Unsharp masking:** Brings back lost details!
            
            **2. 🏥 Medical Images**
            - Blurry MRI/CT scans
            - Microscopic cell images
            - Ultrasound with noise
            - **Unsharp masking:** Makes diagnosis easier!
            
            **3. 🛰️ Satellite/Aerial Photos**
            - Atmospheric haze blurs images
            - Long-distance photography
            - Surveillance footage
            - **Unsharp masking:** Reveals hidden details!
            
            **4. 📱 Smartphone Photos**
            - Tiny camera sensors
            - Digital zoom (makes blurry)
            - Low light = camera shake
            - **Unsharp masking:** Makes photos pop!
            """)
        
        with col2:
            st.info("""
            ### 🎯 The Science Behind It:
            
            **Our Eyes LOVE Sharp Edges!** 👁️
            
            Research shows:
            - Humans perceive sharper images as "higher quality"
            - Our brains process edges first when recognizing objects
            - Sharpness = clarity = trustworthiness
            
            **Example:** Compare these two text samples:
            
            ```
            Blurry: Th1s 1s h4rd to r3ad
            Sharp:  This is easy to read
            ```
            
            Same letters, but sharpness makes ALL the difference!
            
            ---
            
            ### 💡 Fun Fact:
            
            **Instagram/Photoshop "Clarity" Slider = Unsharp Masking!**
            
            Next time you edit a photo:
            1. Slide "Clarity" to the right
            2. Watch edges get sharper
            3. That's unsharp masking working! ✨
            
            **Pro photographers use it ALL THE TIME!** 📷
            
            ---
            
            ### ⚠️ The Dark Side (When NOT to Use):
            
            **Too much sharpening creates:**
            - Halos (light/dark lines around edges) 😇
            - Noise amplification (grain gets worse) 🌪️
            - Artificial/"plastic" look 🎭
            
            **Rule of thumb:** Enhance, don't exaggerate!
            """)
        
        # Visual demonstration
        st.markdown("---")
        st.markdown("### 🎨 See the Impact: Text Readability")
        
        # Create sample text image
        text_img = np.zeros((100, 400), dtype=np.uint8) + 200
        
        # Add blurry text
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_img_bgr = np.stack([text_img, text_img, text_img], axis=2)
        cv2.putText(text_img_bgr, "IMPORTANT DOCUMENT", (50, 60), font, 1, (0, 0, 0), 2)
        text_img_bw = text_img_bgr[:,:,0]
        
        # Apply blur
        kernel = np.ones((5,5), dtype=float) / 25
        padded = np.pad(text_img_bw, 2, mode='constant', constant_values=200)
        blurred_text = np.zeros_like(text_img_bw, dtype=float)
        
        for i in range(100):
            for j in range(400):
                region = padded[i:i+5, j:j+5]
                blurred_text[i, j] = np.sum(region * kernel)
        
        blurred_text = blurred_text.astype(np.uint8)
        
        # Apply unsharp masking
        mask_text = text_img_bw.astype(float) - blurred_text.astype(float)
        sharpened_text = text_img_bw.astype(float) + mask_text * 1.5
        sharpened_text = np.clip(sharpened_text, 0, 255).astype(np.uint8)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(text_img_bw, caption="Original Text", use_container_width=True, channels="L")
            st.caption("Crisp and clear")
        
        with col2:
            st.image(blurred_text, caption="Blurry Version", use_container_width=True, channels="L")
            st.caption("Hard to read 😕")
        
        with col3:
            st.image(sharpened_text, caption="After Sharpening", use_container_width=True, channels="L")
            st.caption("Readable again! ✅")
        
        st.success("""
        🧠 **Powerful Insight:** Sharpening doesn't add new information - 
        it just makes existing information EASIER TO SEE!
        
        This is why it's so valuable in medicine, forensics, and any field 
        where seeing details clearly matters! 🔍
        """)
        
        # Real-world examples
        st.markdown("---")
        st.markdown("### 🏥 Real Impact Stories")
        
        with st.expander("📖 Click to read real examples"):
            st.markdown("""
            **1. Medical Breakthrough:**
            - Researchers studying Alzheimer's
            - MRI scans too blurry to see plaque buildup
            - Applied unsharp masking
            - Could suddenly see early signs!
            - **Result:** Earlier diagnosis possible
            
            **2. Historical Discovery:**
            - Ancient manuscript found, faded by time
            - Text unreadable to naked eye
            - Scanned and sharpened
            - Revealed lost historical records!
            - **Result:** New chapter in history books
            
            **3. Security Success:**
            - Security camera footage of license plate
            - Too blurry to read numbers
            - Applied targeted sharpening
            - Plate became readable!
            - **Result:** Crime solved
            
            **4. Space Exploration:**
            - Mars rover sends back photos
            - Atmospheric dust makes images hazy
            - Apply unsharp masking on Earth
            - Geological features become clear!
            - **Result:** Better scientific analysis
            """)
    
    # ==================== TAB 5: ADVANCED PLAY ====================
    with tab5:
        st.header("🧪 Advanced: Experiment with Unsharp Masking!")
        
        st.markdown("""
        Ready to become a sharpening expert? Let's experiment! 🔬
        
        We'll explore advanced concepts and test your understanding.
        """)
        
        # Different blur types
        st.markdown("---")
        st.markdown("### 🔧 Try Different Blur Methods")
        
        blur_type = st.radio(
            "Choose blur method (instead of Box Blur):",
            ["Box Blur (Standard)", "Gaussian Blur (Smoother)", "Custom Weights"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_blur_size = st.slider("Test Blur Size", 3, 11, 5, step=2, key="adv_blur")
        
        with col2:
            test_amount = st.slider("Test Amount", 0.0, 3.0, 1.0, step=0.1, key="adv_amount")
        
        # Create different kernels
        def create_kernel(blur_type, size):
            if blur_type == "Box Blur (Standard)":
                kernel = np.ones((size, size), dtype=float)
                kernel /= (size * size)
                
            elif blur_type == "Gaussian Blur (Smoother)":
                kernel = np.zeros((size, size), dtype=float)
                center = size // 2
                sigma = size / 3
                
                for i in range(size):
                    for j in range(size):
                        x, y = i - center, j - center
                        kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                
                kernel /= np.sum(kernel)
                
            else:  # Custom Weights
                kernel = np.array([
                    [1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]
                ], dtype=float) / 16
                # Resize if needed (simplified)
                if size != 3:
                    st.info("Custom weights only available for 3×3")
                    kernel = np.ones((size, size), dtype=float) / (size * size)
            
            return kernel
        
        if st.button("🔬 Test This Configuration", type="primary"):
            kernel = create_kernel(blur_type, test_blur_size)
            
            # Apply to image
            padded = np.pad(img_np, test_blur_size//2, mode='reflect')
            test_blurred = np.zeros_like(img_np, dtype=float)
            
            h, w = img_np.shape
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+test_blur_size, j:j+test_blur_size]
                    test_blurred[i, j] = np.sum(region * kernel)
            
            test_mask = img_np - test_blurred
            test_sharpened = img_np + test_mask * test_amount
            test_sharpened = np.clip(test_sharpened, 0, 255).astype(np.uint8)
            
            # Show kernel
            st.markdown("---")
            st.markdown("### 🔢 The Kernel Being Used:")
            
            kernel_df = pd.DataFrame(kernel)
            st.dataframe(kernel_df.style.format("{:.4f}").background_gradient(cmap='YlOrRd'), 
                        use_container_width=True)
            
            st.caption(f"**Sum of weights:** {np.sum(kernel):.4f} (should be 1.0)")
            
            # Show results
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_np.astype(np.uint8), caption="Original", 
                        use_container_width=True, channels="L")
            
            with col2:
                st.image(test_sharpened, caption=f"Sharpened ({blur_type})", 
                        use_container_width=True, channels="L")
            
            # Comparison with standard
            st.markdown("---")
            st.markdown("### 📊 Comparison with Standard Box Blur")
            
            # Apply standard box blur for comparison
            std_kernel = np.ones((test_blur_size, test_blur_size), dtype=float)
            std_kernel /= (test_blur_size * test_blur_size)
            
            std_blurred = np.zeros_like(img_np, dtype=float)
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+test_blur_size, j:j+test_blur_size]
                    std_blurred[i, j] = np.sum(region * std_kernel)
            
            std_mask = img_np - std_blurred
            std_sharpened = img_np + std_mask * test_amount
            std_sharpened = np.clip(std_sharpened, 0, 255).astype(np.uint8)
            
            # Calculate difference
            diff = test_sharpened.astype(float) - std_sharpened.astype(float)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Difference", f"{np.mean(diff):.2f}")
            
            with col2:
                st.metric("Max Difference", f"{np.max(np.abs(diff)):.2f}")
            
            with col3:
                diff_pixels = np.sum(np.abs(diff) > 1)
                st.metric("Different Pixels", f"{diff_pixels:,}")
            
            st.info(f"""
            **Observation:** {blur_type} gives {'smoother' if 'Gaussian' in blur_type else 'different'} results!
            
            - **Box Blur:** Simple, fast, can create artifacts
            - **Gaussian Blur:** Natural, smooth, less artifacts
            - **Custom Weights:** Can tune for specific applications
            """)
        
        # Edge case experiments
        st.markdown("---")
        st.markdown("### 🎯 Edge Case Experiments")
        
        experiment = st.selectbox(
            "Choose an experiment:",
            ["No Experiment", "What if Amount = 0?", "What if Blur Size = 1?", 
             "Negative Amount?", "Extreme Blur + High Amount"]
        )
        
        if experiment != "No Experiment":
            if experiment == "What if Amount = 0?":
                result = img_np.astype(np.uint8)
                explanation = """
                **Amount = 0 means:** Sharpened = Original + Mask × 0 = Original
                
                **Result:** No change! The mask is multiplied by zero, so nothing gets added back.
                """
                
            elif experiment == "What if Blur Size = 1?":
                # Blur size 1 means kernel is just [1] - no blur!
                result = img_np.astype(np.uint8)
                explanation = """
                **Blur Size = 1 means:** Each pixel is "blurred" with just itself!
                
                **Calculation:**
                - Blurred = Original (since 1×1 average is itself)
                - Mask = Original - Original = 0 everywhere
                - Sharpened = Original + 0 × Amount = Original
                
                **Result:** No sharpening! Need at least 3×3 blur to create differences.
                """
                
            elif experiment == "Negative Amount?":
                # Apply with negative amount
                kernel = np.ones((5,5), dtype=float) / 25
                padded = np.pad(img_np, 2, mode='reflect')
                blurred = np.zeros_like(img_np, dtype=float)
                
                for i in range(h):
                    for j in range(w):
                        region = padded[i:i+5, j:j+5]
                        blurred[i, j] = np.sum(region * kernel)
                
                mask = img_np - blurred
                result = img_np + mask * (-1.0)  # Negative amount!
                result = np.clip(result, 0, 255).astype(np.uint8)
                
                explanation = """
                **Negative Amount =  -1.0 means:** We SUBTRACT the mask instead of adding it!
                
                **What happens:**
                - Original edges get REDUCED instead of enhanced
                - Image becomes BLURRIER than the blurred version!
                - We're removing even more detail
                
                **Result:** Image gets extra blurry! This is sometimes used for artistic effects.
                """
                
            else:  # Extreme Blur + High Amount
                kernel = np.ones((15,15), dtype=float) / 225
                padded = np.pad(img_np, 7, mode='reflect')
                blurred = np.zeros_like(img_np, dtype=float)
                
                for i in range(h):
                    for j in range(w):
                        region = padded[i:i+15, j:j+15]
                        blurred[i, j] = np.sum(region * kernel)
                
                mask = img_np - blurred
                result = img_np + mask * 3.0  # High amount!
                result = np.clip(result, 0, 255).astype(np.uint8)
                
                explanation = """
                **Extreme Parameters:** 15×15 blur + Amount = 3.0
                
                **What happens:**
                - Large blur removes ALL fine details
                - Mask contains HUGE differences
                - Multiplying by 3 amplifies them massively
                - Creates strong "halos" around edges
                
                **Result:** Very artificial look with visible artifacts!
                This shows why we need to use reasonable parameters.
                """
            
            # Show experiment
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_np.astype(np.uint8), caption="Original", 
                        use_container_width=True, channels="L")
            
            with col2:
                st.image(result, caption=experiment, 
                        use_container_width=True, channels="L")
            
            st.info(explanation)
        
        # Interactive challenge
        st.markdown("---")
        st.markdown("### 🧠 Sharpening Challenge")
        
        if st.checkbox("Take the sharpening challenge!"):
            st.markdown("""
            **Challenge:** Can you achieve these results?
            
            1. **Mild enhancement** - Just enough to notice, but still natural
            2. **Strong enhancement** - Clearly sharper, but no halos
            3. **Over-enhanced** - Visible artifacts (halos around edges)
            
            **Rules:**
            - Use only Blur Size and Amount controls
            - Try to match the descriptions
            - Check your work with the visual guide below
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                challenge_blur = st.slider("Challenge Blur Size", 3, 15, 5, step=2, 
                                          key="challenge_blur")
            
            with col2:
                challenge_amount = st.slider("Challenge Amount", 0.0, 3.0, 1.0, step=0.1,
                                           key="challenge_amount")
            
            # Apply challenge settings
            challenge_kernel = np.ones((challenge_blur, challenge_blur), dtype=float)
            challenge_kernel /= (challenge_blur * challenge_blur)
            
            padded = np.pad(img_np, challenge_blur//2, mode='reflect')
            challenge_blurred = np.zeros_like(img_np, dtype=float)
            
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+challenge_blur, j:j+challenge_blur]
                    challenge_blurred[i, j] = np.sum(region * challenge_kernel)
            
            challenge_mask = img_np - challenge_blurred
            challenge_result = img_np + challenge_mask * challenge_amount
            challenge_result = np.clip(challenge_result, 0, 255).astype(np.uint8)
            
            # Show result
            st.image(challenge_result, 
                    caption=f"Your Result: Blur={challenge_blur}, Amount={challenge_amount}", 
                    use_container_width=True, channels="L")
            
            # Evaluate
            if challenge_amount < 0.8:
                st.success("✅ **Mild Enhancement Achieved!**")
                st.caption("Subtle but effective - good for professional photos")
            elif challenge_amount < 1.8:
                st.warning("⚠️ **Strong Enhancement**")
                st.caption("Clearly sharper - might be good for web/social media")
            else:
                st.error("❌ **Over-enhanced!**")
                st.caption("Visible artifacts likely - reduce amount or increase blur size")
        
        # Summary
        st.markdown("---")
        st.markdown("### 🎓 What You've Learned")
        
        st.success("""
        **🏆 Congratulations! You've mastered Unsharp Masking!**
        
        **Key Takeaways:**
        1. **Unsharp masking enhances edges** by amplifying differences 📈
        2. **Blur size controls** what level of detail gets enhanced 🔍
        3. **Amount controls** how much enhancement is applied 🎚️
        4. **Formula:** Sharpened = Original + (Original - Blurred) × Amount
        5. **Balance is key** - too much creates artifacts ⚠️
        
        **You now understand:**
        - Why photographers use this technique 📷
        - How medical imaging benefits from it 🏥
        - When to use different blur sizes 🤔
        - How to avoid common mistakes 🚫
        
        **Next Steps:**
        - Try sharpening your own photos!
        - Compare with other sharpening methods
        - Experiment with "High Pass Filter" (related technique)
        
        Remember: The best enhancement is the one you don't notice! 🎯
        """)