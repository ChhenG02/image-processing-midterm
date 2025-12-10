import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def app():
    st.title("🎨 Transparency & Image Blending")
    
    # Simple analogy first
    st.markdown("""
    ### 🤔 The Core Idea (Simple Analogy)
    
    Imagine you have two transparent sheets of colored plastic:
    - **Red sheet** (Image 1) and **Blue sheet** (Image 2)
    - When you **overlap them**, you see a **purple mix**!
    - **Adjust transparency** of the top sheet:
      - 100% opaque red sheet = you only see red (α = 1.0)
      - 0% opaque red sheet = you only see blue underneath (α = 0.0)
      - 50% transparent = perfect purple mix (α = 0.5)
    
    **Alpha blending** does exactly this with images - it's like controlling the transparency of overlapping layers!
    """)
    
    st.markdown("---")
    
    # Load images
    try:
        img1 = Image.open("public/blender.jpg")
        img2 = img1.transpose(Image.FLIP_LEFT_RIGHT)
    except:
        img1 = Image.new('RGB', (300, 300), color='red')
        img2 = Image.new('RGB', (300, 300), color='blue')
    
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    
    # Ensure same size
    if img1_np.shape != img2_np.shape:
        img2 = Image.fromarray(img2_np).resize((img1_np.shape[1], img1_np.shape[0]))
        img2_np = np.array(img2)
    
    # Main control
    st.markdown("### 🎚️ The Magic Slider: Control the Mix!")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        alpha = st.slider(
            "Alpha (α) - Transparency Control", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="Move the slider to blend between the two images!"
        )
    
    with col2:
        st.metric("Current Mix", f"{alpha:.0%} / {(1-alpha):.0%}")
    
    # Visual explanation of current alpha
    if alpha == 0.0:
        st.info("🔵 **α = 0.0** → Showing 100% Image 2 (Image 1 is completely transparent)")
    elif alpha == 1.0:
        st.info("🔴 **α = 1.0** → Showing 100% Image 1 (Image 2 is completely transparent)")
    elif 0.4 <= alpha <= 0.6:
        st.info("🟣 **α ≈ 0.5** → Perfect 50/50 blend (both images equally visible)")
    elif alpha > 0.5:
        st.info(f"🔴 **α = {alpha:.2f}** → More Image 1 visible ({alpha:.0%} vs {(1-alpha):.0%})")
    else:
        st.info(f"🔵 **α = {alpha:.2f}** → More Image 2 visible ({(1-alpha):.0%} vs {alpha:.0%})")
    
    # The math formula with actual numbers
    st.code(f"Result = {alpha:.2f} × Image₁ + {1-alpha:.2f} × Image₂", language=None)
    
    # Perform blending
    blended = (alpha * img1_np + (1 - alpha) * img2_np).astype(np.uint8)
    
    # Display images
    st.markdown("---")
    st.markdown("### 📸 Live Blending Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img1_np, caption="🅰️ Image 1 (Red Layer)", use_container_width=True)
        st.progress(alpha, text=f"Contribution: {alpha:.0%}")
        if alpha > 0.5:
            st.success(f"**Dominant** ({alpha:.0%})")
        elif alpha == 0.5:
            st.warning("**Equal Mix** (50%)")
        else:
            st.error(f"Fading ({alpha:.0%})")
    
    with col2:
        st.image(blended, caption="✨ Blended Result", use_container_width=True)
        st.markdown(f"""
        **Mix Ratio:**  
        🅰️ Image 1: **{alpha:.1%}**  
        🅱️ Image 2: **{(1-alpha):.1%}**
        """)
        # Show color indicator
        if alpha > 0.7:
            st.markdown("🔴 Mostly Image 1")
        elif alpha < 0.3:
            st.markdown("🔵 Mostly Image 2")
        else:
            st.markdown("🟣 Balanced Mix")
    
    with col3:
        st.image(img2_np, caption="🅱️ Image 2 (Blue Layer)", use_container_width=True)
        st.progress(1 - alpha, text=f"Contribution: {(1-alpha):.0%}")
        if alpha < 0.5:
            st.success(f"**Dominant** ({(1-alpha):.0%})")
        elif alpha == 0.5:
            st.warning("**Equal Mix** (50%)")
        else:
            st.error(f"Fading ({(1-alpha):.0%})")
    
    # Quick presets
    st.markdown("---")
    st.markdown("### ⚡ Quick Preset Blends")
    
    presets = {
        "🔴 Only Image 1": 1.0,
        "🔴 Mostly Image 1": 0.75,
        "🟣 Perfect Mix": 0.5,
        "🔵 Mostly Image 2": 0.25,
        "🔵 Only Image 2": 0.0
    }
    
    cols = st.columns(len(presets))
    for idx, (name, value) in enumerate(presets.items()):
        with cols[idx]:
            preset_blend = (value * img1_np + (1 - value) * img2_np).astype(np.uint8)
            st.image(preset_blend, use_container_width=True)
            st.caption(f"{name}\nα = {value}")
    
    # Pixel-level example
    st.markdown("---")
    st.markdown("### 🔬 Pixel-Level Math Example")
    
    st.markdown("Let's see how blending works on individual pixels:")
    
    # Pick sample pixels
    h, w = img1_np.shape[:2]
    sample_points = [
        (h//4, w//4, "Top-Left"),
        (h//2, w//2, "Center"),
        (3*h//4, 3*w//4, "Bottom-Right")
    ]
    
    pixel_data = []
    for y, x, location in sample_points:
        # Get RGB values
        rgb1 = img1_np[y, x]
        rgb2 = img2_np[y, x]
        rgb_blend = blended[y, x]
        
        # Calculate what it should be
        expected = (alpha * rgb1 + (1 - alpha) * rgb2).astype(int)
        
        if len(rgb1.shape) == 0:  # Grayscale
            pixel_data.append({
                "Location": location,
                "Image 1": f"{rgb1}",
                "Image 2": f"{rgb2}",
                "Formula": f"{alpha:.2f}×{rgb1} + {1-alpha:.2f}×{rgb2}",
                "Result": f"{rgb_blend}",
                "Expected": f"{expected}"
            })
        else:  # RGB
            pixel_data.append({
                "Location": location,
                "Image 1 (R,G,B)": f"({rgb1[0]},{rgb1[1]},{rgb1[2]})",
                "Image 2 (R,G,B)": f"({rgb2[0]},{rgb2[1]},{rgb2[2]})",
                "Result (R,G,B)": f"({rgb_blend[0]},{rgb_blend[1]},{rgb_blend[2]})",
                "Calculation": f"α={alpha:.2f}"
            })
    
    df = pd.DataFrame(pixel_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.caption("💡 Each color channel (R, G, B) is blended independently using the same alpha value!")
    
    # Animation/Transition showcase
    st.markdown("---")
    st.markdown("### 🎬 Smooth Transition Animation")
    
    if st.button("▶️ Play Transition (Image 1 → Image 2)", use_container_width=True):
        st.markdown("**Watch as Image 1 gradually fades into Image 2:**")
        
        progress_placeholder = st.empty()
        image_placeholder = st.empty()
        info_placeholder = st.empty()
        
        alphas = np.linspace(1.0, 0.0, 21)  # 21 frames from 1.0 to 0.0
        
        for i, a in enumerate(alphas):
            frame = (a * img1_np + (1 - a) * img2_np).astype(np.uint8)
            
            progress_placeholder.progress(i / (len(alphas) - 1), text=f"Frame {i+1}/21")
            image_placeholder.image(frame, use_container_width=True, caption=f"α = {a:.2f}")
            info_placeholder.info(f"Current Mix: {a:.1%} Image 1 + {(1-a):.1%} Image 2")
            
            # Add small delay for animation effect
            import time
            time.sleep(0.1)
        
        st.success("✅ Animation complete! Notice the smooth transition?")
    
    # Comparison grid
    st.markdown("---")
    st.markdown("### 📊 Blending Comparison Grid")
    st.markdown("See how different alpha values create different results:")
    
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    cols = st.columns(len(alpha_values))
    
    for idx, a in enumerate(alpha_values):
        with cols[idx]:
            blend_sample = (a * img1_np + (1 - a) * img2_np).astype(np.uint8)
            st.image(blend_sample, use_container_width=True)
            st.markdown(f"**α = {a}**")
            st.caption(f"{a:.0%} A + {(1-a):.0%} B")
    
    # Real-world applications
    st.markdown("---")
    st.markdown("### 🌍 Real-World Applications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🎬 Video & Photography:**
        - **Dissolve transitions** in movies (fade from scene A to B)
        - **Double exposure** artistic effects
        - **Cross-fading** in slideshows
        - **HDR photography** (blend different exposures)
        - **Watermarks** (blend logo with α = 0.3 for transparency)
        """)
        
        st.info("""
        **🏥 Medical Imaging:**
        - **Overlay MRI + CT scans** (see bones AND soft tissue)
        - **Compare before/after** treatment scans
        - **Surgical planning** (blend 3D models with X-rays)
        - **Highlight changes** over time
        """)
    
    with col2:
        st.warning("""
        **🎮 Gaming & AR:**
        - **HUD overlays** (health bars with α = 0.7)
        - **Augmented reality** (virtual objects on real video)
        - **Fog/smoke effects** (blend semi-transparent layers)
        - **Ghost/invisibility** effects (character with α = 0.3)
        """)
        
        st.error("""
        **🎨 Graphic Design:**
        - **Layer compositing** in Photoshop/GIMP
        - **Text over images** (with semi-transparent backgrounds)
        - **Creating shadows** and reflections
        - **Color grading** and filters
        - **Logo placement** on photos
        """)
    
    # Interactive challenge
    st.markdown("---")
    st.markdown("### 🎮 Try These Experiments!")
    
    challenges = [
        "**Ghosting Effect**: Set α = 0.3 to make Image 1 look like a ghost over Image 2!",
        "**Perfect Balance**: Find the α value where both images are equally visible (hint: α = 0.5)",
        "**Quick Transition**: Rapidly move the slider from 0 to 1 - see the smooth transition?",
        "**Watermark Simulation**: Set α = 0.7 or 0.8 to simulate a semi-transparent watermark",
        "**Fade Animation**: Click the animation button and watch a cinematic fade effect!"
    ]
    
    for i, challenge in enumerate(challenges, 1):
        st.markdown(f"{i}. {challenge}")
    
    # Mathematical explanation
    with st.expander("🧮 The Math Behind Alpha Blending"):
        st.markdown("""
        ### Formula Breakdown:
        
        **Result = α × Image₁ + (1 - α) × Image₂**
        
        Where:
        - **α** (alpha) = weighting factor between 0 and 1
        - **α = 1.0** means 100% Image₁, 0% Image₂
        - **α = 0.0** means 0% Image₁, 100% Image₂
        - **α = 0.5** means 50% Image₁, 50% Image₂
        
        ### Why (1 - α)?
        The weights must **add up to 1** (100% total):
        - α + (1 - α) = 1 ✅
        - If α = 0.7, then (1 - α) = 0.3
        - Total: 0.7 + 0.3 = 1.0 (perfect!)
        
        ### Pixel-by-Pixel Example:
        ```
        Image₁ pixel = 200
        Image₂ pixel = 100
        α = 0.6
        
        Result = 0.6 × 200 + 0.4 × 100
               = 120 + 40
               = 160
        ```
        
        The result (160) is between 100 and 200, closer to 200 because α = 0.6!
        """)
    
    # Code example
    with st.expander("💻 Python Code Example"):
        st.code("""
import numpy as np
from PIL import Image

# Load images
img1 = np.array(Image.open('image1.jpg'))
img2 = np.array(Image.open('image2.jpg'))

# Set alpha (0.0 to 1.0)
alpha = 0.5

# Blend images
blended = (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)

# Save result
Image.fromarray(blended).save('blended.jpg')

# Pro tip: For smooth transitions
for alpha in np.linspace(0, 1, 30):  # 30 frames
    frame = (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)
    # Save each frame or display
        """, language="python")
    
    st.markdown("---")
    st.caption("💡 Fun Fact: Alpha blending is named after the Greek letter α (alpha), which mathematically represents the opacity/transparency value!")