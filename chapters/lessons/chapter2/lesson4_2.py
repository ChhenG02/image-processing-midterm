import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def app():
    st.title("🎭 Masking & Region Isolation")
    
    # Simple analogy first
    st.markdown("""
    ### 🤔 The Core Idea (Simple Analogy)
    
    Imagine you're **spray painting** with a **stencil**:
    - Put a stencil over paper (the stencil is your **mask**)
    - Spray paint over it
    - **Only the cutout areas get painted!**
    - Remove stencil → you've isolated specific shapes
    
    **Image masking works exactly the same way:**
    - Mask = stencil (0 = blocked, 1 = open)
    - Image × Mask = only selected areas show through
    - Black areas (0) hide the image, White areas (1) reveal it
    """)
    
    # Visual diagram
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🖼️ Original Image")
        st.markdown("Full picture visible")
    with col2:
        st.markdown("#### 🎭 Mask (Stencil)")
        st.markdown("White = show, Black = hide")
    with col3:
        st.markdown("#### ✨ Result")
        st.markdown("Image × Mask")
    
    st.markdown("---")
    
    # Load image
    img = Image.open("public/ch1.1.jpg").convert('L')
    img_np = np.array(img).astype(float)
    height, width = img_np.shape
    
    # Controls
    st.markdown("### 🎛️ Design Your Mask")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        mask_type = st.selectbox(
            "🎨 Choose Mask Shape:",
            ["Circle", "Rectangle", "Horizontal Stripes", "Vertical Stripes", 
             "Checkerboard", "Gradient (Left→Right)", "Gradient (Top→Bottom)", "Ring (Donut)"],
            help="Different shapes for different purposes!"
        )
    
    with col2:
        invert = st.checkbox("🔄 Invert Mask", help="Flip black and white areas")
    
    # Shape-specific controls
    st.markdown("#### 🔧 Adjust Mask Parameters")
    
    if mask_type == "Rectangle":
        col1, col2 = st.columns(2)
        with col1:
            x = st.slider("📍 X Position", 0, width-50, width//4, help="Horizontal position")
            w = st.slider("📏 Width", 50, width-x, width//2)
        with col2:
            y = st.slider("📍 Y Position", 0, height-50, height//4, help="Vertical position")
            h = st.slider("📏 Height", 50, height-y, height//2)
            
    elif mask_type == "Circle":
        col1, col2 = st.columns(2)
        with col1:
            cx = st.slider("📍 Center X", 50, width-50, width//2)
            cy = st.slider("📍 Center Y", 50, height-50, height//2)
        with col2:
            radius = st.slider("📏 Radius", 20, min(width, height)//2, min(width, height)//4)
            
    elif mask_type == "Ring (Donut)":
        col1, col2 = st.columns(2)
        with col1:
            cx = st.slider("📍 Center X", 50, width-50, width//2)
            cy = st.slider("📍 Center Y", 50, height-50, height//2)
        with col2:
            outer_radius = st.slider("📏 Outer Radius", 50, min(width, height)//2, min(width, height)//3)
            inner_radius = st.slider("📏 Inner Radius", 10, outer_radius-10, outer_radius//2)
            
    elif mask_type in ["Horizontal Stripes", "Vertical Stripes"]:
        col1, col2 = st.columns(2)
        with col1:
            num_stripes = st.slider("📊 Number of Stripes", 2, 20, 5)
        with col2:
            thickness = st.slider("📏 Stripe Thickness (%)", 10, 90, 50)
    
    elif mask_type == "Checkerboard":
        col1, col2 = st.columns(2)
        with col1:
            num_blocks = st.slider("📊 Blocks per Side", 4, 20, 8)
    
    # Create mask
    mask = np.zeros((height, width), dtype=float)
    
    if mask_type == "Rectangle":
        mask[y:y+h, x:x+w] = 1.0
        
    elif mask_type == "Circle":
        y_coords, x_coords = np.ogrid[:height, :width]
        circle_mask = (x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2
        mask[circle_mask] = 1.0
        
    elif mask_type == "Ring (Donut)":
        y_coords, x_coords = np.ogrid[:height, :width]
        distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        ring_mask = (distances <= outer_radius) & (distances >= inner_radius)
        mask[ring_mask] = 1.0
        
    elif mask_type == "Horizontal Stripes":
        stripe_height = height // num_stripes
        stripe_thickness = int(stripe_height * thickness / 100)
        for i in range(num_stripes):
            start = i * stripe_height
            end = start + stripe_thickness
            mask[start:min(end, height), :] = 1.0
            
    elif mask_type == "Vertical Stripes":
        stripe_width = width // num_stripes
        stripe_thickness = int(stripe_width * thickness / 100)
        for i in range(num_stripes):
            start = i * stripe_width
            end = start + stripe_thickness
            mask[:, start:min(end, width)] = 1.0
            
    elif mask_type == "Checkerboard":
        block_size = min(width, height) // num_blocks
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                if (i//block_size + j//block_size) % 2 == 0:
                    mask[i:min(i+block_size, height), j:min(j+block_size, width)] = 1.0
                    
    elif mask_type == "Gradient (Left→Right)":
        for j in range(width):
            mask[:, j] = j / width
            
    elif mask_type == "Gradient (Top→Bottom)":
        for i in range(height):
            mask[i, :] = i / height
    
    # Invert if requested
    if invert:
        mask = 1.0 - mask
    
    # Apply mask
    result = img_np * mask
    
    # Calculate statistics
    mask_coverage = np.mean(mask) * 100
    visible_pixels = np.sum(mask > 0.5)
    total_pixels = height * width
    avg_original = np.mean(img_np)
    avg_masked = np.mean(result[mask > 0.5]) if visible_pixels > 0 else 0
    
    # Display results
    st.markdown("---")
    st.markdown("### 📊 Live Masking Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img_np.astype(np.uint8), caption="🖼️ Original Image", 
                use_container_width=True, channels="L")
        st.metric("Avg Brightness", f"{avg_original:.0f}/255")
        st.caption("Full image visible")
    
    with col2:
        # Display mask properly (convert to uint8)
        mask_display = (mask * 255).astype(np.uint8)
        st.image(mask_display, caption=f"🎭 {mask_type} Mask", 
                use_container_width=True, channels="L")
        st.metric("Coverage", f"{mask_coverage:.1f}%")
        st.caption(f"White = reveal, Black = hide")
    
    with col3:
        st.image(result.astype(np.uint8), caption="✨ Masked Result", 
                use_container_width=True, channels="L")
        st.metric("Visible Pixels", f"{visible_pixels:,}")
        st.caption(f"Original × Mask")
    
    # Formula explanation
    st.info(f"""
    **📐 Formula:** Result = Image × Mask  
    **Coverage:** {mask_coverage:.1f}% of image is visible  
    **Hidden:** {100-mask_coverage:.1f}% of image is hidden (black)
    """)
    
    # Pixel-level demonstration
    st.markdown("---")
    st.markdown("### 🔬 Pixel-Level Math: How Masking Works")
    
    # Sample 3 points
    sample_points = [
        (height//4, width//4, "Top-Left"),
        (height//2, width//2, "Center"),
        (3*height//4, 3*width//4, "Bottom-Right")
    ]
    
    pixel_data = []
    for y, x, location in sample_points:
        orig_val = img_np[y, x]
        mask_val = mask[y, x]
        result_val = result[y, x]
        
        pixel_data.append({
            "Location": location,
            "Original Pixel": f"{orig_val:.0f}",
            "Mask Value": f"{mask_val:.2f}",
            "Calculation": f"{orig_val:.0f} × {mask_val:.2f}",
            "Result": f"{result_val:.0f}",
            "Status": "✅ Visible" if mask_val > 0.5 else "❌ Hidden"
        })
    
    df = pd.DataFrame(pixel_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **💡 Understanding the Math:**
    - When **Mask = 1.0** (white) → Pixel × 1.0 = **Original pixel** (unchanged)
    - When **Mask = 0.0** (black) → Pixel × 0.0 = **0** (completely black)
    - When **Mask = 0.5** (gray) → Pixel × 0.5 = **50% brightness** (dimmed)
    """)
    
    # Side-by-side comparison
    st.markdown("---")
    st.markdown("### 👀 Side-by-Side Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img_np.astype(np.uint8), caption="Before Masking", use_container_width=True, channels="L")
    
    with col2:
        st.image(result.astype(np.uint8), caption="After Masking", use_container_width=True, channels="L")
    
    # Overlay visualization
    st.markdown("---")
    st.markdown("### 🎨 Mask Overlay Visualization")
    st.markdown("See exactly which areas are selected (highlighted in red):")
    
    # Create colored overlay
    img_color = np.stack([img_np, img_np, img_np], axis=2).astype(float)
    overlay = img_color.copy()
    
    # Add red tint to masked areas
    overlay[:, :, 0] = np.clip(img_np + mask * 80, 0, 255)  # Boost red
    overlay[:, :, 1] = img_np * (1 - mask * 0.3)  # Reduce green slightly
    overlay[:, :, 2] = img_np * (1 - mask * 0.5)  # Reduce blue more
    
    st.image(overlay.astype(np.uint8), caption="Red Tint = Selected Region", use_container_width=True)
    
    # Comparison of different masks
    st.markdown("---")
    st.markdown("### 📸 Quick Mask Gallery")
    st.markdown("See how different mask shapes affect the same image:")
    
    # Generate quick examples
    quick_masks = {
        "Circle": lambda h, w: ((np.ogrid[:h, :w][1] - w//2)**2 + (np.ogrid[:h, :w][0] - h//2)**2) <= (min(h,w)//4)**2,
        "Rectangle": lambda h, w: np.pad(np.ones((h//2, w//2)), ((h//4, h//4), (w//4, w//4)), mode='constant'),
        "Cross": lambda h, w: (np.abs(np.ogrid[:h, :w][0] - h//2) < h//8) | (np.abs(np.ogrid[:h, :w][1] - w//2) < w//8),
        "Corners": lambda h, w: ((np.ogrid[:h, :w][0] < h//3) & (np.ogrid[:h, :w][1] < w//3)) | 
                               ((np.ogrid[:h, :w][0] > 2*h//3) & (np.ogrid[:h, :w][1] > 2*w//3))
    }
    
    cols = st.columns(4)
    for idx, (name, mask_func) in enumerate(quick_masks.items()):
        with cols[idx]:
            quick_mask = mask_func(height, width).astype(float)
            quick_result = img_np * quick_mask
            st.image(quick_result.astype(np.uint8), caption=name, use_container_width=True, channels="L")
            coverage = np.mean(quick_mask) * 100
            st.caption(f"{coverage:.0f}% visible")
    
    # Real-world applications
    st.markdown("---")
    st.markdown("### 🌍 Real-World Applications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🏥 Medical Imaging:**
        - **Circle mask**: Isolate tumors or lesions
        - **Rectangle mask**: Define region of interest (ROI)
        - **Ring mask**: Analyze surrounding tissue
        - **Gradient mask**: Measure depth/layers
        
        **Example:** Doctor circles a suspicious area on an X-ray → 
        Software creates circular mask → Analyzes only that region
        """)
        
        st.info("""
        **📸 Photo Editing:**
        - **Circle mask**: Vignette effects (darken edges)
        - **Gradient mask**: Smooth transitions
        - **Custom shapes**: Selective color adjustments
        - **Multiple masks**: Complex compositions
        
        **Example:** Brighten only the subject's face → 
        Create circular mask → Apply brightness adjustment
        """)
    
    with col2:
        st.warning("""
        **🛰️ Satellite Imagery:**
        - **Polygon masks**: Isolate specific land areas
        - **Threshold masks**: Separate land from water
        - **Multi-masks**: Track changes over time
        - **Checkerboard**: Sample multiple regions
        
        **Example:** Monitoring deforestation → 
        Mask forest areas → Compare across years
        """)
        
        st.error("""
        **🔒 Privacy & Security:**
        - **Face masking**: Blur faces in surveillance
        - **License plate masking**: Hide vehicle IDs
        - **Object masking**: Remove sensitive items
        - **Inverse masking**: Keep only backgrounds
        
        **Example:** Blur faces in a crowd photo → 
        Detect faces → Create circular masks → Apply blur
        """)
    
    # Interactive challenges
    st.markdown("---")
    st.markdown("### 🎮 Try These Experiments!")
    
    challenges = [
        "**Spotlight Effect**: Use a circular mask in the center - looks like a spotlight!",
        "**Privacy Mode**: Create a small rectangle in the center, then INVERT the mask - hides everything except that area!",
        "**Zebra Pattern**: Try vertical stripes with 10 stripes and 30% thickness",
        "**Target Practice**: Use the Ring (Donut) mask - looks like a target!",
        "**Fade Away**: Try both gradient masks - see how the image gradually disappears?",
        "**Chess Board**: Use checkerboard with 12 blocks - what patterns do you see?"
    ]
    
    for i, challenge in enumerate(challenges, 1):
        st.markdown(f"{i}. {challenge}")
    
    # Common use cases
    st.markdown("---")
    st.markdown("### 🎯 When to Use Each Mask Type")
    
    use_cases = {
        "Circle": "Isolate single objects, create vignettes, focus attention",
        "Rectangle": "Crop regions, standard ROI analysis, document scanning",
        "Ring (Donut)": "Analyze surrounding areas, create borders, exclude centers",
        "Horizontal/Vertical Stripes": "Sample multiple rows/columns, create venetian blind effects",
        "Checkerboard": "Sample evenly across image, create patterns, data validation",
        "Gradient": "Smooth transitions, fade effects, depth simulation"
    }
    
    for mask_name, use_case in use_cases.items():
        st.markdown(f"- **{mask_name}**: {use_case}")
    
    # Code example
    with st.expander("💻 Python Code: Create Your Own Masks"):
        st.code("""
import numpy as np
from PIL import Image

# Load image
img = np.array(Image.open('image.jpg').convert('L'))
height, width = img.shape

# Create circular mask
y, x = np.ogrid[:height, :width]
center_y, center_x = height // 2, width // 2
radius = min(height, width) // 4
circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
mask = circle_mask.astype(float)

# Apply mask
result = img * mask

# Save
Image.fromarray(result.astype(np.uint8)).save('masked.jpg')

# Pro tip: Combine masks with logical operations
mask1 = circle_mask.astype(float)
mask2 = rectangle_mask.astype(float)
combined = np.maximum(mask1, mask2)  # Union (OR)
intersection = np.minimum(mask1, mask2)  # Intersection (AND)
        """, language="python")
    
    st.markdown("---")
    st.caption("💡 Fun Fact: Instagram filters often use gradient masks to create those trendy fade effects! The 'vignette' effect is just a radial gradient mask.")