import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

def app():
    st.title("🎯 Noise Reduction via Averaging")
    
    # Simple analogy first
    st.markdown("""
    ### 🤔 The Core Idea (Simple Analogy)
    
    Imagine you're trying to measure your exact height, but your measuring tape is a bit wobbly:
    - **One measurement:** 170.3 cm (could be wrong due to wobbles)
    - **Ten measurements:** 169.8, 170.2, 170.1, 170.5... → Average = **170.1 cm** (much more accurate!)
    
    **The same principle works with images!** 
    - Random noise = "wobbles" that affect each pixel differently each time
    - Multiple noisy images = multiple "measurements" 
    - Averaging = finding the true value by canceling out random errors
    """)
    
    st.markdown("---")
    
    # Load image
    img = Image.open("public/ch1.1.jpg").convert('L')
    img_np = np.array(img).astype(float)
    
    # Controls
    st.markdown("### 🎛️ Experiment Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        noise_level = st.slider("🔊 Noise Intensity", 
                                min_value=10, max_value=80, value=30,
                                help="Higher = more grainy/snowy image")
    
    with col2:
        num_images = st.slider("📸 Images to Average", 
                               min_value=2, max_value=50, value=10,
                               help="More images = better noise reduction")
    
    with col3:
        show_theory = st.checkbox("📚 Show Math Theory", value=False)
    
    # Theory section (optional)
    if show_theory:
        st.info("""
        **Mathematical Principle:**
        - Noise follows Gaussian distribution: N(0, σ²)
        - When averaging n independent samples, noise variance reduces by factor of n
        - Standard deviation of averaged noise: σ/√n
        - Signal (original image) remains constant while noise decreases!
        """)
    
    # Generate noisy images
    np.random.seed(42)  # For reproducibility
    noisy_images = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_images):
        status_text.text(f"Generating noisy image {i+1}/{num_images}...")
        noise = np.random.normal(0, noise_level, img_np.shape)
        noisy = img_np + noise
        noisy = np.clip(noisy, 0, 255)
        noisy_images.append(noisy)
        progress_bar.progress((i + 1) / num_images)
    
    status_text.empty()
    progress_bar.empty()
    
    # Calculate cumulative averages
    average_image = np.mean(noisy_images, axis=0)
    
    # Main comparison
    st.markdown("---")
    st.markdown("### 📊 Main Comparison: The Power of Averaging")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img_np.astype(np.uint8), caption="✨ Original (Clean)", 
                use_container_width=True, channels="L")
        st.success("**Perfect Quality**")
    
    with col2:
        st.image(noisy_images[0].astype(np.uint8), caption="😵 Single Noisy Image", 
                use_container_width=True, channels="L")
        mse1 = np.mean((img_np - noisy_images[0]) ** 2)
        psnr1 = 10 * np.log10((255**2) / (mse1 + 1e-10))
        st.warning(f"**PSNR: {psnr1:.1f} dB** (Poor)")
    
    with col3:
        st.image(average_image.astype(np.uint8), caption=f"🎯 Average of {num_images} Images", 
                use_container_width=True, channels="L")
        mse_avg = np.mean((img_np - average_image) ** 2)
        psnr_avg = 10 * np.log10((255**2) / (mse_avg + 1e-10))
        improvement = psnr_avg - psnr1
        st.success(f"**PSNR: {psnr_avg:.1f} dB** (+{improvement:.1f} dB better!)")
    
    # Visual explanation of what's happening
    st.markdown("---")
    st.markdown("### 🔍 What's Happening: Pixel-Level View")
    
    st.markdown("""
    Let's zoom into a single pixel to see how averaging works:
    """)
    
    # Pick a random pixel to demonstrate
    y, x = img_np.shape[0] // 2, img_np.shape[1] // 2
    true_value = img_np[y, x]
    noisy_values = [noisy_images[i][y, x] for i in range(min(10, num_images))]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        **Example Pixel (center of image):**
        - True value: **{true_value:.0f}**
        - Noisy measurements: {', '.join([f'{v:.0f}' for v in noisy_values[:5]])}...
        - Average of {min(10, num_images)}: **{np.mean(noisy_values):.0f}**
        - Error reduced from ±{noise_level} to ±{noise_level/np.sqrt(min(10, num_images)):.1f}!
        """)
    
    with col2:
        # Create DataFrame for bar chart
        chart_data = pd.DataFrame({
            'Measurement': [f'#{i+1}' for i in range(len(noisy_values))],
            'Pixel Value': noisy_values,
            'True Value': [true_value] * len(noisy_values),
            'Average': [np.mean(noisy_values)] * len(noisy_values)
        })
        
        st.markdown("**📊 Noisy Measurements vs True Value**")
        st.bar_chart(chart_data.set_index('Measurement')['Pixel Value'])
        st.caption(f"Green line would be True Value ({true_value:.0f}), Red line would be Average ({np.mean(noisy_values):.0f})")
    
    # Progressive improvement visualization
    st.markdown("---")
    st.markdown("### 📈 Progressive Improvement: Watch Quality Increase")
    
    steps = [1, 2, 3, 5, 10, 20, 30, 50]
    steps = [s for s in steps if s <= num_images]
    
    cols_per_row = 4
    for row_start in range(0, len(steps), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, step in enumerate(steps[row_start:row_start + cols_per_row]):
            with cols[idx]:
                avg_n = np.mean(noisy_images[:step], axis=0)
                st.image(avg_n.astype(np.uint8), 
                        caption=f"Average of {step} image{'s' if step > 1 else ''}", 
                        use_container_width=True, channels="L")
                mse_n = np.mean((img_np - avg_n) ** 2)
                psnr_n = 10 * np.log10((255**2) / (mse_n + 1e-10))
                st.metric("PSNR", f"{psnr_n:.1f} dB")
    
    # Quality improvement graph
    st.markdown("---")
    st.markdown("### 📉 Quality vs Number of Images")
    
    n_range = range(1, min(num_images + 1, 51))
    psnr_curve = []
    
    for n in n_range:
        avg_n = np.mean(noisy_images[:n], axis=0)
        mse_n = np.mean((img_np - avg_n) ** 2)
        psnr_n = 10 * np.log10((255**2) / (mse_n + 1e-10))
        psnr_curve.append(psnr_n)
    
    # Create DataFrame for line chart
    quality_data = pd.DataFrame({
        'Number of Images': list(n_range),
        'PSNR (dB)': psnr_curve
    })
    
    st.line_chart(quality_data.set_index('Number of Images'))
    st.caption("📈 Notice how quality improves rapidly at first, then levels off!")
    
    # Show numerical improvements
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("1 Image", f"{psnr_curve[0]:.1f} dB", delta=None)
    with col2:
        mid_idx = len(psnr_curve) // 2
        st.metric(f"{mid_idx + 1} Images", f"{psnr_curve[mid_idx]:.1f} dB", 
                 delta=f"+{psnr_curve[mid_idx] - psnr_curve[0]:.1f} dB")
    with col3:
        st.metric(f"{len(psnr_curve)} Images", f"{psnr_curve[-1]:.1f} dB", 
                 delta=f"+{psnr_curve[-1] - psnr_curve[0]:.1f} dB")
    
    # Key takeaways
    st.markdown("---")
    st.markdown("### 💡 Key Takeaways")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Why This Works:**
        1. Noise is **random** - different in each image
        2. True signal is **constant** - same in each image
        3. Random values average toward zero
        4. Constant values stay constant
        5. Result: Signal preserved, noise reduced!
        """)
    
    with col2:
        st.info("""
        **🌍 Real Applications:**
        - **Astronomy:** Stacking telescope images
        - **Medicine:** Improving MRI/CT scans
        - **Photography:** Night mode on smartphones
        - **Security:** Enhancing surveillance footage
        - **Science:** Reducing measurement errors
        """)
    
    # Interactive challenge
    st.markdown("---")
    st.markdown("### 🎮 Try It Yourself!")
    st.markdown("""
    **Experiment Ideas:**
    1. Set noise to maximum (80) with just 2 images - see how noisy it is?
    2. Keep the same high noise but use 50 images - watch the magic!
    3. Find the minimum number of images needed to get "good" quality (PSNR > 30 dB)
    
    **Challenge Question:** If averaging 10 images improves PSNR by X dB, 
    how much improvement would you expect from 40 images? (Hint: It's not 4X!)
    """)
    
    # Answer to challenge (hidden in expander)
    with st.expander("💡 Click to see the answer"):
        st.markdown("""
        **Answer:** About 1.5X improvement!
        
        **Why?** The improvement follows a **logarithmic** pattern, not linear:
        - Noise reduction is proportional to √n (square root of number of images)
        - Going from 10 to 40 images means 2× more images
        - But noise only reduces by √2 ≈ 1.41 times
        - So PSNR improvement is about 1.5× the original improvement
        
        This is why the graph curves - you get diminishing returns!
        """)
    
    st.markdown("---")
    st.caption("💡 Tip: PSNR (Peak Signal-to-Noise Ratio) measures quality. Higher PSNR = better quality. >30 dB is generally considered good.")