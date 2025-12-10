import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import os

def app():
    st.title("🎯 Fourier Transform: The Magic Behind Frequency Domain")
    
    # Progressive learning tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ What is Fourier Transform?", 
        "2️⃣ Time → Frequency", 
        "3️⃣ DFT in Action",
        "4️⃣ 2D DFT for Images"
    ])
    
    # Load image
    image_dir = "public/lab5"
    if os.path.exists(image_dir):
        available_images = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if available_images:
            selected_image = st.sidebar.selectbox("📷 Choose Image", available_images, key="ft_img")
            img_path = os.path.join(image_dir, selected_image)
            img = Image.open(img_path).convert('L')
        else:
            img = create_demo_image()
    else:
        img = create_demo_image()
    
    img_np = np.array(img).astype(float)
    
    # ==================== TAB 1: WHAT IS FT ====================
    with tab1:
        st.header("🎯 What is Fourier Transform?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎵 Musical Analogy
            
            Imagine you hear a **chord** on a piano:
            
            C + E + G played together
            
            Your ears hear one sound, but...
            
            **Fourier Transform** can tell you:
            - Note C (261.6 Hz)
            - Note E (329.6 Hz)  
            - Note G (392.0 Hz)
            
            It **decomposes** the sound into individual notes!
            """)
        
        with col2:
            st.markdown("""
            ### 🖼️ For Images
            
            An image is like that chord - a complex mix!
            
            **Fourier Transform** decomposes it into:
            - **Sine waves** (smooth patterns)
            - **Cosine waves** (different smooth patterns)
            - At different **frequencies**
            
            Each wave has:
            - **Amplitude** (how strong)
            - **Phase** (where it starts)
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🔄 The Transform Pairs
        
        **Forward FT:**
        ```
        Image → Frequencies
        (What we see) → (What makes it up)
        ```
        
        **Inverse FT:**
        ```
        Frequencies → Image  
        (Recipe) → (Final dish)
        ```
        
        **No information lost!** You can go back and forth perfectly! 🎯
        """)
        
        # Visual representation
        st.markdown("### 📊 Visualizing the Concept")
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Create example signals
        t = np.linspace(0, 2*np.pi, 100)
        
        # Signal 1: Simple sine wave
        axes[0].plot(t, np.sin(t))
        axes[0].set_title("Single Frequency")
        axes[0].set_xlabel("Time/Space")
        axes[0].set_ylabel("Amplitude")
        
        # Signal 2: Two frequencies
        axes[1].plot(t, np.sin(t) + 0.5*np.sin(3*t))
        axes[1].set_title("Two Frequencies")
        axes[1].set_xlabel("Time/Space")
        
        # Signal 3: Complex mix
        axes[2].plot(t, np.sin(t) + 0.5*np.sin(3*t) + 0.3*np.sin(5*t))
        axes[2].set_title("Three Frequencies")
        axes[2].set_xlabel("Time/Space")
        
        st.pyplot(fig)
        
        st.info("""
        **Key Insight:** Any complex signal (right) can be broken down into 
        simple sine waves (left) using Fourier Transform!
        """)
    
    # ==================== TAB 2: TIME → FREQUENCY ====================
    with tab2:
        st.header("⏱️ → 🎵 Time/Space Domain to Frequency Domain")
        
        st.markdown("""
        ### Let's see the transformation step by step
        """)
        
        # Create a simple 1D example
        st.markdown("### 📈 1D Example First (Easier to Understand)")
        
        # Create a simple signal
        t = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(t) + 0.5*np.sin(3*t + np.pi/4) + 0.3*np.sin(5*t + np.pi/2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🕐 Time/Space Domain")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t, signal, 'b-', linewidth=2)
            ax.set_xlabel("Position (x)")
            ax.set_ylabel("Intensity")
            ax.set_title("What We See: Mixed Signal")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.markdown("""
            **What this shows:**
            - Signal intensity at each position
            - The final result of mixing
            - But NOT what's inside!
            """)
        
        with col2:
            st.markdown("#### 🎵 Frequency Domain (after FT)")
            # Calculate FFT
            fft_signal = np.fft.fft(signal)
            freq = np.fft.fftfreq(len(t))
            magnitude = np.abs(fft_signal)
            
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.stem(freq[:50], magnitude[:50], 'r-', markerfmt='ro')
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Magnitude (Strength)")
            ax.set_title("What FT Sees: Component Frequencies")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.markdown("""
            **What this shows:**
            - **Peak 1**: Main frequency (strongest)
            - **Peak 2**: 3× frequency (medium)
            - **Peak 3**: 5× frequency (weakest)
            - Each peak = one sine wave component
            """)
        
        st.markdown("---")
        
        st.markdown("### 🔍 Interactive Decomposition")
        
        # Let user control signal composition
        st.markdown("Build your own signal and see its frequency components:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amp1 = st.slider("Frequency 1 Amplitude", 0.0, 2.0, 1.0, 0.1)
            freq1 = st.slider("Frequency 1 Value", 1, 10, 1, 1)
        
        with col2:
            amp2 = st.slider("Frequency 2 Amplitude", 0.0, 2.0, 0.5, 0.1)
            freq2 = st.slider("Frequency 2 Value", 1, 10, 3, 1)
        
        with col3:
            amp3 = st.slider("Frequency 3 Amplitude", 0.0, 2.0, 0.3, 0.1)
            freq3 = st.slider("Frequency 3 Value", 1, 10, 5, 1)
        
        # Create custom signal
        custom_signal = (amp1 * np.sin(freq1 * t) + 
                        amp2 * np.sin(freq2 * t + np.pi/4) + 
                        amp3 * np.sin(freq3 * t + np.pi/2))
        
        # Calculate FFT
        custom_fft = np.fft.fft(custom_signal)
        custom_magnitude = np.abs(custom_fft)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(t, custom_signal, 'b-', linewidth=2)
        axes[0].set_title("Your Custom Signal")
        axes[0].set_xlabel("Position")
        axes[0].set_ylabel("Intensity")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].stem(freq[:50], custom_magnitude[:50], 'r-', markerfmt='ro')
        axes[1].set_title("Frequency Components")
        axes[1].set_xlabel("Frequency")
        axes[1].set_ylabel("Strength")
        axes[1].grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.success(f"""
        ✅ **Analysis:**
        - You should see peaks at frequencies {freq1}, {freq2}, and {freq3}
        - Peak heights proportional to amplitudes: {amp1:.1f}, {amp2:.1f}, {amp3:.1f}
        - This is exactly what Fourier Transform does - finds what's inside!
        """)
    
    # ==================== TAB 3: DFT IN ACTION ====================
    with tab3:
        st.header("💻 Discrete Fourier Transform (DFT) in Action")
        
        st.markdown("""
        ### From Continuous to Digital
        
        Real images are **discrete** (pixels), not continuous!
        
        So we use **Discrete Fourier Transform (DFT)** instead of continuous FT.
        """)
        
        st.markdown("#### 📝 The DFT Formula (1D):")
        
        st.latex(r'''
        F(k) = \sum_{n=0}^{N-1} f(n) \cdot e^{-j2\pi kn/N}
        ''')
        
        st.markdown("Where:")
        st.markdown("- \( f(n) \) = pixel values at position \( n \)")
        st.markdown("- \( F(k) \) = frequency component at frequency \( k \)")
        st.markdown("- \( N \) = total number of pixels")
        st.markdown("- \( e^{-j2\pi kn/N} \) = complex exponential (sine + cosine)")
        
        st.markdown("---")
        
        st.markdown("### 🧮 Let's Compute DFT Step by Step")
        
        # Create a tiny signal for manual computation
        tiny_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        
        st.markdown(f"#### Our tiny signal: `{tiny_signal}`")
        
        if st.checkbox("Show manual computation steps", False):
            st.markdown("**Step 1:** Choose a frequency k (let's pick k=1)")
            st.markdown("**Step 2:** For each pixel n, compute:")
            st.latex(r'''f(n) \cdot e^{-j2\pi \cdot 1 \cdot n / 8}''')
            
            st.markdown("**Step 3:** Since \( e^{-j\theta} = \cos(\theta) - j\sin(\theta) \):")
            
            n_values = np.arange(8)
            angles = -2 * np.pi * 1 * n_values / 8
            cos_vals = np.cos(angles)
            sin_vals = np.sin(angles)
            
            df = st.dataframe({
                'n': n_values,
                'f(n)': tiny_signal,
                'cos(θ)': cos_vals.round(3),
                'sin(θ)': sin_vals.round(3),
                'Real part': (tiny_signal * cos_vals).round(3),
                'Imag part': (tiny_signal * -sin_vals).round(3)
            }, use_container_width=True)
            
            real_sum = np.sum(tiny_signal * cos_vals)
            imag_sum = np.sum(tiny_signal * -sin_vals)
            
            st.markdown(f"**Step 4:** Sum real parts = {real_sum:.3f}")
            st.markdown(f"**Step 5:** Sum imaginary parts = {imag_sum:.3f}")
            st.markdown(f"**Result:** F(1) = {real_sum:.3f} + j{imag_sum:.3f}")
        
        # Compare manual vs numpy
        numpy_dft = np.fft.fft(tiny_signal)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Manual Understanding")
            st.markdown("""
            For each frequency k:
            1. Create a wave pattern
            2. Multiply with signal
            3. Sum results
            4. Repeat for all k
            
            **Output:** Complex numbers showing:
            - How much of each frequency exists
            - Phase information
            """)
        
        with col2:
            st.markdown("#### NumPy FFT Result")
            st.dataframe({
                'k (frequency)': range(8),
                'Real(F(k))': np.real(numpy_dft).round(3),
                'Imag(F(k))': np.imag(numpy_dft).round(3),
                '|F(k)|': np.abs(numpy_dft).round(3)
            }, use_container_width=True)
        
        st.info("""
        💡 **Key Points:**
        - DFT works on **discrete samples** (pixels)
        - Output is **complex numbers** (Real + j*Imag)
        - **Magnitude** = how strong the frequency is
        - **Phase** = where the wave pattern starts
        """)
    
    # ==================== TAB 4: 2D DFT FOR IMAGES ====================
    with tab4:
        st.header("🖼️ 2D DFT for Images")
        
        st.markdown("""
        Images are 2D (width × height), so we need **2D DFT**!
        
        Instead of time → frequency, we have:
        - **x position** → **horizontal frequency (u)**
        - **y position** → **vertical frequency (v)**
        """)
        
        st.latex(r'''
        F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \cdot e^{-j2\pi (ux/M + vy/N)}
        ''')
        
        # Apply 2D DFT to our image
        fft_2d = fft2(img_np)
        fft_shifted = fftshift(fft_2d)
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log(magnitude + 1)
        
        # Get real and imaginary parts
        real_part = np.real(fft_shifted)
        imag_part = np.imag(fft_shifted)
        
        st.markdown("---")
        
        st.markdown("### 🔬 Explore 2D DFT Components")
        
        component = st.radio(
            "Select DFT component to view:",
            ["Magnitude Spectrum", "Real Part", "Imaginary Part", "All Three"]
        )
        
        if component == "Magnitude Spectrum":
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img_np, cmap='gray')
                ax.set_title("Original Image")
                ax.axis('off')
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(magnitude_log, cmap='gray')
                ax.set_title("Magnitude Spectrum")
                ax.axis('off')
                st.pyplot(fig)
            
            st.markdown("""
            **Magnitude Spectrum shows:**
            - Bright center = low frequencies (average brightness)
            - Pattern from center = frequency directions
            - Bright spots = strong frequencies in that direction
            """)
        
        elif component == "Real Part":
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img_np, cmap='gray')
                ax.set_title("Original Image")
                ax.axis('off')
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(real_part, cmap='gray')
                ax.set_title("Real Part (Cosine components)")
                ax.axis('off')
                st.pyplot(fig)
            
            st.markdown("""
            **Real Part shows:**
            - Cosine wave patterns
            - Even symmetry
            - Strength of patterns aligned with cosine waves
            """)
        
        elif component == "Imaginary Part":
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(img_np, cmap='gray')
                ax.set_title("Original Image")
                ax.axis('off')
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(imag_part, cmap='gray')
                ax.set_title("Imaginary Part (Sine components)")
                ax.axis('off')
                st.pyplot(fig)
            
            st.markdown("""
            **Imaginary Part shows:**
            - Sine wave patterns
            - Odd symmetry
            - Phase information (where patterns start)
            """)
        
        else:  # All Three
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            axes[0, 0].imshow(img_np, cmap='gray')
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(magnitude_log, cmap='gray')
            axes[0, 1].set_title("Magnitude Spectrum")
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(real_part, cmap='gray')
            axes[1, 0].set_title("Real Part (Cosine)")
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(imag_part, cmap='gray')
            axes[1, 1].set_title("Imaginary Part (Sine)")
            axes[1, 1].axis('off')
            
            st.pyplot(fig)
            
            st.success("""
            **Putting it all together:**
            - **Magnitude**: How strong each frequency is
            - **Real Part**: Cosine components (even symmetry)
            - **Imaginary Part**: Sine components (odd symmetry)
            - **Together**: Complete frequency representation!
            """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Interactive Frequency Explorer")
        
        st.markdown("Click on the spectrum to see what that frequency represents:")
        
        # Create a grid for frequency selection
        freq_u = st.slider("Horizontal Frequency (u)", -50, 50, 0, 1)
        freq_v = st.slider("Vertical Frequency (v)", -50, 50, 0, 1)
        
        # Create frequency pattern
        rows, cols = img_np.shape
        y, x = np.ogrid[:rows, :cols]
        
        # Create wave pattern for selected frequency
        pattern_real = np.cos(2 * np.pi * (freq_u * x / cols + freq_v * y / rows))
        pattern_imag = np.sin(2 * np.pi * (freq_u * x / cols + freq_v * y / rows))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"#### Wave Pattern (u={freq_u}, v={freq_v})")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(pattern_real, cmap='gray')
            ax.set_title("Cosine Wave")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### In Spectrum")
            # Mark position in spectrum
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(magnitude_log, cmap='gray')
            center_x, center_y = cols//2, rows//2
            ax.plot(center_x + freq_u, center_y + freq_v, 'ro', markersize=10)
            ax.set_title("Position in Spectrum")
            ax.axis('off')
            st.pyplot(fig)
        
        with col3:
            st.markdown("#### What it Means")
            if freq_u == 0 and freq_v == 0:
                st.success("**DC Component**\nAverage brightness of image")
            elif abs(freq_u) < 10 and abs(freq_v) < 10:
                st.info(f"**Low Frequency**\nSmooth, gradual changes")
            else:
                st.warning(f"**High Frequency**\nRapid changes, edges, details")
            
            # Check if this frequency is strong in image
            idx_u = center_x + freq_u
            idx_v = center_y + freq_v
            
            if 0 <= idx_u < cols and 0 <= idx_v < rows:
                strength = magnitude_log[idx_v, idx_u]
                st.metric("Strength in Image", f"{strength:.1f}")
        
        st.markdown("---")
        
        st.info("""
        ### 📚 Summary: Fourier Transform for Images
        
        1. **Images → Frequencies**: FT decomposes images into wave patterns
        2. **2D DFT**: Images need 2D transform (horizontal + vertical frequencies)
        3. **Complex Output**: Real + Imaginary parts give complete information
        4. **Magnitude**: Shows frequency strength (what we usually visualize)
        5. **Spectrum**: Center = low frequencies, edges = high frequencies
        
        **Next**: We'll learn to interpret and manipulate these frequencies! 🚀
        """)

def create_demo_image():
    """Create a synthetic image for demonstration"""
    size = 256
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add some patterns
    for i in range(size):
        for j in range(size):
            # Low frequency gradient
            img[i, j] = 128 + 50 * np.sin(0.05 * i) + 50 * np.sin(0.05 * j)
    
    # Add some lines (medium/high frequency)
    img[100:120, :] = 255  # Horizontal line
    img[:, 100:120] = 200  # Vertical line
    
    # Add diagonal pattern
    for i in range(size):
        for j in range(size):
            if (i + j) % 20 < 10:
                img[i, j] = min(255, img[i, j] + 50)
    
    return Image.fromarray(img.astype(np.uint8))

if __name__ == "__main__":
    app()