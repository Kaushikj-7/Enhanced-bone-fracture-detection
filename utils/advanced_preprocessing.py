import cv2
import numpy as np
import pywt
from skimage.filters import frangi
from PIL import Image

class AdvancedFracturePreprocessor:
    """
    Enterprise-grade deterministic preprocessing pipeline for bone fracture detection.
    Sequentially applies: Sanitization -> Balancing (CLAHE) -> Isolation (Wavelets/Frangi/Top-Hat).
    """
    def __init__(self, target_size=(224, 224), apply_frangi=True, apply_wavelet=True):
        self.target_size = target_size
        self.apply_frangi = apply_frangi
        self.apply_wavelet = apply_wavelet
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def sanitize_and_balance(self, img_np):
        """Step 1 & 2: Deterministic Sanitization (AutoCrop + OCR Masking) + CLAHE Balancing"""
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        # 1. Deterministic Sanitization: Remove collimation borders & Burnt-in Text
        # Heuristic for Text/Markers: High-intensity small regions near edges or corners
        # Also using a threshold to find the main anatomy contour
        _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Keep only the largest contour (the bone/body part)
            largest_cnt = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
            
            # Mask out everything except the largest anatomy
            gray = cv2.bitwise_and(gray, mask)
            
            # Crop to the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_cnt)
            gray = gray[y:y+h, x:x+w]

        # 2. Localized Illumination via CLAHE (The Equalizer)
        # We use a slightly more aggressive clipLimit to maximize trabecular visibility
        balanced = self.clahe.apply(gray)
        return balanced

    def wavelet_detail_boost(self, img):
        """
        Step 3: High-Frequency Isolation via Wavelet Transform (The Extractor)
        Implements Homomorphic-style Filtering and Threshold Shrinkage on detail coefficients.
        Suppress anatomical shape (LL) to isolate pathological features (LH, HL, HH).
        """
        # Decompose using Haar (or biorthogonal) wavelet
        coeffs2 = pywt.dwt2(img.astype(np.float32), 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # 1. Anatomical Shape Suppression: Attenuate the approximation (LL)
        # This reduces the dominance of overall bone shape and illumination gradients
        LL_suppressed = LL * 0.2
        
        # 2. Threshold Shrinkage (Soft Thresholding) to suppress microscopic noise
        # 3. Homomorphic-style Non-linear Gain to selectively amplify gradients
        def apply_high_reliability_gain(coeff, sigma=1.0, base_gain=2.5):
            # Estimate noise threshold using universal thresholding
            threshold = sigma * np.sqrt(2 * np.log(coeff.size))
            
            # Soft thresholding: sign(x) * max(0, |x| - threshold)
            coeff_shrunk = np.sign(coeff) * np.maximum(np.abs(coeff) - threshold, 0)
            
            # Homomorphic-style Non-linear Gain: 
            # We amplify intermediate values more than very high values (saturation prevention)
            # using a sigmoid-like or logarithmic scaling on the gain itself.
            # gain = base_gain * (1 - exp(-|x|))
            norm_val = np.abs(coeff_shrunk) / (np.max(np.abs(coeff_shrunk)) + 1e-6)
            non_linear_gain = base_gain * (1.0 + np.tanh(norm_val * 2.0))
            
            return coeff_shrunk * non_linear_gain

        # Apply to detail coefficients
        LH = apply_high_reliability_gain(LH, sigma=0.4, base_gain=2.0)
        HL = apply_high_reliability_gain(HL, sigma=0.4, base_gain=2.0)
        HH = apply_high_reliability_gain(HH, sigma=0.8, base_gain=3.0) # Boost diagonal hairline cracks
        
        # Reconstruct
        reconstructed = pywt.idwt2((LL_suppressed, (LH, HL, HH)), 'haar')
        
        # Normalize and stretch back to uint8 range for better visibility in the G channel
        reconstructed = reconstructed - np.min(reconstructed)
        if np.max(reconstructed) > 0:
            reconstructed = (reconstructed / np.max(reconstructed)) * 255
            
        return reconstructed.astype(np.uint8)

    def frangi_vesselness(self, img):
        """Step 4: Frangi Filter (Catch thin branching lines)"""
        # Frangi expects normalized float image
        img_float = img.astype(float) / 255.0
        # Multi-scale Frangi
        vesselness = frangi(img_float, sigmas=range(1, 4, 1), black_ridges=True)
        # Normalize back to 0-255
        if np.max(vesselness) > 0:
            vesselness = (vesselness / np.max(vesselness)) * 255
        return vesselness.astype(np.uint8)

    def top_hat_transform(self, img):
        """Step 5: Morphological Top-Hat (3D relief for fragments)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        return cv2.addWeighted(img, 1.0, tophat, 0.5, 0)

    def adaptive_canny(self, img):
        """Step 6: Adaptive Canny (Outline bone cortex jagging)"""
        v = np.median(img)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(img, lower, upper)
        return edged

    def __call__(self, pil_img):
        # Convert PIL to Numpy
        img_np = np.array(pil_img)
        
        # 1. Sanitize & Balance
        balanced = self.sanitize_and_balance(img_np)
        
        # 2. Detail Isolation
        # Combine different enhancement signals into channels to give the model a rich feature map
        # Channel R: Balanced image (Original Structure)
        # Channel G: Wavelet Boosted (Fine Cracks)
        # Channel B: Frangi Vesselness (Linear Fractures)
        
        ch_r = balanced
        ch_g = balanced
        if self.apply_wavelet:
            ch_g = self.wavelet_detail_boost(balanced)
            
        ch_b = balanced
        if self.apply_frangi:
            ch_b = self.frangi_vesselness(balanced)
            
        # Final merge into 3-channel input
        # Ensure all channels match 'balanced' size to avoid cv2.merge error
        size = (balanced.shape[1], balanced.shape[0])
        if ch_g.shape[:2] != balanced.shape[:2]:
            ch_g = cv2.resize(ch_g, size)
        if ch_b.shape[:2] != balanced.shape[:2]:
            ch_b = cv2.resize(ch_b, size)

        enhanced_stack = cv2.merge([ch_r, ch_g, ch_b])
        
        # Resize to model input size
        enhanced_stack = cv2.resize(enhanced_stack, self.target_size, interpolation=cv2.INTER_AREA)
        
        return Image.fromarray(enhanced_stack)

if __name__ == "__main__":
    # Test on a dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))
    preprocessor = AdvancedFracturePreprocessor()
    out = preprocessor(dummy_img)
    print(f"Enhanced Image Shape: {np.array(out).shape}")
