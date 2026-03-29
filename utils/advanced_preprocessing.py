import cv2
import numpy as np
import pywt
from skimage.filters import frangi
from PIL import Image


class AdvancedFracturePreprocessor:
    """
    Enterprise-grade deterministic preprocessing pipeline for bone fracture detection.
    Sequentially applies: Sanitization -> Balancing (CLAHE) -> Isolation (Wavelets/Frangi).
    """

    def __init__(self, target_size=(224, 224), apply_frangi=True, apply_wavelet=True):
        self.target_size = target_size
        self.apply_frangi = apply_frangi
        self.apply_wavelet = apply_wavelet
        # Don't initialize CLAHE here (causes pickling errors on Windows)
        self._clahe = None

    def sanitize_and_balance(self, img_np):
        """Step 1 & 2: Deterministic Sanitization (AutoCrop + Mask Text) + CLAHE Balancing"""
        # Ensure uint8
        if img_np.dtype != np.uint8:
            img_np = (img_np / (img_np.max() + 1e-6) * 255).astype(np.uint8)

        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        # 1. Deterministic Sanitization: Remove collimation borders
        _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
            gray = cv2.bitwise_and(gray, mask)

            x, y, w, h = cv2.boundingRect(largest_cnt)
            # Ensure crop is not empty
            if w > 10 and h > 10:
                gray = gray[y : y + h, x : x + w]

        # 2. Resize intermediate for text masking
        gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)

        # 3. Mask text/annotations (bright small connected components near borders)
        self._mask_text_regions(gray)

        # 4. Localized Illumination via CLAHE
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        balanced = self._clahe.apply(gray)
        return balanced

    def _mask_text_regions(self, img):
        """Inpaint bright text markers."""
        # Detect bright small blobs in border regions (top 20%, bottom 20%)
        h, w = img.shape
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[: int(h * 0.2), :] = 255
        border_mask[int(h * 0.8) :, :] = 255
        border_mask[:, : int(w * 0.2)] = 255
        border_mask[:, int(w * 0.8) :] = 255

        _, bright = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        text_mask = cv2.bitwise_and(bright, border_mask)

        # Dilate slightly so inpaint covers full character
        kernel = np.ones((5, 5), np.uint8)
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)

        # Inpaint with mean value instead of TELEA (faster, no library dep)
        mean_val = int(np.mean(img[text_mask == 0])) if np.any(text_mask == 0) else 0
        img[text_mask > 0] = mean_val

    def wavelet_detail_boost(self, img):
        """
        Step 3: High-Frequency Isolation via Wavelet Transform.
        """
        coeffs2 = pywt.dwt2(img.astype(np.float32), "haar")
        LL, (LH, HL, HH) = coeffs2

        # Suppress approximation (anatomical shape)
        LL_suppressed = LL * 0.1

        def apply_high_reliability_gain(coeff, sigma=0.4, base_gain=2.5):
            # Soft thresholding
            threshold = sigma * np.sqrt(2 * np.log(coeff.size + 1e-6))
            coeff_shrunk = np.sign(coeff) * np.maximum(np.abs(coeff) - threshold, 0)

            # Non-linear Gain
            norm_val = np.abs(coeff_shrunk) / (np.max(np.abs(coeff_shrunk)) + 1e-6)
            non_linear_gain = base_gain * (1.0 + np.tanh(norm_val * 2.0))
            return coeff_shrunk * non_linear_gain

        LH = apply_high_reliability_gain(LH, sigma=0.4, base_gain=2.0)
        HL = apply_high_reliability_gain(HL, sigma=0.4, base_gain=2.0)
        HH = apply_high_reliability_gain(HH, sigma=0.8, base_gain=3.0)

        reconstructed = pywt.idwt2((LL_suppressed, (LH, HL, HH)), "haar")

        # Normalize to uint8
        reconstructed = reconstructed - np.min(reconstructed)
        if np.max(reconstructed) > 0:
            reconstructed = (reconstructed / np.max(reconstructed)) * 255

        return reconstructed.astype(np.uint8)

    def frangi_vesselness(self, img):
        """Step 4: Frangi Filter for linear fracture lines"""
        img_float = img.astype(float) / 255.0
        vesselness = frangi(img_float, sigmas=range(1, 4, 1), black_ridges=True)
        if np.max(vesselness) > 0:
            vesselness = (vesselness / np.max(vesselness)) * 255
        return vesselness.astype(np.uint8)

    def __call__(self, pil_img):
        img_np = np.array(pil_img)
        balanced = self.sanitize_and_balance(img_np)

        ch_r = balanced

        # If disabled, we just replicate the balanced image to 3 channels (Less is More)
        if not self.apply_wavelet and not self.apply_frangi:
            enhanced_stack = cv2.merge([ch_r, ch_r, ch_r])
        else:
            ch_g = (
                self.wavelet_detail_boost(balanced) if self.apply_wavelet else balanced
            )
            ch_b = self.frangi_vesselness(balanced) if self.apply_frangi else balanced

            # Match sizes
            size = (balanced.shape[1], balanced.shape[0])
            if ch_g.shape[:2] != balanced.shape[:2]:
                ch_g = cv2.resize(ch_g, size)
            if ch_b.shape[:2] != balanced.shape[:2]:
                ch_b = cv2.resize(ch_b, size)

            enhanced_stack = cv2.merge([ch_r, ch_g, ch_b])

        enhanced_stack = cv2.resize(
            enhanced_stack, self.target_size, interpolation=cv2.INTER_AREA
        )
        return Image.fromarray(enhanced_stack)
