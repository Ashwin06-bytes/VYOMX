"""
preprocessing.py -- Module 3
============================
Frame Enhancement & Noise Reduction for VYOMX Pipeline.

Operations (applied in this order):
  1. Gaussian blur          -- removes high-frequency sensor noise (sigma=1.0)
  2. CLAHE                  -- Contrast Limited Adaptive Histogram Equalisation
                              enhances local contrast without amplifying noise
  3. Sobel edge detection   -- sharpens structural boundaries (optional overlay)

All parameters are controlled via config.yaml['preprocessing'].

Outputs:
  - enhanced frame as np.ndarray uint8
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("OpenCV not installed -- preprocessing will use fallbacks.")

try:
    from scipy.ndimage import gaussian_filter, sobel as scipy_sobel
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class Preprocessing:
    """
    Image enhancement pipeline for satellite thermal frames.

    Parameters
    ----------
    config : dict
        Sub-dictionary from config.yaml['preprocessing'].
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}

        self.gaussian_sigma  : float      = float(self.config.get("gaussian_sigma",  1.0))
        self.clahe_clip      : float      = float(self.config.get("clahe_clip_limit", 2.0))
        self.clahe_tile      : Tuple[int,int] = tuple(
            self.config.get("clahe_tile_grid", [8, 8])
        )
        self.sobel_ksize     : int        = int(self.config.get("sobel_ksize", 3))
        self.apply_sobel     : bool       = self.config.get("apply_sobel",  True)
        self.apply_clahe     : bool       = self.config.get("apply_clahe",  True)
        self.apply_gaussian  : bool       = self.config.get("apply_gaussian", True)

        logger.info(
            "Preprocessing initialised | gaussian_sigma=%.1f | clahe_clip=%.1f "
            "| apply_sobel=%s | apply_clahe=%s | apply_gaussian=%s",
            self.gaussian_sigma, self.clahe_clip,
            self.apply_sobel, self.apply_clahe, self.apply_gaussian,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(
        self,
        frame: np.ndarray,
        label: str = "frame",
    ) -> np.ndarray:
        """
        Run the full enhancement pipeline on a single frame.

        Parameters
        ----------
        frame : np.ndarray (H, W) uint8
        label : Human-readable label for log messages.

        Returns
        -------
        enhanced : np.ndarray (H, W) uint8
        """
        logger.info("Preprocessing: enhancing '%s' | shape=%s", label, frame.shape)

        result = frame.astype(np.uint8).copy()

        # Step 1 -- Gaussian blur (noise reduction)
        if self.apply_gaussian:
            result = self._gaussian_blur(result)
            logger.debug("[%s] Gaussian blur applied (sigma=%.1f)", label, self.gaussian_sigma)

        # Step 2 -- CLAHE (local contrast enhancement)
        if self.apply_clahe:
            result = self._clahe(result)
            logger.debug("[%s] CLAHE applied (clip=%.1f tile=%s)", label, self.clahe_clip, self.clahe_tile)

        # Step 3 -- Sobel edge overlay (optional sharpening)
        if self.apply_sobel:
            result = self._sobel_enhance(result)
            logger.debug("[%s] Sobel edge enhancement applied", label)

        logger.info(
            "Preprocessing complete | '%s' | out_range=[%d, %d]",
            label, int(result.min()), int(result.max()),
        )
        return result

    def enhance_pair(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhance both frames using the same pipeline.

        Parameters
        ----------
        frame_a : np.ndarray (H, W) uint8
        frame_b : np.ndarray (H, W) uint8

        Returns
        -------
        (enhanced_a, enhanced_b) : Tuple of enhanced frames.
        """
        enhanced_a = self.enhance(frame_a, label="frame_A")
        enhanced_b = self.enhance(frame_b, label="frame_B")
        return enhanced_a, enhanced_b

    def get_edge_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a standalone Sobel edge magnitude map (not blended).

        Useful for visualisation and structural analysis.

        Parameters
        ----------
        frame : np.ndarray (H, W) uint8

        Returns
        -------
        edge_map : np.ndarray (H, W) uint8
        """
        return self._compute_sobel_magnitude(frame)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur for noise suppression.
        Uses OpenCV if available, otherwise falls back to scipy.
        """
        if _HAS_CV2:
            # Kernel size must be odd; derive from sigma
            k = max(3, int(self.gaussian_sigma * 6) | 1)
            return cv2.GaussianBlur(frame, (k, k), self.gaussian_sigma)
        elif _HAS_SCIPY:
            blurred = gaussian_filter(frame.astype(np.float32), sigma=self.gaussian_sigma)
            return np.clip(blurred, 0, 255).astype(np.uint8)
        else:
            logger.warning("No Gaussian blur backend available -- skipping.")
            return frame

    def _clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).
        Uses OpenCV's createCLAHE if available, otherwise falls back to
        a global histogram equalisation.
        """
        if _HAS_CV2:
            clahe_obj = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=self.clahe_tile,
            )
            return clahe_obj.apply(frame)
        else:
            # Simple fallback: global histogram stretch
            logger.warning("OpenCV unavailable -- applying linear stretch as CLAHE fallback.")
            return self._linear_stretch(frame)

    def _sobel_enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute Sobel edge magnitude and blend it additively with the frame.

        Blend formula:
            enhanced = clip(frame * 0.85 + edges * 0.15)
        This sharpens structural boundaries without destroying photometry.
        """
        edges = self._compute_sobel_magnitude(frame)
        blended = frame.astype(np.float32) * 0.85 + edges.astype(np.float32) * 0.15
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _compute_sobel_magnitude(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute the Sobel gradient magnitude of a grayscale frame.

        Uses OpenCV's Sobel operator if available, otherwise uses scipy.ndimage.
        """
        if _HAS_CV2:
            f32 = frame.astype(np.float32)
            gx  = cv2.Sobel(f32, cv2.CV_32F, 1, 0, ksize=self.sobel_ksize)
            gy  = cv2.Sobel(f32, cv2.CV_32F, 0, 1, ksize=self.sobel_ksize)
            mag = np.sqrt(gx ** 2 + gy ** 2)
        elif _HAS_SCIPY:
            f32 = frame.astype(np.float32)
            gx  = scipy_sobel(f32, axis=1)
            gy  = scipy_sobel(f32, axis=0)
            mag = np.sqrt(gx ** 2 + gy ** 2)
        else:
            logger.warning("No Sobel backend -- returning zero edge map.")
            return np.zeros_like(frame, dtype=np.uint8)

        # Normalise to [0, 255]
        mag_max = mag.max()
        if mag_max > 0:
            mag = (mag / mag_max * 255.0)
        return np.clip(mag, 0, 255).astype(np.uint8)

    def _linear_stretch(self, frame: np.ndarray) -> np.ndarray:
        """Linearly stretch pixel values to [0, 255]."""
        f = frame.astype(np.float32)
        f_min, f_max = f.min(), f.max()
        if f_max - f_min < 1e-6:
            return frame
        stretched = (f - f_min) / (f_max - f_min) * 255.0
        return np.clip(stretched, 0, 255).astype(np.uint8)
