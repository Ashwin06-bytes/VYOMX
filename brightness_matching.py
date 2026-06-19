"""
brightness_matching.py -- Module 1
==================================
Brightness / Histogram Normalisation for VYOMX Pipeline.

Ensures that Frame A and Frame B share the same radiometric baseline
before interpolation, eliminating illumination jumps caused by sensor
drift, scan-angle variations, or different acquisition conditions.

Technique:
  - scikit-image match_histograms() maps the grey-level distribution
    of the source frame onto the reference frame, channel by channel.
  - NaN / fill pixels are masked out before matching and restored after.

Outputs:
  - brightness-corrected frame as np.ndarray (uint8)
  - saved .npy and .png artefacts in data/brightness_corrected/
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports -- graceful degradation
try:
    from skimage.exposure import match_histograms
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    logger.warning("scikit-image not installed -- histogram matching unavailable.")

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class BrightnessMatching:
    """
    Radiometric normalisation via histogram matching.

    Parameters
    ----------
    config : dict
        Sub-dictionary from config.yaml['brightness_matching'].
    output_dir : str | Path
        Directory where corrected frames are saved.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        output_dir: Union[str, Path] = "data/brightness_corrected",
    ) -> None:
        self.config     = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.method    : str  = self.config.get("method", "histogram")
        self.reference : str  = self.config.get("reference_frame", "B")
        self.save_png  : bool = self.config.get("save_png", True)
        self.save_npy  : bool = self.config.get("save_npy", True)

        logger.info(
            "BrightnessMatching initialised | method=%s | reference=%s",
            self.method, self.reference,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        prefix: str = "frame",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalise Frame A to match the histogram of Frame B.

        Parameters
        ----------
        frame_a : np.ndarray (H, W) or (H, W, C) -- source frame.
        frame_b : np.ndarray (H, W) or (H, W, C) -- reference frame.
        prefix  : Output filename prefix.

        Returns
        -------
        matched_a : np.ndarray -- histogram-matched version of Frame A.
        frame_b   : np.ndarray -- Frame B (unchanged, returned for symmetry).
        """
        logger.info(
            "Brightness matching | frame_a shape=%s | frame_b shape=%s",
            frame_a.shape, frame_b.shape,
        )

        if frame_a.shape != frame_b.shape:
            logger.warning(
                "Frame shape mismatch: %s vs %s. Automatically resizing Frame B to match Frame A.",
                frame_a.shape, frame_b.shape
            )
            if _HAS_CV2:
                # cv2.resize expects (width, height)
                frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]), interpolation=cv2.INTER_LINEAR)
            elif _HAS_PIL:
                img_b = PILImage.fromarray(frame_b)
                img_b = img_b.resize((frame_a.shape[1], frame_a.shape[0]), PILImage.Resampling.BILINEAR)
                frame_b = np.array(img_b)
            else:
                raise ValueError(
                    f"Frame shape mismatch: {frame_a.shape} vs {frame_b.shape} and no resizing library available. "
                    "Please install opencv-python or pillow, or ensure frames have identical dimensions."
                )

        if self.method == "histogram":
            matched_a = self._histogram_match(frame_a, frame_b)
        elif self.method == "linear":
            matched_a = self._linear_match(frame_a, frame_b)
        elif self.method == "none":
            logger.info("Brightness matching disabled (method='none').")
            matched_a = frame_a.copy()
        else:
            logger.warning(
                "Unknown method '%s', falling back to histogram.", self.method
            )
            matched_a = self._histogram_match(frame_a, frame_b)

        # Save artefacts
        if self.save_npy:
            npy_path = self.output_dir / f"{prefix}_matched_a.npy"
            np.save(str(npy_path), matched_a)
            logger.debug("Saved matched npy -> %s", npy_path)

        if self.save_png:
            self._save_png(matched_a, self.output_dir / f"{prefix}_matched_a.png")
            self._save_png(frame_b,   self.output_dir / f"{prefix}_frame_b_ref.png")

        return matched_a, frame_b

    def compute_statistics(
        self,
        frame: np.ndarray,
        label: str = "frame",
    ) -> Dict:
        """
        Compute basic radiometric statistics of a frame.

        Parameters
        ----------
        frame : np.ndarray
        label : Human-readable label for logging.

        Returns
        -------
        dict : {mean, std, min, max, median}
        """
        stats = {
            "label":  label,
            "mean":   float(np.mean(frame)),
            "std":    float(np.std(frame)),
            "min":    float(np.min(frame)),
            "max":    float(np.max(frame)),
            "median": float(np.median(frame)),
        }
        logger.info(
            "Stats [%s] mean=%.2f std=%.2f min=%.2f max=%.2f",
            label, stats["mean"], stats["std"], stats["min"], stats["max"],
        )
        return stats

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _histogram_match(
        self,
        source: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Apply scikit-image histogram matching, handling NaN / fill values.

        NaN pixels in the source are masked, matching is done on valid pixels,
        then NaN pixels are restored to 0.
        """
        if not _HAS_SKIMAGE:
            logger.warning(
                "scikit-image unavailable -- using linear stretch as fallback."
            )
            return self._linear_match(source, reference)

        source_f    = source.astype(np.float64)
        reference_f = reference.astype(np.float64)

        # Build validity masks
        valid_src = np.isfinite(source_f)
        valid_ref = np.isfinite(reference_f)

        if not np.any(valid_src):
            logger.warning("Source frame has no valid pixels -- returning zeros.")
            return np.zeros_like(source, dtype=np.uint8)

        # Perform histogram matching on the full arrays;
        # scikit-image handles multichannel automatically via channel_axis param
        if source_f.ndim == 2:
            matched = match_histograms(
                np.where(valid_src, source_f, 0.0),
                np.where(valid_ref, reference_f, 0.0),
            )
        else:
            # (H, W, C) -- match per channel
            matched = match_histograms(
                np.where(valid_src[..., None], source_f, 0.0),
                np.where(valid_ref[..., None], reference_f, 0.0),
                channel_axis=-1,
            )

        # Restore original invalid locations to 0
        matched = np.where(valid_src, matched, 0.0)
        return np.clip(matched, 0, 255).astype(np.uint8)

    def _linear_match(
        self,
        source: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Linear radiometric matching:
        matched = (source - src_mean) / src_std * ref_std + ref_mean
        """
        src_f = source.astype(np.float64)
        ref_f = reference.astype(np.float64)

        src_mean, src_std = np.nanmean(src_f), np.nanstd(src_f)
        ref_mean, ref_std = np.nanmean(ref_f), np.nanstd(ref_f)

        if src_std < 1e-6:
            logger.warning("Source frame std ~= 0 -- returning reference frame copy.")
            return reference.copy()

        matched = (src_f - src_mean) / src_std * ref_std + ref_mean
        return np.clip(matched, 0, 255).astype(np.uint8)

    def _save_png(self, array: np.ndarray, path: Path) -> None:
        """Save uint8 numpy array as a greyscale PNG."""
        if not _HAS_PIL:
            logger.debug("PIL unavailable -- skipping PNG save for %s", path.name)
            return
        try:
            img = PILImage.fromarray(array.astype(np.uint8), mode="L")
            img.save(str(path))
            logger.debug("Saved PNG -> %s", path.name)
        except Exception as exc:
            logger.error("Failed to save PNG %s: %s", path.name, exc)
