"""Image loading and validation utilities.

Menyediakan pembacaan gambar dari path atau array dengan validasi dasar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np


ImageInput = Union[str, Path, np.ndarray]


class ImageLoadError(ValueError):
    """Kesalahan saat memuat atau memvalidasi gambar."""


class ImageLoader:
    """Loader gambar dengan validasi format dan ukuran."""

    SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    @staticmethod
    def load(
        source: ImageInput,
        read_mode: int = cv2.IMREAD_COLOR,
        min_size: int = 1,
    ) -> Dict[str, Any]:
        """Muat gambar dari path atau ndarray BGR.

        Args:
            source: Path gambar atau numpy array (diasumsikan BGR untuk color image).
            read_mode: Mode baca OpenCV (default BGR).
            min_size: Ukuran minimum sisi gambar (piksel).

        Returns:
            Dict hasil:
            {
              "image": np.ndarray,
              "path": str|None,
              "shape": [h, w, c],
              "dtype": str,
              "source_type": "path"|"ndarray",
            }

        Raises:
            ImageLoadError: jika input tidak valid atau gagal dibaca.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise ImageLoadError(f"File gambar tidak ditemukan: {path}")
            if not path.is_file():
                raise ImageLoadError(f"Path bukan file: {path}")
            if path.suffix.lower() not in ImageLoader.SUPPORTED_SUFFIXES:
                # allow but warn-like behavior not possible; keep permissive read attempt
                pass
            image = cv2.imread(str(path), read_mode)
            if image is None:
                raise ImageLoadError(f"Gagal membaca gambar: {path}")
            src_path: Optional[str] = str(path)
            source_type = "path"
        elif isinstance(source, np.ndarray):
            image = source
            src_path = None
            source_type = "ndarray"
        else:
            raise ImageLoadError(
                f"Tipe source tidak didukung: {type(source).__name__}. Gunakan path atau numpy.ndarray."
            )

        if not isinstance(image, np.ndarray):
            raise ImageLoadError("Hasil baca bukan numpy.ndarray")
        if image.size == 0:
            raise ImageLoadError("Gambar kosong (size=0)")
        if image.ndim not in (2, 3):
            raise ImageLoadError(f"Dimensi gambar tidak didukung: {image.ndim}")

        h, w = int(image.shape[0]), int(image.shape[1])
        if h < min_size or w < min_size:
            raise ImageLoadError(f"Ukuran gambar terlalu kecil: {w}x{h}")

        if image.ndim == 2:
            shape = [h, w, 1]
        else:
            shape = [h, w, int(image.shape[2])]

        return {
            "image": image,
            "path": src_path,
            "shape": shape,
            "dtype": str(image.dtype),
            "source_type": source_type,
        }

    @staticmethod
    def ensure_bgr(image: np.ndarray) -> np.ndarray:
        """Pastikan image dalam format BGR 3-channel."""
        if not isinstance(image, np.ndarray):
            raise ImageLoadError("image harus numpy.ndarray")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ImageLoadError(f"Format channel tidak didukung: shape={image.shape}")
