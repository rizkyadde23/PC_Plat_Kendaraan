"""Modul deteksi plat nomor (YOLO baseline).

Menyediakan wrapper inferensi reusable untuk model deteksi plat.
"""

from .plate_detector import PlateDetector

__all__ = ["PlateDetector"]
