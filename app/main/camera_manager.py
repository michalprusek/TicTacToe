"""
Camera manager compatibility module.
DEPRECATED: Functionality moved to camera_controller.py
"""

# Re-export for backward compatibility
from app.main.camera_controller import CameraController as CameraManager

__all__ = ['CameraManager']