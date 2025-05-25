"""
Camera manager module for the TicTacToe application.
"""
import logging
import time
from typing import Tuple, Optional

import numpy as np
import cv2


class CameraManager:
    """Manages camera operations for the TicTacToe application."""

    def __init__(self, camera_index=0, config=None, logger=None):
        """Initialize the camera manager.
        
        Args:
            camera_index: Index of the camera to use
            config: Configuration object
            logger: Logger instance
        """
        self.camera_index = camera_index
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Camera settings
        self.disable_autofocus = getattr(config, 'disable_autofocus', True)
        self.cap = None
        self.frame_width = getattr(config, 'frame_width', 640)
        self.frame_height = getattr(config, 'frame_height', 480)
        
        # Initialize camera
        self.setup_camera()

    def setup_camera(self) -> bool:
        """Sets up the camera with the specified settings.
        
        Returns:
            True if camera setup was successful, False otherwise
        """
        try:
            # Initialize camera capture
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                self.logger.error("Failed to open camera %s", self.camera_index)
                return False
                
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            # Disable autofocus if requested
            if self.disable_autofocus:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                autofocus_status = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
                self.logger.info("Autofocus status: %s", autofocus_status)
                
            # Log camera properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.logger.info("Camera initialized with resolution: %sx%s", 
                            actual_width, actual_height)
                
            return True
        except Exception as e:
            self.logger.error("Error setting up camera: %s", e)
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Camera is not initialized")
            return False, None
            
        # Read frame from camera
        ret, frame = self.cap.read()
        
        if not ret:
            self.logger.warning("Failed to read frame from camera")
            return False, None
            
        return True, frame

    def release(self) -> None:
        """Releases the camera resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.logger.info("Camera released")

    def is_opened(self) -> bool:
        """Checks if the camera is opened.
        
        Returns:
            True if the camera is opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()

    def get_camera_properties(self) -> dict:
        """Gets the camera properties.
        
        Returns:
            Dictionary of camera properties
        """
        if not self.cap or not self.cap.isOpened():
            return {}
            
        properties = {
            'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'autofocus': self.cap.get(cv2.CAP_PROP_AUTOFOCUS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
            'hue': self.cap.get(cv2.CAP_PROP_HUE),
            'gain': self.cap.get(cv2.CAP_PROP_GAIN),
            'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE)
        }
        
        return properties

    def set_camera_property(self, property_id: int, value: float) -> bool:
        """Sets a camera property.
        
        Args:
            property_id: OpenCV property ID
            value: Property value
            
        Returns:
            True if property was set successfully, False otherwise
        """
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Camera is not initialized")
            return False
            
        result = self.cap.set(property_id, value)
        
        if result:
            self.logger.info("Set camera property %s to %s", property_id, value)
        else:
            self.logger.warning("Failed to set camera property %s to %s", property_id, value)
            
        return result
