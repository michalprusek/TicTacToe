"""Comprehensive tests for frame_utils.py module."""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

from app.main.frame_utils import FrameConverter


class TestFrameConverterBgrToRgb:
    """Test BGR to RGB conversion."""
    
    def test_bgr_to_rgb_conversion(self):
        """Test BGR to RGB color conversion."""
        # Create a BGR frame (blue channel=255, green=128, red=64)
        bgr_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = 255  # Blue channel
        bgr_frame[:, :, 1] = 128  # Green channel  
        bgr_frame[:, :, 2] = 64   # Red channel
        
        with patch('cv2.cvtColor') as mock_cvtcolor:
            # Mock the conversion result
            mock_rgb_frame = np.zeros((10, 10, 3), dtype=np.uint8)
            mock_rgb_frame[:, :, 0] = 64   # Red channel (was blue)
            mock_rgb_frame[:, :, 1] = 128  # Green channel (same)
            mock_rgb_frame[:, :, 2] = 255  # Blue channel (was red)
            mock_cvtcolor.return_value = mock_rgb_frame
            
            result = FrameConverter.bgr_to_rgb(bgr_frame)
            
            # Verify cv2.cvtColor was called with correct parameters
            mock_cvtcolor.assert_called_once_with(bgr_frame, cv2.COLOR_BGR2RGB)
            assert np.array_equal(result, mock_rgb_frame)
    
    def test_bgr_to_rgb_preserves_shape(self):
        """Test that BGR to RGB conversion preserves frame shape."""
        # Test different frame sizes
        shapes = [(100, 100, 3), (480, 640, 3), (720, 1280, 3)]
        
        for shape in shapes:
            bgr_frame = np.random.randint(0, 255, shape, dtype=np.uint8)
            
            with patch('cv2.cvtColor', return_value=bgr_frame) as mock_cvtcolor:
                result = FrameConverter.bgr_to_rgb(bgr_frame)
                
                assert result.shape == shape
                mock_cvtcolor.assert_called_once_with(bgr_frame, cv2.COLOR_BGR2RGB)


class TestFrameConverterToQImage:
    """Test frame to QImage conversion."""
    
    @patch('app.main.frame_utils.FrameConverter.bgr_to_rgb')
    @patch('app.main.frame_utils.QImage')
    def test_frame_to_qimage_basic(self, mock_qimage, mock_bgr_to_rgb):
        """Test basic frame to QImage conversion."""
        # Create test frame
        test_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        mock_rgb_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # Mock the BGR to RGB conversion
        mock_bgr_to_rgb.return_value = mock_rgb_frame
        
        # Mock QImage constructor
        mock_qimage_instance = Mock()
        mock_qimage.return_value = mock_qimage_instance
        mock_qimage.Format_RGB888 = "RGB888_FORMAT"
        
        result = FrameConverter.frame_to_qimage(test_frame)
        
        # Verify BGR to RGB conversion was called
        mock_bgr_to_rgb.assert_called_once_with(test_frame)
        
        # Verify QImage was constructed with correct parameters
        h, w, ch = test_frame.shape
        bytes_per_line = ch * w
        mock_qimage.assert_called_once_with(
            mock_rgb_frame.data,
            w, h,
            bytes_per_line,
            "RGB888_FORMAT"
        )
        
        assert result == mock_qimage_instance
    
    @patch('app.main.frame_utils.FrameConverter.bgr_to_rgb')
    @patch('app.main.frame_utils.QImage')
    def test_frame_to_qimage_different_sizes(self, mock_qimage, mock_bgr_to_rgb):
        """Test frame to QImage conversion with different frame sizes."""
        sizes = [(50, 50, 3), (480, 640, 3), (720, 1280, 3)]
        
        for h, w, ch in sizes:
            test_frame = np.random.randint(0, 255, (h, w, ch), dtype=np.uint8)
            mock_rgb_frame = np.random.randint(0, 255, (h, w, ch), dtype=np.uint8)
            
            mock_bgr_to_rgb.return_value = mock_rgb_frame
            mock_qimage.reset_mock()
            mock_qimage.Format_RGB888 = "RGB888_FORMAT"
            
            FrameConverter.frame_to_qimage(test_frame)
            
            # Check correct dimensions were passed
            bytes_per_line = ch * w
            mock_qimage.assert_called_once_with(
                mock_rgb_frame.data,
                w, h,
                bytes_per_line,
                "RGB888_FORMAT"
            )


class TestFrameConverterToPixmap:
    """Test frame to QPixmap conversion."""
    
    @patch('app.main.frame_utils.FrameConverter.frame_to_qimage')
    @patch('app.main.frame_utils.QPixmap')
    @patch('app.main.frame_utils.Qt')
    def test_frame_to_pixmap_keep_aspect(self, mock_qt, mock_qpixmap, mock_frame_to_qimage):
        """Test frame to QPixmap conversion with aspect ratio preservation."""
        test_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        mock_qimage = Mock()
        mock_frame_to_qimage.return_value = mock_qimage
        
        # Mock QPixmap and its methods
        mock_pixmap_instance = Mock()
        mock_scaled_pixmap = Mock()
        mock_pixmap_instance.scaled.return_value = mock_scaled_pixmap
        mock_qpixmap.fromImage.return_value = mock_pixmap_instance
        
        # Mock Qt constants
        mock_qt.KeepAspectRatio = "KEEP_ASPECT"
        mock_qt.SmoothTransformation = "SMOOTH"
        
        result = FrameConverter.frame_to_pixmap(test_frame, 300, 150, keep_aspect=True)
        
        # Verify the chain of calls
        mock_frame_to_qimage.assert_called_once_with(test_frame)
        mock_qpixmap.fromImage.assert_called_once_with(mock_qimage)
        mock_pixmap_instance.scaled.assert_called_once_with(
            300, 150, "KEEP_ASPECT", "SMOOTH"
        )
        
        assert result == mock_scaled_pixmap
    
    @patch('app.main.frame_utils.FrameConverter.frame_to_qimage')
    @patch('app.main.frame_utils.QPixmap')
    def test_frame_to_pixmap_no_aspect(self, mock_qpixmap, mock_frame_to_qimage):
        """Test frame to QPixmap conversion without aspect ratio preservation."""
        test_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        mock_qimage = Mock()
        mock_frame_to_qimage.return_value = mock_qimage
        
        # Mock QPixmap and its methods
        mock_pixmap_instance = Mock()
        mock_scaled_pixmap = Mock()
        mock_pixmap_instance.scaled.return_value = mock_scaled_pixmap
        mock_qpixmap.fromImage.return_value = mock_pixmap_instance
        
        result = FrameConverter.frame_to_pixmap(test_frame, 400, 300, keep_aspect=False)
        
        # Verify the calls
        mock_frame_to_qimage.assert_called_once_with(test_frame)
        mock_qpixmap.fromImage.assert_called_once_with(mock_qimage)
        mock_pixmap_instance.scaled.assert_called_once_with(400, 300)
        
        assert result == mock_scaled_pixmap
    
    @patch('app.main.frame_utils.FrameConverter.frame_to_qimage')
    @patch('app.main.frame_utils.QPixmap')
    @patch('app.main.frame_utils.Qt')
    def test_frame_to_pixmap_default_keep_aspect(self, mock_qt, mock_qpixmap, mock_frame_to_qimage):
        """Test frame to QPixmap conversion with default keep_aspect=True."""
        test_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        mock_qimage = Mock()
        mock_frame_to_qimage.return_value = mock_qimage
        
        mock_pixmap_instance = Mock()
        mock_scaled_pixmap = Mock()
        mock_pixmap_instance.scaled.return_value = mock_scaled_pixmap
        mock_qpixmap.fromImage.return_value = mock_pixmap_instance
        
        mock_qt.KeepAspectRatio = "KEEP_ASPECT"
        mock_qt.SmoothTransformation = "SMOOTH"
        
        # Call without keep_aspect parameter (should default to True)
        result = FrameConverter.frame_to_pixmap(test_frame, 200, 100)
        
        # Should use keep aspect ratio
        mock_pixmap_instance.scaled.assert_called_once_with(
            200, 100, "KEEP_ASPECT", "SMOOTH"
        )
        assert result == mock_scaled_pixmap


class TestFrameConverterResizeFrame:
    """Test frame resizing functionality."""
    
    def test_resize_frame_scale_down(self):
        """Test frame resizing with scale down."""
        # Create test frame
        test_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        with patch('cv2.resize') as mock_resize:
            mock_resized = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
            mock_resize.return_value = mock_resized
            
            result = FrameConverter.resize_frame(test_frame, 0.5)
            
            # Verify cv2.resize was called with correct parameters
            mock_resize.assert_called_once_with(test_frame, (100, 50))  # (new_width, new_height)
            assert np.array_equal(result, mock_resized)
    
    def test_resize_frame_scale_up(self):
        """Test frame resizing with scale up."""
        test_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        with patch('cv2.resize') as mock_resize:
            mock_resized = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
            mock_resize.return_value = mock_resized
            
            result = FrameConverter.resize_frame(test_frame, 2.0)
            
            # 100*2=200 height, 200*2=400 width
            mock_resize.assert_called_once_with(test_frame, (400, 200))
            assert np.array_equal(result, mock_resized)
    
    def test_resize_frame_no_scaling(self):
        """Test frame resizing with scale factor 1.0 (no change)."""
        test_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        with patch('cv2.resize') as mock_resize:
            mock_resize.return_value = test_frame
            
            result = FrameConverter.resize_frame(test_frame, 1.0)
            
            # Should still call resize but with same dimensions
            mock_resize.assert_called_once_with(test_frame, (200, 100))
            assert np.array_equal(result, test_frame)
    
    def test_resize_frame_fractional_scaling(self):
        """Test frame resizing with fractional scaling."""
        test_frame = np.random.randint(0, 255, (150, 300, 3), dtype=np.uint8)
        
        with patch('cv2.resize') as mock_resize:
            mock_resized = np.random.randint(0, 255, (120, 240, 3), dtype=np.uint8)
            mock_resize.return_value = mock_resized
            
            result = FrameConverter.resize_frame(test_frame, 0.8)
            
            # 150*0.8=120, 300*0.8=240 
            mock_resize.assert_called_once_with(test_frame, (240, 120))
            assert np.array_equal(result, mock_resized)
    
    def test_resize_frame_different_shapes(self):
        """Test frame resizing with different input shapes."""
        shapes_and_scales = [
            ((50, 100, 3), 0.5),
            ((480, 640, 3), 0.25),
            ((720, 1280, 3), 1.5)
        ]
        
        for shape, scale in shapes_and_scales:
            test_frame = np.random.randint(0, 255, shape, dtype=np.uint8)
            
            with patch('cv2.resize') as mock_resize:
                height, width = shape[:2]
                expected_new_height = int(height * scale)
                expected_new_width = int(width * scale)
                
                mock_resized = np.random.randint(
                    0, 255, (expected_new_height, expected_new_width, 3), dtype=np.uint8
                )
                mock_resize.return_value = mock_resized
                
                result = FrameConverter.resize_frame(test_frame, scale)
                
                mock_resize.assert_called_once_with(
                    test_frame, (expected_new_width, expected_new_height)
                )
                assert np.array_equal(result, mock_resized)


class TestFrameConverterIntegration:
    """Test integration scenarios and edge cases."""
    
    def test_frame_processing_pipeline(self):
        """Test a complete frame processing pipeline."""
        # Create a test frame
        original_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        with patch('cv2.cvtColor') as mock_cvtcolor, \
             patch('cv2.resize') as mock_resize, \
             patch('app.main.frame_utils.QImage') as mock_qimage, \
             patch('app.main.frame_utils.QPixmap') as mock_qpixmap:
            
            # Mock the entire pipeline
            rgb_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            resized_frame = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
            mock_cvtcolor.return_value = rgb_frame
            mock_resize.return_value = resized_frame
            
            # Step 1: Resize
            step1 = FrameConverter.resize_frame(original_frame, 0.5)
            
            # Step 2: Convert to RGB
            step2 = FrameConverter.bgr_to_rgb(step1)
            
            # Verify the pipeline
            mock_resize.assert_called_once_with(original_frame, (100, 50))
            mock_cvtcolor.assert_called_once_with(step1, cv2.COLOR_BGR2RGB)
    
    def test_error_handling_in_conversions(self):
        """Test error handling in frame conversions."""
        # Test with invalid frame (wrong dimensions)
        invalid_frame = np.array([[1, 2], [3, 4]])  # 2D array instead of 3D
        
        # The methods should handle this gracefully or raise appropriate errors
        # This tests that the methods don't crash unexpectedly
        try:
            # These might raise exceptions, which is acceptable
            FrameConverter.bgr_to_rgb(invalid_frame)
        except Exception as e:
            # Exception is acceptable for invalid input - cv2.error is also valid
            import cv2
            assert isinstance(e, (ValueError, AttributeError, IndexError, cv2.error))


if __name__ == "__main__":
    pytest.main([__file__])
