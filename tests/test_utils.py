"""
Tests for utils module.
"""
import pytest
import time
from unittest.mock import patch

from app.core.utils import FPSCalculator


class TestFPSCalculator:
    """Test class for FPSCalculator."""
    
    def test_initialization_default(self):
        """Test FPS calculator initialization with default buffer size."""
        calc = FPSCalculator()
        assert calc.buffer_size == 10
        assert calc.get_fps() == 0.0
    
    def test_initialization_custom_buffer(self):
        """Test FPS calculator initialization with custom buffer size."""
        calc = FPSCalculator(buffer_size=5)
        assert calc.buffer_size == 5
        assert calc.get_fps() == 0.0
    
    def test_initialization_invalid_buffer(self):
        """Test FPS calculator initialization with invalid buffer size."""
        with pytest.raises(ValueError, match="Buffer size must be a positive integer"):
            FPSCalculator(buffer_size=0)
        
        with pytest.raises(ValueError, match="Buffer size must be a positive integer"):
            FPSCalculator(buffer_size=-1)
    
    def test_tick_single(self):
        """Test single tick (no FPS calculated yet)."""
        calc = FPSCalculator()
        calc.tick()
        assert calc.get_fps() == 0.0  # Need at least 2 ticks for duration    
    @patch('time.perf_counter')
    def test_tick_multiple(self, mock_time):
        """Test multiple ticks with mocked time."""
        calc = FPSCalculator()
        
        # Mock time sequence: 0.0, 0.1, 0.2 (0.1 second intervals = 10 FPS)
        mock_time.side_effect = [0.0, 0.1, 0.2]
        
        calc.tick()  # First tick at 0.0
        calc.tick()  # Second tick at 0.1 (duration = 0.1)
        calc.tick()  # Third tick at 0.2 (duration = 0.1)
        
        fps = calc.get_fps()
        assert abs(fps - 10.0) < 0.001  # Should be 10 FPS
    
    @patch('time.perf_counter')
    def test_variable_fps(self, mock_time):
        """Test FPS calculation with variable intervals."""
        calc = FPSCalculator()
        
        # Different intervals: 0.05s (20 FPS), 0.1s (10 FPS) -> average ~13.33 FPS
        mock_time.side_effect = [0.0, 0.05, 0.15]
        
        calc.tick()  # First tick at 0.0
        calc.tick()  # Second tick at 0.05 (duration = 0.05)
        calc.tick()  # Third tick at 0.15 (duration = 0.1)
        
        fps = calc.get_fps()
        expected_avg_duration = (0.05 + 0.1) / 2  # 0.075
        expected_fps = 1.0 / expected_avg_duration  # ~13.33
        assert abs(fps - expected_fps) < 0.001    
    @patch('time.perf_counter')
    def test_buffer_overflow(self, mock_time):
        """Test buffer overflow handling."""
        calc = FPSCalculator(buffer_size=2)
        
        # Three intervals, but buffer only holds 2
        mock_time.side_effect = [0.0, 0.1, 0.2, 0.4]
        
        calc.tick()  # First tick at 0.0
        calc.tick()  # Second tick at 0.1 (duration = 0.1)
        calc.tick()  # Third tick at 0.2 (duration = 0.1)
        calc.tick()  # Fourth tick at 0.4 (duration = 0.2)
        
        # Buffer should contain only last 2 durations: [0.1, 0.2]
        fps = calc.get_fps()
        expected_avg_duration = (0.1 + 0.2) / 2  # 0.15
        expected_fps = 1.0 / expected_avg_duration  # ~6.67
        assert abs(fps - expected_fps) < 0.001
    
    @patch('time.perf_counter')
    def test_zero_duration_handling(self, mock_time):
        """Test handling of zero duration between ticks."""
        calc = FPSCalculator()
        
        # Same timestamp for consecutive ticks
        mock_time.side_effect = [0.0, 0.0, 0.1]
        
        calc.tick()  # First tick at 0.0
        calc.tick()  # Second tick at 0.0 (duration = 0.0, should be ignored)
        calc.tick()  # Third tick at 0.1 (duration = 0.1)
        
        fps = calc.get_fps()
        # Should only have one valid duration (0.1), so FPS = 10
        assert abs(fps - 10.0) < 0.001    
    def test_reset(self):
        """Test resetting the FPS calculator."""
        calc = FPSCalculator()
        
        # Add some ticks
        calc.tick()
        calc.tick()
        
        # Reset and check
        calc.reset()
        assert calc.get_fps() == 0.0
        assert calc._last_tick_time == 0.0
        assert len(calc._timestamps) == 0
    
    def test_real_time_calculation(self):
        """Test FPS calculation with real time (integration test)."""
        calc = FPSCalculator()
        
        calc.tick()
        time.sleep(0.01)  # 10ms delay
        calc.tick()
        time.sleep(0.01)  # 10ms delay
        calc.tick()
        
        fps = calc.get_fps()
        # Should be approximately 100 FPS (0.01s intervals)
        # Allow some tolerance for timing variations
        assert 80 < fps < 120  # Reasonable range around 100 FPS