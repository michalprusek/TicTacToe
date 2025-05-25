"""
Comprehensive tests for utils module.
"""
import unittest
import time
from unittest.mock import patch

from app.core.utils import FPSCalculator


class TestFPSCalculator(unittest.TestCase):
    
    def setUp(self):
        """Set up a fresh FPSCalculator for each test."""
        self.fps_calc = FPSCalculator()
    
    def test_init_default(self):
        """Test default initialization."""
        calc = FPSCalculator()
        self.assertEqual(calc.buffer_size, 10)
        self.assertEqual(len(calc._timestamps), 0)
        self.assertEqual(calc._last_tick_time, 0.0)
    
    def test_init_custom_buffer_size(self):
        """Test initialization with custom buffer size."""
        calc = FPSCalculator(buffer_size=5)
        self.assertEqual(calc.buffer_size, 5)
        self.assertEqual(len(calc._timestamps), 0)
        self.assertEqual(calc._last_tick_time, 0.0)
    
    def test_init_invalid_buffer_size(self):
        """Test initialization with invalid buffer size."""
        with self.assertRaises(ValueError):
            FPSCalculator(buffer_size=0)
        
        with self.assertRaises(ValueError):
            FPSCalculator(buffer_size=-1)
    
    def test_get_fps_no_ticks(self):
        """Test get_fps with no ticks recorded."""
        self.assertEqual(self.fps_calc.get_fps(), 0.0)
    
    @patch('time.perf_counter')
    def test_tick_single(self, mock_time):
        """Test single tick recording."""
        mock_time.return_value = 1.0
        self.fps_calc.tick()
        
        # First tick should not add any duration
        self.assertEqual(len(self.fps_calc._timestamps), 0)
        self.assertEqual(self.fps_calc._last_tick_time, 1.0)
    
    @patch('time.perf_counter')
    def test_tick_multiple(self, mock_time):
        """Test multiple ticks with proper timing."""
        # First tick at time 1.0
        mock_time.return_value = 1.0
        self.fps_calc.tick()
        
        # Second tick at time 2.0 (1 second difference)
        mock_time.return_value = 2.0
        self.fps_calc.tick()
        
        # Should now have one duration recorded
        self.assertEqual(len(self.fps_calc._timestamps), 1)
        self.assertEqual(self.fps_calc._timestamps[0], 1.0)
        
        # Third tick at time 3.0 (another 1 second difference)
        mock_time.return_value = 3.0
        self.fps_calc.tick()
        
        # Should now have two durations
        self.assertEqual(len(self.fps_calc._timestamps), 2)
        self.assertEqual(self.fps_calc._timestamps[1], 1.0)
    
    @patch('time.perf_counter')
    def test_get_fps_calculation(self, mock_time):
        """Test FPS calculation with known timing."""
        # Create ticks at 1 second intervals (should give 1 FPS)
        mock_time.return_value = 1.0
        self.fps_calc.tick()
        
        mock_time.return_value = 2.0
        self.fps_calc.tick()
        
        fps = self.fps_calc.get_fps()
        self.assertEqual(fps, 1.0)  # 1 second duration = 1 FPS
        
        # Add another tick with 0.5 second interval
        mock_time.return_value = 2.5
        self.fps_calc.tick()
        
        fps = self.fps_calc.get_fps()
        # Average duration: (1.0 + 0.5) / 2 = 0.75, FPS = 1/0.75 = 1.333...
        self.assertAlmostEqual(fps, 1.333333333333333, places=6)
    
    def test_reset(self):
        """Test reset functionality."""
        # Add some data first
        self.fps_calc._timestamps.append(1.0)
        self.fps_calc._last_tick_time = 5.0
        
        # Reset
        self.fps_calc.reset()
        
        # Verify reset
        self.assertEqual(len(self.fps_calc._timestamps), 0)
        self.assertEqual(self.fps_calc._last_tick_time, 0.0)


if __name__ == '__main__':
    unittest.main()