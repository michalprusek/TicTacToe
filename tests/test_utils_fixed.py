# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Fixed tests for utils module based on actual API.
Tests FPSCalculator functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch
from app.core.utils import FPSCalculator


class TestFPSCalculator:
    """Test FPSCalculator functionality."""

    def test_fps_calculator_initialization_default(self):
        """Test FPS calculator initialization with default buffer size."""
        fps_calc = FPSCalculator()
        assert fps_calc.buffer_size == 10
        assert len(fps_calc._timestamps) == 0
        assert fps_calc._last_tick_time == 0.0

    def test_fps_calculator_initialization_custom(self):
        """Test FPS calculator initialization with custom buffer size."""
        fps_calc = FPSCalculator(buffer_size=5)
        assert fps_calc.buffer_size == 5
        assert len(fps_calc._timestamps) == 0

    def test_fps_calculator_initialization_invalid_buffer_size(self):
        """Test FPS calculator initialization with invalid buffer size."""
        with pytest.raises(ValueError, match="Buffer size must be a positive integer"):
            FPSCalculator(buffer_size=0)
        
        with pytest.raises(ValueError, match="Buffer size must be a positive integer"):
            FPSCalculator(buffer_size=-1)

    def test_fps_calculator_tick_first_time(self):
        """Test first tick (no duration recorded yet)."""
        fps_calc = FPSCalculator()
        
        fps_calc.tick()
        assert fps_calc._last_tick_time > 0
        assert len(fps_calc._timestamps) == 0  # No duration recorded on first tick

    def test_fps_calculator_tick_multiple_times(self):
        """Test multiple ticks record durations."""
        fps_calc = FPSCalculator()
        
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.2]):
            fps_calc.tick()  # First tick at 1.0
            fps_calc.tick()  # Second tick at 1.1 (duration: 0.1)
            fps_calc.tick()  # Third tick at 1.2 (duration: 0.1)
        
        assert len(fps_calc._timestamps) == 2
        assert abs(fps_calc._timestamps[0] - 0.1) < 1e-10
        assert abs(fps_calc._timestamps[1] - 0.1) < 1e-10

    def test_fps_calculator_get_fps_no_data(self):
        """Test FPS calculation with no data."""
        fps_calc = FPSCalculator()
        
        fps = fps_calc.get_fps()
        assert fps == 0.0

    def test_fps_calculator_get_fps_with_data(self):
        """Test FPS calculation with recorded durations."""
        fps_calc = FPSCalculator()
        
        # Simulate consistent 0.1 second intervals (10 FPS)
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.2, 1.3]):
            fps_calc.tick()
            fps_calc.tick()
            fps_calc.tick()
            fps_calc.tick()
        
        fps = fps_calc.get_fps()
        assert abs(fps - 10.0) < 1e-10  # 1 / 0.1 = 10 FPS

    def test_fps_calculator_get_fps_variable_durations(self):
        """Test FPS calculation with variable durations."""
        fps_calc = FPSCalculator()
        
        # Simulate variable intervals: 0.1, 0.2, 0.1 seconds (average 0.133...)
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.3, 1.4]):
            fps_calc.tick()
            fps_calc.tick()  # Duration: 0.1
            fps_calc.tick()  # Duration: 0.2
            fps_calc.tick()  # Duration: 0.1
        
        fps = fps_calc.get_fps()
        expected_avg_duration = (0.1 + 0.2 + 0.1) / 3
        expected_fps = 1.0 / expected_avg_duration
        assert abs(fps - expected_fps) < 1e-10

    def test_fps_calculator_buffer_overflow(self):
        """Test FPS calculator with buffer overflow."""
        fps_calc = FPSCalculator(buffer_size=2)
        
        # Add more durations than buffer size
        times = [1.0, 1.1, 1.2, 1.3, 1.4]  # Will create 4 durations in buffer of size 2
        with patch('time.perf_counter', side_effect=times):
            for _ in times:
                fps_calc.tick()
        
        # Should only keep last 2 durations
        assert len(fps_calc._timestamps) == 2
        assert abs(fps_calc._timestamps[0] - 0.1) < 1e-10  # Second to last duration
        assert abs(fps_calc._timestamps[1] - 0.1) < 1e-10  # Last duration

    def test_fps_calculator_reset(self):
        """Test FPS calculator reset."""
        fps_calc = FPSCalculator()
        
        # Add some data
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.2]):
            fps_calc.tick()
            fps_calc.tick()
            fps_calc.tick()
        
        assert len(fps_calc._timestamps) > 0
        assert fps_calc._last_tick_time > 0
        
        # Reset
        fps_calc.reset()
        
        assert len(fps_calc._timestamps) == 0
        assert fps_calc._last_tick_time == 0.0

    def test_fps_calculator_zero_duration_handling(self):
        """Test handling of zero duration (simultaneous ticks)."""
        fps_calc = FPSCalculator()
        
        # Simulate simultaneous ticks (same time)
        with patch('time.perf_counter', side_effect=[1.0, 1.0, 1.1]):
            fps_calc.tick()
            fps_calc.tick()  # Same time, duration = 0, should not be recorded
            fps_calc.tick()  # Duration = 0.1, should be recorded
        
        assert len(fps_calc._timestamps) == 1
        assert abs(fps_calc._timestamps[0] - 0.1) < 1e-10

    def test_fps_calculator_real_timing(self):
        """Test FPS calculator with real timing (integration test)."""
        fps_calc = FPSCalculator()
        
        # Record a few ticks with small delays
        fps_calc.tick()
        time.sleep(0.01)  # 10ms
        fps_calc.tick()
        time.sleep(0.01)  # 10ms
        fps_calc.tick()
        
        fps = fps_calc.get_fps()
        
        # Should be approximately 100 FPS (1/0.01), with some tolerance
        assert 50 < fps < 200  # Allow for timing variations

    def test_fps_calculator_edge_cases(self):
        """Test edge cases for FPS calculator."""
        fps_calc = FPSCalculator()
        
        # Test with very small buffer
        fps_calc_small = FPSCalculator(buffer_size=1)
        
        with patch('time.perf_counter', side_effect=[1.0, 1.1]):
            fps_calc_small.tick()
            fps_calc_small.tick()
        
        assert len(fps_calc_small._timestamps) == 1
        assert abs(fps_calc_small.get_fps() - 10.0) < 1e-10

    def test_fps_calculator_consistency(self):
        """Test FPS calculator consistency over multiple calculations."""
        fps_calc = FPSCalculator()
        
        # Add consistent timing data
        with patch('time.perf_counter', side_effect=[1.0, 1.05, 1.10, 1.15]):
            fps_calc.tick()
            fps_calc.tick()  # 0.05s
            fps_calc.tick()  # 0.05s
            fps_calc.tick()  # 0.05s
        
        fps1 = fps_calc.get_fps()
        fps2 = fps_calc.get_fps()
        
        # Multiple calls should return same result
        assert fps1 == fps2
        assert abs(fps1 - 20.0) < 1e-10  # 1/0.05 = 20 FPS

    def test_fps_calculator_performance(self):
        """Test FPS calculator performance with many operations."""
        fps_calc = FPSCalculator(buffer_size=1000)
        
        start_time = time.time()
        
        # Perform many operations
        for i in range(1000):
            with patch('time.perf_counter', return_value=float(i * 0.001)):
                fps_calc.tick()
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete quickly

        # Should still work correctly
        fps = fps_calc.get_fps()
        assert fps > 0