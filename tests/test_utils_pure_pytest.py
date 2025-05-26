# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
"""
Pure pytest tests for utils module.
"""
import pytest
import numpy as np
from app.core.utils import FPSCalculator


class TestFPSCalculator:
    """Pure pytest test class for FPSCalculator."""
    
    @pytest.fixture
    def fps_calc(self):
        """Create FPS calculator for testing."""
        return FPSCalculator(buffer_size=5)
    
    def test_fps_calculator_initialization(self, fps_calc):
        """Test FPS calculator initialization."""
        assert fps_calc is not None
        assert fps_calc.buffer_size == 5
        assert fps_calc.get_fps() == 0.0
    
    def test_fps_calculator_tick(self, fps_calc):
        """Test FPS calculator tick method."""
        fps_calc.tick()
        assert fps_calc.get_fps() == 0.0  # First tick, no duration yet
        
        import time
        time.sleep(0.01)  # Small delay
        fps_calc.tick()
        fps = fps_calc.get_fps()
        assert fps > 0
    
    def test_fps_calculator_reset(self, fps_calc):
        """Test FPS calculator reset."""
        fps_calc.tick()
        fps_calc.reset()
        assert fps_calc.get_fps() == 0.0
    
    @pytest.mark.parametrize("buffer_size", [1, 5, 10, 20])
    def test_different_buffer_sizes(self, buffer_size):
        """Test FPS calculator with different buffer sizes."""
        fps_calc = FPSCalculator(buffer_size=buffer_size)
        assert fps_calc.buffer_size == buffer_size
        assert fps_calc.get_fps() == 0.0
    
    def test_invalid_buffer_size(self):
        """Test FPS calculator with invalid buffer size."""
        with pytest.raises(ValueError):
            FPSCalculator(buffer_size=0)
        
        with pytest.raises(ValueError):
            FPSCalculator(buffer_size=-1)
