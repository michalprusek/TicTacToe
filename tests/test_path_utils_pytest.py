"""
Pure pytest tests for path utils module.
"""
import pytest
from pathlib import Path
from app.main.path_utils import get_project_root, get_weights_dir, get_model_path


class TestPathUtils:
    """Pure pytest test class for path utils."""
    
    def test_get_project_root(self):
        """Test getting project root directory."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
        assert root.is_dir()
    
    def test_get_weights_dir(self):
        """Test getting weights directory."""
        weights_dir = get_weights_dir()
        assert isinstance(weights_dir, Path)
        assert "weights" in str(weights_dir)
    
    def test_get_model_path(self):
        """Test getting model file path."""
        model_path = get_model_path("test_model.pt")
        assert isinstance(model_path, Path)
        assert str(model_path).endswith("test_model.pt")
        assert "weights" in str(model_path)
    
    def test_path_operations(self, tmp_path):
        """Test basic path operations."""
        test_dir = tmp_path / "test"
        assert isinstance(test_dir, Path)
        
        test_file = test_dir / "test.txt"
        assert str(test_file).endswith("test.txt")
    
    def test_path_validation(self):
        """Test path validation."""
        root = get_project_root()
        assert root.is_absolute()
        
        weights = get_weights_dir()
        assert weights.is_absolute()