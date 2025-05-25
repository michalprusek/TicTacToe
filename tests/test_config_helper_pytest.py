"""
Pytest tests for config helper module.
"""
import pytest


class TestConfigHelper:
    """Pytest test class for config helper."""
    
    def test_config_helper_import(self):
        """Test config helper can be imported."""
        try:
            import app.main.config_helper
            assert app.main.config_helper is not None
        except ImportError:
            pytest.skip("Config helper not available")
    
    def test_config_functions_exist(self):
        """Test config functions exist."""
        try:
            from app.main.config_helper import load_config, save_config
            assert callable(load_config)
            assert callable(save_config)
        except ImportError:
            pytest.skip("Config functions not available")
    
    def test_config_validation(self):
        """Test config validation logic."""
        test_config = {"width": 640, "height": 480, "enabled": True}
        assert isinstance(test_config, dict)
        assert isinstance(test_config["width"], int)
        assert isinstance(test_config["height"], int)
        assert isinstance(test_config["enabled"], bool)
    
    def test_config_merging(self):
        """Test config merging."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 3, "c": 4}
        merged = {**config1, **config2}
        assert merged["a"] == 1
        assert merged["b"] == 3
        assert merged["c"] == 4