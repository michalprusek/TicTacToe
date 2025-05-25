"""
Extended tests for config module.
"""
import unittest

from app.core.config import (
    AppConfig, GameDetectorConfig, ArmControllerConfig, GameConfig
)


class TestConfigExtended(unittest.TestCase):
    
    def test_app_config_to_dict(self):
        """Test AppConfig.to_dict method."""
        config = AppConfig()
        config.debug_mode = True
        config.calibration_file = "test.json"
        
        config_dict = config.to_dict()
        
        self.assertIn('game_detector', config_dict)
        self.assertIn('arm_controller', config_dict)
        self.assertIn('game', config_dict)
        self.assertIn('calibration_file', config_dict)
        self.assertIn('debug_mode', config_dict)
        
        self.assertEqual(config_dict['calibration_file'], "test.json")
        self.assertEqual(config_dict['debug_mode'], True)
    
    def test_app_config_from_dict_empty(self):
        """Test AppConfig.from_dict with empty dict."""
        config_dict = {}
        config = AppConfig.from_dict(config_dict)
        
        self.assertIsInstance(config, AppConfig)
        self.assertIsInstance(config.game_detector, GameDetectorConfig)
        self.assertIsInstance(config.arm_controller, ArmControllerConfig)
        self.assertIsInstance(config.game, GameConfig)
    
    def test_app_config_from_dict_partial(self):
        """Test AppConfig.from_dict with partial data."""
        config_dict = {
            "calibration_file": "test_calibration.json",
            "debug_mode": True
        }
        
        config = AppConfig.from_dict(config_dict)
        
        self.assertEqual(config.calibration_file, "test_calibration.json")
        self.assertEqual(config.debug_mode, True)    
    def test_app_config_from_dict_invalid_keys(self):
        """Test from_dict with invalid attribute keys."""
        config_dict = {
            "game_detector": {"invalid_key": 999},
            "arm_controller": {"another_invalid": "test"},
            "game": {"bad_attr": 123}
        }
        
        config = AppConfig.from_dict(config_dict)
        # Should still create valid config, just ignore invalid keys
        self.assertIsInstance(config, AppConfig)


if __name__ == '__main__':
    unittest.main()