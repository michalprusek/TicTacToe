# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Path utilities for the TicTacToe application.
Consolidates repeated path setup patterns to avoid duplication.
Cross-platform compatible using pathlib.
"""

import sys
from pathlib import Path
from typing import Optional


def setup_project_path() -> Path:
    """
    Setup project root in sys.path for imports.
    Used across multiple modules to ensure consistent path handling.

    Returns:
        Path: The project root directory path
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent

    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def setup_uarm_sdk_path() -> Optional[Path]:
    """
    Setup uArm SDK path for arm controller.
    Returns the uArm SDK path if it exists.

    Returns:
        Optional[Path]: The uArm SDK path if it exists, None otherwise
    """
    project_root = get_project_root()
    uarm_sdk_path = project_root / "uArm-Python-SDK"

    if uarm_sdk_path.exists():
        uarm_sdk_path_str = str(uarm_sdk_path)
        if uarm_sdk_path_str not in sys.path:
            sys.path.insert(0, uarm_sdk_path_str)
        return uarm_sdk_path
    return None


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path: The project root directory path
    """
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_weights_path() -> Path:
    """
    Get the weights directory path.

    Returns:
        Path: The weights directory path
    """
    return get_project_root() / "weights"


def get_calibration_path() -> Path:
    """
    Get the calibration directory path.

    Returns:
        Path: The calibration directory path
    """
    return get_project_root() / "app" / "calibration"


def get_calibration_file_path() -> Path:
    """
    Get the calibration file path.

    Returns:
        Path: The calibration file path
    """
    return get_calibration_path() / "hand_eye_calibration.json"


def get_detection_model_path() -> Path:
    """
    Get the detection model file path.

    Returns:
        Path: The detection model file path
    """
    return get_weights_path() / "best_detection.pt"


def get_pose_model_path() -> Path:
    """
    Get the pose model file path.

    Returns:
        Path: The pose model file path
    """
    return get_weights_path() / "best_pose.pt"


# Backward compatibility functions that return strings
def get_project_root_str() -> str:
    """Get the project root directory as string (backward compatibility)."""
    return str(get_project_root())


def get_weights_path_str() -> str:
    """Get the weights directory path as string (backward compatibility)."""
    return str(get_weights_path())


def get_calibration_path_str() -> str:
    """Get calibration directory path as string (backward compatibility)."""
    return str(get_calibration_path())
