"""
Shared styling constants and functions to avoid style duplication.
"""

# Common button style patterns
BUTTON_BASE_STYLE = """
    QPushButton {{
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }}
"""

BUTTON_HOVER_PRESSED_PATTERN = """
    QPushButton:hover {{
        background-color: {hover_color};
    }}
    QPushButton:pressed {{
        background-color: {pressed_color};
    }}
"""

# Predefined button color schemes
BUTTON_COLOR_SCHEMES = {
    "primary": {
        "background": "#3498db",
        "hover": "#2980b9",
        "pressed": "#1f618d"
    },
    "danger": {
        "background": "#e74c3c", 
        "hover": "#c0392b",
        "pressed": "#a93226"
    },
    "warning": {
        "background": "#f39c12",
        "hover": "#d35400", 
        "pressed": "#c0392b"
    },
    "success": {
        "background": "#27ae60",
        "hover": "#229954",
        "pressed": "#1e8449"
    },
    "purple": {
        "background": "#9b59b6",
        "hover": "#8e44ad",
        "pressed": "#7d3c98"
    },
    "teal": {
        "background": "#1abc9c",
        "hover": "#16a085",
        "pressed": "#138d75"
    }
}


def create_button_style(color_scheme: str = "primary", padding: str = "8px 16px") -> str:
    """Create a complete button style with the specified color scheme."""
    scheme = BUTTON_COLOR_SCHEMES.get(color_scheme, BUTTON_COLOR_SCHEMES["primary"])

    base_style = BUTTON_BASE_STYLE.format(padding=padding)
    background_style = (f"QPushButton {{ background-color: {scheme['background']}; "
                       f"padding: {padding};")
    base_style = base_style.replace("QPushButton {", background_style)

    hover_pressed = BUTTON_HOVER_PRESSED_PATTERN.format(
        hover_color=scheme["hover"],
        pressed_color=scheme["pressed"]
    )

    return base_style + hover_pressed


def create_reset_button_style() -> str:
    """Create the standard reset button style used across components."""
    return """
        QPushButton {
            background-color: #e74c3c;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #c0392b;
        }
        QPushButton:pressed {
            background-color: #a93226;
        }
    """