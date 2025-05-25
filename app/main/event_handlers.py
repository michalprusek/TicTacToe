"""
Event handlers compatibility module.
DEPRECATED: Functionality moved to ui_event_handlers.py
"""

# Re-export for backward compatibility
from app.main.ui_event_handlers import UIEventHandlers as GameEventHandler
from app.main.ui_event_handlers import UIEventHandlers as ArmEventHandler
from app.main.ui_event_handlers import UIEventHandlers as UIEventHandler

__all__ = ['GameEventHandler', 'ArmEventHandler', 'UIEventHandler']