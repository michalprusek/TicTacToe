# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and symbol caching implementation
"""
Symbol caching system for TicTacToe application.

This module implements intelligent symbol caching that maintains symbol memory
even when they become temporarily occluded by the player's hand, ensuring
consistent AI decision-making based on complete board state.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

# Define EMPTY constant locally to avoid circular import
EMPTY = ' '  # Must match game_state.py constant


class CachedSymbol:
    """Represents a cached symbol with metadata."""
    
    def __init__(self, symbol: str, position: Tuple[int, int], 
                 confidence: float, timestamp: float, 
                 center_uv: np.ndarray):
        """Initialize a cached symbol.
        
        Args:
            symbol: The symbol ('X' or 'O')
            position: Board position (row, col)
            confidence: Detection confidence when cached
            timestamp: When the symbol was last detected
            center_uv: UV coordinates of symbol center
        """
        self.symbol = symbol
        self.position = position
        self.confidence = confidence
        self.last_detected = timestamp
        self.center_uv = center_uv.copy() if center_uv is not None else None
        self.occlusion_count = 0  # How many frames it's been occluded
        self.total_detections = 1  # Total times this symbol was detected
        self.is_persistent = False  # Whether this symbol should persist


class OcclusionDetector:
    """Detects when parts of the board are likely occluded by player's hand."""
    
    def __init__(self, logger=None):
        """Initialize occlusion detector."""
        self.logger = logger or logging.getLogger(__name__)
        self.detection_history = []  # History of detection counts
        self.max_history = 10  # Keep last 10 frames
        
    def detect_occlusion(self, current_detections: List[Dict], 
                        cached_symbols: Dict[Tuple[int, int], CachedSymbol],
                        cell_centers: np.ndarray) -> Set[Tuple[int, int]]:
        """
        Detect which board areas are likely occluded.
        
        Args:
            current_detections: Currently detected symbols
            cached_symbols: Previously cached symbols
            cell_centers: UV coordinates of cell centers
            
        Returns:
            Set of (row, col) positions that are likely occluded
        """
        occluded_areas = set()
        
        # Track detection count for this frame
        detection_count = len(current_detections)
        self.detection_history.append(detection_count)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
            
        # If we have cached symbols but suddenly see fewer detections,
        # suspect occlusion
        if len(cached_symbols) > 0 and detection_count < len(cached_symbols):
            # Find which cached symbols are missing from current detections
            current_positions = set()
            for detection in current_detections:
                if 'center_uv' in detection:
                    pos = self._find_closest_cell(detection['center_uv'], cell_centers)
                    if pos:
                        current_positions.add(pos)
            
            # Cached symbols not in current detections are potentially occluded
            for pos, cached_symbol in cached_symbols.items():
                if pos not in current_positions:
                    # Check if this area shows signs of occlusion
                    if self._is_area_likely_occluded(pos, current_detections, cell_centers):
                        occluded_areas.add(pos)
                        self.logger.debug(
                            "Detected potential occlusion at position %s", pos)
        
        return occluded_areas
    
    def _find_closest_cell(self, center_uv: np.ndarray, 
                          cell_centers: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the closest cell to a UV coordinate."""
        if cell_centers is None or len(cell_centers) != 9:
            return None
            
        min_distance = float('inf')
        closest_cell = None
        
        for i, cell_center in enumerate(cell_centers):
            distance = np.linalg.norm(center_uv - cell_center)
            if distance < min_distance:
                min_distance = distance
                closest_cell = i
                
        if closest_cell is not None:
            row, col = closest_cell // 3, closest_cell % 3
            return (row, col)
        return None
    
    def _is_area_likely_occluded(self, position: Tuple[int, int],
                                current_detections: List[Dict],
                                cell_centers: np.ndarray) -> bool:
        """
        Determine if a specific area is likely occluded.
        
        This uses heuristics like sudden detection loss, nearby detection
        patterns, and detection history.
        """
        # Simple heuristic: if we had consistent detections before and
        # suddenly lost them, it's likely occlusion
        if len(self.detection_history) >= 3:
            recent_avg = sum(self.detection_history[-3:]) / 3
            current_count = len(current_detections)
            
            # If current detections are significantly lower than recent average
            if current_count < recent_avg * 0.7:
                return True
                
        # Additional heuristic: check if nearby cells are also missing detections
        # (suggests a hand covering multiple cells)
        row, col = position
        nearby_missing = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    # Check if this nearby position is also missing
                    found = False
                    for detection in current_detections:
                        if 'center_uv' in detection:
                            pos = self._find_closest_cell(detection['center_uv'], cell_centers)
                            if pos == (nr, nc):
                                found = True
                                break
                    if not found:
                        nearby_missing += 1
        
        # If multiple nearby cells are missing, likely occlusion
        return nearby_missing >= 2


class SymbolCache:
    """
    Intelligent symbol caching system that maintains symbol memory
    even during temporary occlusion.
    """
    
    def __init__(self, logger=None):
        """Initialize symbol cache."""
        self.logger = logger or logging.getLogger(__name__)
        self.cached_symbols: Dict[Tuple[int, int], CachedSymbol] = {}
        self.occlusion_detector = OcclusionDetector(logger)
        
        # Configuration
        self.max_occlusion_time = 5.0  # Max seconds to keep occluded symbol
        self.min_confidence_for_cache = 0.8  # Min confidence to cache
        self.occlusion_threshold = 3  # Frames before considering removal
        
        # State tracking
        self.using_cached_data = False
        self.last_cache_update = time.time()
        
    def update_cache(self, current_detections: List[Dict], 
                    cell_centers: np.ndarray, timestamp: float,
                    class_id_to_player: Dict[int, str]) -> Tuple[List[List[str]], bool]:
        """
        Update symbol cache with current detections and return board state.
        
        Args:
            current_detections: Current YOLO detections
            cell_centers: UV coordinates of cell centers  
            timestamp: Current timestamp
            class_id_to_player: Mapping from class ID to player symbol
            
        Returns:
            Tuple of (board_state, using_cached_data)
        """
        self.last_cache_update = timestamp
        
        # Convert detections to expected format
        converted_detections = self._convert_detections(
            current_detections, class_id_to_player)
        
        # Detect occlusion
        occluded_areas = self.occlusion_detector.detect_occlusion(
            converted_detections, self.cached_symbols, cell_centers)
        
        # Update cache with current detections
        self._update_cached_symbols(converted_detections, cell_centers, timestamp)
        
        # Clean up old/invalid cached symbols
        self._cleanup_cache(timestamp, occluded_areas)
        
        # Build board state from cache
        board_state = self._build_board_from_cache()
        
        # Determine if we're using cached data
        self.using_cached_data = len(occluded_areas) > 0 or len(self.cached_symbols) > len(converted_detections)
        
        if self.using_cached_data:
            self.logger.info(
                "🔄 Using cached symbol data - %d cached, %d detected, %d occluded",
                len(self.cached_symbols), len(converted_detections), len(occluded_areas))
        
        return board_state, self.using_cached_data
    
    def _convert_detections(self, detections: List[Dict], 
                           class_id_to_player: Dict[int, str]) -> List[Dict]:
        """Convert raw detections to standardized format."""
        converted = []
        for detection in detections:
            if all(key in detection for key in ['box', 'label', 'confidence', 'class_id']):
                # Calculate center from bounding box
                box = detection['box']
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                converted.append({
                    'center_uv': np.array([center_x, center_y]),
                    'player': detection['label'],  # Use label directly
                    'confidence': detection['confidence']
                })
        return converted
    
    def _update_cached_symbols(self, detections: List[Dict], 
                              cell_centers: np.ndarray, timestamp: float):
        """Update cached symbols with current detections."""
        if cell_centers is None or len(cell_centers) != 9:
            return
            
        # Track which positions we've seen in current frame
        current_positions = set()
        
        for detection in detections:
            if detection['confidence'] < self.min_confidence_for_cache:
                continue
                
            # Find closest cell
            center_uv = detection['center_uv']
            position = self._find_closest_cell(center_uv, cell_centers)
            
            if position:
                current_positions.add(position)
                
                # Update or create cached symbol
                if position in self.cached_symbols:
                    cached = self.cached_symbols[position]
                    cached.last_detected = timestamp
                    cached.confidence = max(cached.confidence, detection['confidence'])
                    cached.occlusion_count = 0  # Reset occlusion counter
                    cached.total_detections += 1
                    cached.is_persistent = cached.total_detections >= 3
                else:
                    # New symbol
                    self.cached_symbols[position] = CachedSymbol(
                        symbol=detection['player'],
                        position=position,
                        confidence=detection['confidence'],
                        timestamp=timestamp,
                        center_uv=center_uv
                    )
                    self.logger.info(
                        "📝 Cached new symbol %s at %s (conf: %.3f)",
                        detection['player'], position, detection['confidence'])
        
        # Increment occlusion count for symbols not seen this frame
        for position, cached_symbol in self.cached_symbols.items():
            if position not in current_positions:
                cached_symbol.occlusion_count += 1
    
    def _find_closest_cell(self, center_uv: np.ndarray, 
                          cell_centers: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the closest cell to a UV coordinate."""
        min_distance = float('inf')
        closest_cell = None
        
        for i, cell_center in enumerate(cell_centers):
            distance = np.linalg.norm(center_uv - cell_center)
            if distance < min_distance:
                min_distance = distance
                closest_cell = i
                
        if closest_cell is not None:
            row, col = closest_cell // 3, closest_cell % 3
            return (row, col)
        return None
    
    def _cleanup_cache(self, timestamp: float, occluded_areas: Set[Tuple[int, int]]):
        """Remove old or invalid cached symbols."""
        to_remove = []
        
        for position, cached_symbol in self.cached_symbols.items():
            # Remove if too old and not in occluded area
            time_since_detection = timestamp - cached_symbol.last_detected
            
            if position not in occluded_areas:
                # Not occluded, so if we haven't seen it for a while, remove it
                if (time_since_detection > self.max_occlusion_time or 
                    cached_symbol.occlusion_count > self.occlusion_threshold):
                    to_remove.append(position)
                    self.logger.info(
                        "🗑️ Removing cached symbol %s at %s (age: %.1fs, occlusion: %d)",
                        cached_symbol.symbol, position, time_since_detection,
                        cached_symbol.occlusion_count)
            else:
                # In occluded area, be more lenient
                if time_since_detection > self.max_occlusion_time * 2:
                    to_remove.append(position)
                    self.logger.info(
                        "🗑️ Removing old occluded symbol %s at %s (age: %.1fs)",
                        cached_symbol.symbol, position, time_since_detection)
        
        for position in to_remove:
            del self.cached_symbols[position]
    
    def _build_board_from_cache(self) -> List[List[str]]:
        """Build board state from cached symbols."""
        board = [[EMPTY for _ in range(3)] for _ in range(3)]
        
        for position, cached_symbol in self.cached_symbols.items():
            row, col = position
            board[row][col] = cached_symbol.symbol
            
        return board
    
    def clear_cache(self):
        """Clear all cached symbols (e.g., on game reset)."""
        self.logger.info("🧹 Clearing symbol cache")
        self.cached_symbols.clear()
        self.using_cached_data = False
    
    def get_cache_status(self) -> Dict:
        """Get current cache status for debugging."""
        return {
            'cached_count': len(self.cached_symbols),
            'using_cached_data': self.using_cached_data,
            'cached_positions': list(self.cached_symbols.keys()),
            'last_update': self.last_cache_update
        }
