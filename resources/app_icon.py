"""
Simple icon generator for the TicTacToe application
Creates an icon with a modern tic-tac-toe board design
"""
import os
import sys
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QBrush, QColor, QLinearGradient
from PyQt5.QtCore import Qt, QSize, QRect, QPoint
from PyQt5.QtWidgets import QApplication

def create_tictactoe_icon(size=512):
    """Create a TicTacToe icon with modern design"""
    # Create a pixmap
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)  # Start with transparent background
    
    # Create a painter
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Create a rounded rectangle for the background
    background_rect = QRect(0, 0, size, size)
    
    # Fill with gradient
    gradient = QLinearGradient(0, 0, size, size)
    gradient.setColorAt(0, QColor(52, 73, 94))  # Dark blue
    gradient.setColorAt(1, QColor(44, 62, 80))  # Slightly darker blue
    painter.fillRect(background_rect, gradient)
    
    # Draw a border
    pen = QPen(QColor(41, 128, 185))  # Light blue
    pen.setWidth(size // 20)
    painter.setPen(pen)
    painter.drawRoundedRect(background_rect.adjusted(size//40, size//40, -size//40, -size//40), size//10, size//10)
    
    # Calculate cell size
    cell_size = size // 4
    
    # Draw the grid lines
    pen = QPen(QColor(236, 240, 241))  # Light gray
    pen.setWidth(size // 40)
    painter.setPen(pen)
    
    # Vertical lines
    painter.drawLine(size//3, size//10, size//3, size - size//10)
    painter.drawLine(2*size//3, size//10, 2*size//3, size - size//10)
    
    # Horizontal lines
    painter.drawLine(size//10, size//3, size - size//10, size//3)
    painter.drawLine(size//10, 2*size//3, size - size//10, 2*size//3)
    
    # Draw X in top left
    pen = QPen(QColor(231, 76, 60))  # Red
    pen.setWidth(size // 25)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    margin = size // 15
    painter.drawLine(margin, margin, size//3 - margin, size//3 - margin)
    painter.drawLine(margin, size//3 - margin, size//3 - margin, margin)
    
    # Draw O in center
    pen = QPen(QColor(46, 204, 113))  # Green
    pen.setWidth(size // 25)
    painter.setPen(pen)
    center_x = size // 2
    center_y = size // 2
    radius = size // 9
    painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
    
    # Draw X in bottom right
    pen = QPen(QColor(231, 76, 60))  # Red
    pen.setWidth(size // 25)
    painter.setPen(pen)
    margin = size // 15
    start_x = 2*size//3 + margin
    start_y = 2*size//3 + margin
    end_x = size - margin
    end_y = size - margin
    painter.drawLine(start_x, start_y, end_x, end_y)
    painter.drawLine(start_x, end_y, end_x, start_y)
    
    # End painting
    painter.end()
    
    return QIcon(pixmap)

def save_icon_to_file(icon, size=512, filename="app_icon.png"):
    """Save the icon to a file"""
    pixmap = icon.pixmap(size, size)
    pixmap.save(filename)
    print(f"Icon saved to {filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Create icon
    icon = create_tictactoe_icon()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # Save to file
    save_icon_to_file(icon, filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.png"))