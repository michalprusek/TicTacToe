"""
TicTacToe board widget module for the TicTacToe application.
"""
import logging
from PyQt5.QtWidgets import QWidget, QTimer
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QLinearGradient, QRadialGradient

from app.main import game_logic


class TicTacToeBoard(QWidget):
    """Widget for displaying and interacting with the game board"""
    cell_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = game_logic.create_board()
        self.cell_size = 150  # Zvětšení buněk
        self.setMinimumSize(3 * self.cell_size, 3 * self.cell_size)
        self.setStyleSheet("background-color: #252830; border-radius: 10px;")
        self.winning_line = None  # Pro uložení výherní čáry
        self.highlighted_cells = []  # Seznam buněk, které mají být zvýrazněny
        self.highlight_timer = QTimer(self)  # Timer pro animaci zvýraznění
        self.highlight_timer.timeout.connect(self.update_highlight)
        self.highlight_alpha = 0  # Průhlednost zvýraznění (0-255)
        self.highlight_fade_in = True  # Směr animace (fade in/out)
        self.logger = logging.getLogger(__name__)

    def paintEvent(self, event):
        """Paint the board and symbols"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # Vyhlazování hran

        # Vyplnění pozadí - moderní tmavý gradient
        bg_gradient = QLinearGradient(0, 0, self.width(), self.height())
        bg_gradient.setColorAt(0, QColor(37, 40, 48))  # Tmavě modrá
        bg_gradient.setColorAt(1, QColor(32, 35, 42))  # Ještě tmavší modrá
        painter.fillRect(self.rect(), bg_gradient)

        # Zvýraznění buněk (pokud existují)
        if self.highlighted_cells:
            for r, c in self.highlighted_cells:
                x = c * self.cell_size
                y = r * self.cell_size
                highlight_color = QColor(255, 255, 0, self.highlight_alpha)  # Žlutá s průhledností
                painter.fillRect(x, y, self.cell_size, self.cell_size, highlight_color)

        # Draw grid lines - moderní design
        grid_pen = QPen(QColor(100, 100, 120))  # Světlejší šedá pro mřížku
        grid_pen.setWidth(3)  # Silnější čáry
        painter.setPen(grid_pen)

        # Vertical lines
        painter.drawLine(self.cell_size, 0, self.cell_size, 3 * self.cell_size)
        painter.drawLine(
            2 * self.cell_size,
            0,
            2 * self.cell_size,
            3 * self.cell_size)

        # Horizontal lines
        painter.drawLine(0, self.cell_size, 3 * self.cell_size, self.cell_size)
        painter.drawLine(
            0,
            2 * self.cell_size,
            3 * self.cell_size,
            2 * self.cell_size)

        # Draw X and O - moderní design
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == game_logic.PLAYER_X:
                    self._draw_x(painter, c, r)
                elif self.board[r][c] == game_logic.PLAYER_O:
                    self._draw_o(painter, c, r)

        # Draw winning line if exists
        if self.winning_line:
            self._draw_winning_line(painter)

    def _draw_x(self, painter, col, row):
        """Draw X symbol with gradient effect"""
        # Nastavení pera pro X - moderní design
        x_pen = QPen(QColor(52, 152, 219))  # Modrá barva pro X
        x_pen.setWidth(10)  # Silnější čáry
        x_pen.setCapStyle(Qt.RoundCap)  # Zakulacené konce
        painter.setPen(x_pen)

        # Výpočet souřadnic
        margin = self.cell_size * 0.2  # Menší okraj pro větší symbol
        x1 = col * self.cell_size + margin
        y1 = row * self.cell_size + margin
        x2 = (col + 1) * self.cell_size - margin
        y2 = (row + 1) * self.cell_size - margin

        # Kreslení X s gradientem
        gradient = QLinearGradient(x1, y1, x2, y2)
        gradient.setColorAt(0, QColor(41, 128, 185))  # Tmavší modrá
        gradient.setColorAt(1, QColor(52, 152, 219))  # Světlejší modrá
        x_pen.setBrush(gradient)
        painter.setPen(x_pen)

        # Kreslení čar
        painter.drawLine(x1, y1, x2, y2)
        painter.drawLine(x1, y2, x2, y1)

    def _draw_o(self, painter, col, row):
        """Draw O symbol with gradient effect"""
        # Výpočet souřadnic
        center_x = col * self.cell_size + self.cell_size / 2
        center_y = row * self.cell_size + self.cell_size / 2
        radius = self.cell_size * 0.35  # Menší poloměr pro lepší vzhled

        # Nastavení pera pro O - moderní design
        o_pen = QPen(QColor(231, 76, 60))  # Červená barva pro O
        o_pen.setWidth(10)  # Silnější čáry
        painter.setPen(o_pen)

        # Kreslení O s radiálním gradientem
        gradient = QRadialGradient(center_x, center_y, radius)
        gradient.setColorAt(0, QColor(231, 76, 60))  # Světlejší červená
        gradient.setColorAt(1, QColor(192, 57, 43))  # Tmavší červená
        o_pen.setBrush(gradient)
        painter.setPen(o_pen)

        # Kreslení kruhu
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

    def _draw_winning_line(self, painter):
        """Draw the winning line with animation effect"""
        if not self.winning_line or len(self.winning_line) != 3:
            return

        # Nastavení pera pro výherní čáru
        win_pen = QPen(QColor(46, 204, 113))  # Zelená barva pro výherní čáru
        win_pen.setWidth(15)  # Silnější čára
        win_pen.setCapStyle(Qt.RoundCap)  # Zakulacené konce
        painter.setPen(win_pen)

        # Výpočet souřadnic
        start_row, start_col = self.winning_line[0]
        end_row, end_col = self.winning_line[2]

        start_x = start_col * self.cell_size + self.cell_size / 2
        start_y = start_row * self.cell_size + self.cell_size / 2
        end_x = end_col * self.cell_size + self.cell_size / 2
        end_y = end_row * self.cell_size + self.cell_size / 2

        # Kreslení čáry s gradientem
        gradient = QLinearGradient(start_x, start_y, end_x, end_y)
        gradient.setColorAt(0, QColor(46, 204, 113))  # Světlejší zelená
        gradient.setColorAt(1, QColor(39, 174, 96))  # Tmavší zelená
        win_pen.setBrush(gradient)
        painter.setPen(win_pen)

        # Kreslení čáry
        painter.drawLine(start_x, start_y, end_x, end_y)

    def mousePressEvent(self, event):
        """Handle mouse press events for cell selection"""
        if event.button() == Qt.LeftButton:
            # Výpočet řádku a sloupce
            col = event.x() // self.cell_size
            row = event.y() // self.cell_size

            # Kontrola platných souřadnic
            if 0 <= row < 3 and 0 <= col < 3:
                # Emitujeme signál s informací o kliknuté buňce
                self.cell_clicked.emit(row, col)

    def highlight_cells(self, cells):
        """Highlight specified cells with animation"""
        self.highlighted_cells = cells
        self.highlight_alpha = 0
        self.highlight_fade_in = True
        self.highlight_timer.start(30)  # 30ms interval pro plynulou animaci

    def update_highlight(self):
        """Update highlight animation"""
        if self.highlight_fade_in:
            self.highlight_alpha += 15
            if self.highlight_alpha >= 150:
                self.highlight_alpha = 150
                self.highlight_fade_in = False
        else:
            self.highlight_alpha -= 10
            if self.highlight_alpha <= 0:
                self.highlight_alpha = 0
                self.highlight_timer.stop()
                self.highlighted_cells = []

        self.update()  # Překreslení widgetu

    def update_board(self, board, winning_line=None, highlight_changes=False):
        """Update the board state and redraw"""
        # Zjistíme změny mezi starým a novým stavem
        changes = []
        if highlight_changes:
            for r in range(3):
                for c in range(3):
                    if self.board[r][c] != board[r][c] and board[r][c] != game_logic.EMPTY:
                        changes.append((r, c))

        # Aktualizujeme stav
        self.board = board
        if winning_line is not None:
            self.winning_line = winning_line

        # Zvýrazníme změny, pokud existují
        if changes:
            self.highlight_cells(changes)

        # Překreslíme widget
        self.update()
