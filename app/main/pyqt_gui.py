# This file is being refactored - main functionality moved to main_gui.py
# This file will be deprecated after refactoring is complete

import sys
import os
import logging

# Add project root to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new main GUI module
from app.main.main_gui import TicTacToeApp

# For backward compatibility, export the main class
__all__ = ['TicTacToeApp']


# Entry point for backward compatibility
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    # Basic logger configuration if not set elsewhere
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )

    app = QApplication(sys.argv)
    window = TicTacToeApp(config=None)
    window.show()
    sys.exit(app.exec_())





        try:
            icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     "resources", "app_icon.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
            else:
                self.logger.warning(f"Soubor ikony nenalezen: {icon_path}")
        except Exception as e:
            self.logger.error(f"Chyba při nastavování ikony: {e}")


        if 'pytest' not in sys.modules:
            self.showFullScreen()

        # Game state attributes
        self.human_player = game_logic.PLAYER_X
        self.ai_player = game_logic.PLAYER_O # AI/Ruka bude hrát za O
        self.current_turn = game_logic.PLAYER_X # Hráč (X) vždy začíná
        self.game_over = False
        self.winner = None
        self.move_counter = 0 # Počítadlo všech platných tahů

        # Arm control flags
        self.waiting_for_detection = False
        self.arm_move_in_progress = False
        self.arm_move_scheduled = False # Tento příznak se zdá nadbytečný, pokud správně řídíme in_progress a cooldown
        self.last_arm_move_time = 0
        self.arm_move_cooldown = 3.0 # Sekundy

        # Detection retry logic
        self.ai_move_row = None # Kam ruka naposledy kreslila
        self.ai_move_col = None
        self.expected_symbol = None # Jaký symbol ruka kreslila
        self.ai_move_retry_count = 0
        self.max_retry_count = 2 # Sníženo pro rychlejší reakci
        self.detection_wait_time = 0.0
        self.max_detection_wait_time = 5.0

        self.tracking_enabled = False
        self.game_paused = False
        self.tracking_timer = QTimer(self)
        self.tracking_timer.timeout.connect(self.track_grid_center)
        self.tracking_interval = 200

        self.debug_mode = self.config.debug_mode if hasattr(self.config, 'debug_mode') else False
        self.debug_window = None

        self.strategy_selector = BernoulliStrategySelector(
            difficulty=self.config.game.default_difficulty if hasattr(self.config, 'game') else DEFAULT_DIFFICULTY)

        self.init_game_components() # Musí být před init_ui, pokud UI závisí na komponentách
        self.init_ui()

        self.camera_thread = CameraThread(camera_index=DEFAULT_CAMERA_INDEX)
        self.camera_thread.game_state_updated.connect(self.handle_detected_game_state)
        self.camera_thread.frame_ready.connect(self.update_camera_view)
        self.camera_thread.fps_updated.connect(self.update_fps_display)
        self.camera_thread.start()

        if self.debug_mode:
            QTimer.singleShot(1000, self.show_debug_window)

        self.timer_setup()
        self.reset_game() # Začít s čistou hrou a stavy

    def timer_setup(self):
        self.update_ui_texts()
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_game_state_machine)
        self.update_timer.start(100) # 10 FPS

    def tr(self, key):
        return self.current_language.get(key, key)

    def _convert_board_1d_to_2d(self, board_1d):
        if isinstance(board_1d, list) and len(board_1d) == 9:
            return [board_1d[i:i + 3] for i in range(0, 9, 3)]
        return board_1d # Předpokládá, že už je 2D nebo None

    def _get_board_symbol_counts(self, board):
        if board is None: return 0, 0, 0
        board_2d = self._convert_board_1d_to_2d(board)
        if not isinstance(board_2d, list) or not all(isinstance(row, list) for row in board_2d):
            return 0,0,0 # Nevalidní formát desky
        x_count = sum(row.count(game_logic.PLAYER_X) for row in board_2d)
        o_count = sum(row.count(game_logic.PLAYER_O) for row in board_2d)
        return x_count, o_count, x_count + o_count

    def _check_arm_availability(self):
        arm_thread_available = (hasattr(self, 'arm_thread') and self.arm_thread and self.arm_thread.connected)
        # arm_controller je teď spíše záložní/legacy, arm_thread je preferovaný
        return arm_thread_available, False # Druhá hodnota je pro arm_controller

    def set_status_style_safe(self, style_key, style_css):
        if not hasattr(self, '_current_style_key'): self._current_style_key = None
        if self._current_style_key != style_key:
            self._current_style_key = style_key
            if hasattr(self, 'main_status_panel') and self.main_status_panel:
                self.main_status_panel.setStyleSheet(style_css)

    def update_status(self, message_key_or_text, is_key=True):
        message = self.tr(message_key_or_text) if is_key else message_key_or_text

        if not hasattr(self, '_status_lock_time'): self._status_lock_time = 0
        if not hasattr(self, '_current_status_text'): self._current_status_text = ""

        current_time = time.time()
        # Zjednodušené zamykání statusu - pokud se zpráva změní, aktualizuj
        # ale ne příliš často pro stejný typ zprávy (např. arm_turn)
        if message == self._current_status_text and current_time - self._status_lock_time < 1.0:
            return

        self._current_status_text = message
        self._status_lock_time = current_time

        if hasattr(self, 'main_status_message') and self.main_status_message:
            board_for_status = None
            if hasattr(self, 'camera_thread') and self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
                 board_for_status = self.camera_thread.last_board_state
            elif hasattr(self, 'board_widget'):
                 board_for_status = self.board_widget.board

            x_count, o_count, total_symbols = self._get_board_symbol_counts(board_for_status)

            status_text_to_show = message.upper()
            style_key = "default"

            # Nastavení stylu podle typu zprávy
            if message_key_or_text == "your_turn":
                status_text_to_show = f"{self.tr('your_turn')} ({self.human_player}) [S: {total_symbols}]"
                self.set_status_style_safe("player", self._get_status_style("player"))
            elif message_key_or_text == "arm_turn":
                status_text_to_show = f"{self.tr('arm_turn')} ({self.ai_player}) [S: {total_symbols}]"
                self.set_status_style_safe("arm", self._get_status_style("arm"))
            elif message_key_or_text == "arm_moving":
                self.set_status_style_safe("arm_moving", self._get_status_style("arm"))
            elif message_key_or_text == "win":
                winner_symbol = self.winner if self.winner else "?"
                status_text_to_show = f"{self.tr('win')} - {winner_symbol}"
                self.set_status_style_safe("win", self._get_status_style("win"))
            elif message_key_or_text == "draw":
                self.set_status_style_safe("draw", self._get_status_style("draw"))
            elif message_key_or_text == "game_over":
                 self.set_status_style_safe("game_over", self._get_status_style("error")) # Použijeme error styl pro konec hry
            elif message_key_or_text == "grid_not_visible":
                 self.set_status_style_safe("error", self._get_status_style("error"))
            elif message_key_or_text == "grid_visible":
                 self.set_status_style_safe("success", self._get_status_style("success"))
                 QTimer.singleShot(2000, self.reset_status_panel_style) # Reset po chvíli
            # ... další specifické zprávy

            self.main_status_message.setText(status_text_to_show)

        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.setText("") # Starý status label se již nepoužívá pro hlavní info

    def _get_status_style(self, status_type):
        styles = {
            'arm': "background-color: #9b59b6; border-radius: 10px; border: 2px solid #8e44ad;",
            'ai': "background-color: #3498db; border-radius: 10px; border: 2px solid #2980b9;", # AI je v podstatě ruka
            'player': "background-color: #e74c3c; border-radius: 10px; border: 2px solid #c0392b;",
            'win': "background-color: #2ecc71; border-radius: 10px; border: 2px solid #27ae60;",
            'draw': "background-color: #f1c40f; border-radius: 10px; border: 2px solid #f39c12;",
            'error': "background-color: #e74c3c; border-radius: 10px; border: 2px solid #c0392b;",
            'success': "background-color: #2ecc71; border-radius: 10px; border: 2px solid #27ae60;",
            'default': "background-color: #34495e; border-radius: 10px; border: 2px solid #2c3e50;"
        }
        return styles.get(status_type, styles['default'])

    def reset_status_panel_style(self):
        self.set_status_style_safe("default", self._get_status_style("default"))
        # Po resetu stylu aktualizujeme text podle aktuálního stavu hry
        if self.game_over:
            if self.winner == game_logic.TIE: self.update_status("draw")
            elif self.winner: self.update_status("win")
            else: self.update_status("game_over")
        elif self.current_turn == self.human_player: self.update_status("your_turn")
        elif self.current_turn == self.ai_player: self.update_status("arm_turn")
        else: self.update_status(self.tr("new_game_detected"), is_key=False)


    def change_language(self):
        self.is_czech = not self.is_czech
        self.current_language = LANG_CS if self.is_czech else LANG_EN
        self.update_ui_texts()

    def update_ui_texts(self):
        if hasattr(self, 'reset_button'): self.reset_button.setText(self.tr("new_game"))
        if hasattr(self, 'debug_button'): self.debug_button.setText(self.tr("debug"))
        if hasattr(self, 'language_button'): self.language_button.setText("🇨🇿" if self.is_czech else "🇬🇧")
        if hasattr(self, 'difficulty_label'): self.difficulty_label.setText(self.tr("difficulty"))

        # Aktualizace hlavního statusu na základě aktuálního stavu
        if self.game_over:
            if self.winner == game_logic.TIE: self.update_status("draw")
            elif self.winner: self.update_status("win") # Zpráva "WIN" je obecná, konkrétní výherce v textu
            else: self.update_status("game_over") # Pokud není jasný výherce, ale hra skončila
        elif self.current_turn == self.human_player:
            self.update_status("your_turn")
        elif self.current_turn == self.ai_player:
             # Rozlišení, zda ruka kreslí nebo přemýšlí
            if self.arm_move_in_progress or self.waiting_for_detection:
                 self.update_status("arm_moving")
            else:
                 self.update_status("arm_turn")
        else: # Hra ještě nezačala nebo je v neznámém stavu
            self.update_status(self.tr("new_game_detected"), is_key=False)


    def update_fps_display(self, fps):
        if hasattr(self, 'debug_window') and self.debug_window and hasattr(self.debug_window, 'update_fps'):
            self.debug_window.update_fps(fps)
        # Můžete přidat i zobrazení FPS do hlavního okna, pokud chcete
        # if hasattr(self, 'main_fps_label'): self.main_fps_label.setText(f"FPS: {fps:.1f}")


    def update_camera_view(self, frame):
        if frame is None: return

        processed_frame, game_state_from_detection = self._get_detection_data()

        self._update_main_camera_view(frame) # Vždy zobrazí surový frame, pokud není implementováno jinak

        if game_state_from_detection: # game_state_from_detection je objekt GameState nebo podobný
            self._handle_grid_warnings(game_state_from_detection)
            # Aktualizace desky v GUI z detekce se děje v handle_detected_game_state
            # self._update_board_from_game_state(game_state_from_detection) # Toto může být duplicitní

        self._update_debug_window(processed_frame if processed_frame is not None else frame,
                                  frame, game_state_from_detection)

    def _get_detection_data(self):
        processed_frame, game_state = None, None
        if hasattr(self, 'camera_thread') and self.camera_thread:
            if hasattr(self.camera_thread, 'detection_thread') and self.camera_thread.detection_thread:
                result = self.camera_thread.detection_thread.get_latest_result()
                if result and result[0] is not None: processed_frame = result[0]
                if hasattr(self.camera_thread.detection_thread, 'latest_game_state'):
                    game_state = self.camera_thread.detection_thread.latest_game_state
        return processed_frame, game_state

    def _update_main_camera_view(self, frame):
        # V této aplikaci hlavní okno nemá přímý CameraView, ten je v DebugWindow.
        # Pokud byste ho chtěli přidat, zde by byla aktualizace.
        pass

    def _handle_grid_warnings(self, game_state_obj): # game_state_obj je instance GameState
        has_grid_issue = False
        grid_issue_message = ""

        if hasattr(game_state_obj, 'is_physical_grid_valid') and callable(game_state_obj.is_physical_grid_valid):
            if not game_state_obj.is_physical_grid_valid():
                has_grid_issue = True
                # Pokus o získání konkrétnější zprávy, pokud je k dispozici
                if hasattr(game_state_obj, 'grid_issue_message'):
                    grid_issue_message = game_state_obj.grid_issue_message
                elif hasattr(game_state_obj, '_grid_points'):
                     non_zero_count = np.count_nonzero(np.sum(np.abs(game_state_obj._grid_points), axis=1)) if game_state_obj._grid_points is not None else 0
                     grid_issue_message = f"Mřížka není kompletně viditelná! Detekováno {non_zero_count}/16 bodů."
                else:
                    grid_issue_message = self.tr("grid_not_visible")

        if has_grid_issue:
            if not hasattr(self, 'grid_warning_active') or not self.grid_warning_active:
                self.logger.warning(f"Zobrazuji varování mřížky: {grid_issue_message}")
                self._show_grid_warning_panel(grid_issue_message) # ZMĚNA: Přepracováno pro panel
                self.grid_warning_active = True
                self.update_status("grid_not_visible")
        else:
            if hasattr(self, 'grid_warning_active') and self.grid_warning_active:
                self.logger.info("Skrývám varování mřížky.")
                self._hide_grid_warning_panel() # ZMĚNA: Přepracováno pro panel
                self.grid_warning_active = False
                self.update_status("grid_visible")
                QTimer.singleShot(2000, self.reset_status_panel_style)


    def _show_grid_warning_panel(self, message):
        if not hasattr(self, 'warning_panel'):
            self.warning_panel = QWidget(self)
            self.warning_panel.setStyleSheet("background-color: rgba(231, 76, 60, 0.9); border-radius: 10px; border: 1px solid #c0392b;")
            layout = QVBoxLayout(self.warning_panel)
            self.warning_icon_label = QLabel("⚠️")
            self.warning_icon_label.setAlignment(Qt.AlignCenter)
            self.warning_icon_label.setStyleSheet("font-size: 30px; color: white; margin-bottom: 5px;")
            self.warning_text_label = QLabel(message)
            self.warning_text_label.setAlignment(Qt.AlignCenter)
            self.warning_text_label.setWordWrap(True)
            self.warning_text_label.setStyleSheet("font-size: 14px; color: white; font-weight: bold;")
            layout.addWidget(self.warning_icon_label)
            layout.addWidget(self.warning_text_label)
            self.warning_panel.setFixedSize(400, 120) # Menší panel

        self.warning_text_label.setText(message) # Aktualizace textu, pokud se změní
        # Umístění panelu (např. nad herní deskou nebo uprostřed)
        if hasattr(self, 'board_widget') and self.board_widget:
            board_rect = self.board_widget.geometry()
            panel_x = board_rect.center().x() - self.warning_panel.width() // 2
            panel_y = board_rect.y() - self.warning_panel.height() - 10 # Nad deskou
            if panel_y < 0: panel_y = board_rect.center().y() - self.warning_panel.height() // 2 # Pokud by šlo mimo, tak doprostřed
            self.warning_panel.move(max(0, panel_x), max(0, panel_y))
        else: # Fallback na střed okna
            self.warning_panel.move((self.width() - self.warning_panel.width()) // 2, (self.height() - self.warning_panel.height()) // 2)

        self.warning_panel.show()
        self.warning_panel.raise_()

    def _hide_grid_warning_panel(self):
        if hasattr(self, 'warning_panel') and self.warning_panel.isVisible():
            self.warning_panel.hide()

    def _update_board_from_game_state(self, game_state_obj): # game_state_obj je instance GameState
        if hasattr(game_state_obj, '_board_state') and hasattr(self, 'board_widget'):
            # game_state_obj._board_state by měla být 2D deska
            detected_board = self._convert_board_1d_to_2d(game_state_obj._board_state)
            if detected_board:
                 self.board_widget.update_board(detected_board, None, highlight_changes=False)

    def _update_debug_window(self, processed_frame, raw_frame, game_state_obj):
        if hasattr(self, 'debug_window') and self.debug_window and self.debug_window.isVisible():
            display_frame = processed_frame if processed_frame is not None else raw_frame
            try:
                if hasattr(self.debug_window, 'camera_view') and self.debug_window.camera_view:
                     self.debug_window.camera_view.update_frame(display_frame.copy())
                if game_state_obj and hasattr(game_state_obj, '_board_state'):
                    board_to_update = self._convert_board_1d_to_2d(game_state_obj._board_state)
                    if board_to_update and hasattr(self.debug_window, 'update_board_state'):
                        self.debug_window.update_board_state(board_to_update)
            except Exception as e:
                self.logger.error(f"Chyba při aktualizaci debug okna: {e}")


    def handle_cell_clicked(self, row, col):
        self.logger.info(f"Hráč klikl na buňku ({row}, {col})")
        if self.game_over or self.current_turn != self.human_player or self.arm_move_in_progress or self.waiting_for_detection:
            self.logger.warning(f"Klik ignorován: game_over={self.game_over}, current_turn={self.current_turn}, arm_busy={self.arm_move_in_progress or self.waiting_for_detection}")
            return

        if self.board_widget.board[row][col] != game_logic.EMPTY:
            self.logger.info("Buňka je již obsazena.")
            return

        # Hráčův tah se nezaznamenává přímo do self.board_widget.board zde.
        # Místo toho čekáme, až kamera detekuje nový symbol.
        # Tato metoda v podstatě jen signalizuje záměr hráče.
        # Hlavní logika se odehraje v handle_detected_game_state.
        self.logger.info(f"Hráč ({self.human_player}) zamýšlí táhnout na ({row},{col}). Čekám na detekci symbolem.")
        self.update_status(self.tr("waiting_detection"), is_key=False) # Informujeme, že čekáme na detekci hráčova tahu


    def handle_reset_button_click(self):
        self.logger.info("Stisknuto tlačítko Reset.")
        self.reset_game()

    def reset_game(self):
        self.logger.info("Resetuji hru.")
        if hasattr(self, 'board_widget') and self.board_widget:
            empty_board = game_logic.create_board()
            self.board_widget.update_board(empty_board, None, highlight_changes=False)
            self.board_widget.winning_line = None
            self.board_widget.update()

        self.game_over = False
        self.winner = None

        self.human_player = game_logic.PLAYER_X
        self.ai_player = game_logic.PLAYER_O
        self.current_turn = game_logic.PLAYER_X # Hráč X vždy začíná
        self.move_counter = 0

        # Reset arm flags
        self.waiting_for_detection = False
        self.arm_move_in_progress = False
        self.arm_move_scheduled = False # I když je možná nadbytečný, resetujeme
        self.ai_move_row = None
        self.ai_move_col = None
        self.expected_symbol = None
        self.ai_move_retry_count = 0
        self.detection_wait_time = 0.0
        self.last_arm_move_time = 0 # Aby ruka mohla hned hrát, pokud je na tahu po resetu

        if hasattr(self, '_celebration_triggered'): # Reset příznaku pro oslavu
            del self._celebration_triggered

        self.update_status("new_game_detected")
        self.logger.info(f"Hra resetována. Na tahu: {self.current_turn}")
        self.move_to_neutral_position() # Po resetu přesuň ruku do neutrálu


    def handle_debug_button_click(self): self.show_debug_window()
    def handle_calibrate_button_click(self): self.calibrate_arm()
    def handle_park_button_click(self): self.park_arm()

    def park_arm(self):
        return self._unified_arm_command('park', x=PARK_X, y=PARK_Y, wait=True)

    def calibrate_arm(self):
        if not hasattr(self, 'arm_thread') or not self.arm_thread or not self.arm_thread.connected: # ZMĚNA: Kontrola arm_thread
            self.update_status(self.tr("Robotická ruka není připojena!"), is_key=False)
            return
        self.update_status(self.tr("Probíhá kalibrace... (není implementováno)"), is_key=False)
        self.logger.info("Funkce kalibrace není plně implementována v tomto zjednodušeném kódu.")


    def handle_difficulty_changed(self, value):
        if hasattr(self, 'difficulty_value_label'): self.difficulty_value_label.setText(f"{value}")
        if hasattr(self, 'strategy_selector'):
            self.strategy_selector.difficulty = value
            new_p = self.strategy_selector.p
            self.logger.info(f"Obtížnost změněna na {value}/10 -> p={new_p:.2f}")

    def handle_track_checkbox_changed(self, state):
        self.tracking_enabled = state == Qt.Checked
        if self.tracking_enabled:
            self.game_paused = True # Pozastavit hru během sledování
            self.update_status("tracking")
            self.tracking_timer.start(self.tracking_interval)
            self.track_grid_center() # Zkusit hned
            self.logger.info("Sledování hrací plochy aktivováno.")
        else:
            self.tracking_timer.stop()
            self.game_paused = False
            self.move_to_neutral_position() # Vrátit ruku
            self.update_status("your_turn") # Vrátit normální stav (předpoklad)
            self.logger.info("Sledování hrací plochy deaktivováno.")

    def track_grid_center(self):
        if not self.tracking_enabled or self.game_paused == False : return # Sledujeme jen když je aktivní A hra je pauznutá

        arm_available, _ = self._check_arm_availability()
        if not arm_available:
            self.logger.warning("Sledování: Ruka není připojena.")
            return

        _, game_state_obj = self._get_detection_data()
        if not game_state_obj or not hasattr(game_state_obj, '_grid_points') or game_state_obj._grid_points is None:
            self.logger.warning("Sledování: Mřížka není detekována.")
            return

        grid_points = game_state_obj._grid_points
        if len(grid_points) < 16: # Potřebujeme všechny body pro stabilní střed
            self.logger.warning(f"Sledování: Nedostatek bodů mřížky ({len(grid_points)}/16).")
            return

        grid_center_uv = np.mean(grid_points, axis=0) # Střed v pixelech

        # Převod na souřadnice ruky
        # Tato část vyžaduje, aby get_cell_coordinates_from_yolo byla upravena tak,
        # aby mohla přijímat přímo UV souřadnice, nebo zde implementovat transformaci.
        # Pro zjednodušení předpokládáme, že máme funkci pro transformaci UV na XY.
        # Zde použijeme zjednodušenou verzi get_cell_coordinates_from_yolo,
        # která by měla interně zvládnout transformaci.
        # Předáváme fiktivní řádek/sloupec, protože chceme transformovat grid_center_uv.

        # Dočasné řešení pro transformaci (mělo by být robustnější)
        # Zde by měla být logika z get_cell_coordinates_from_yolo pro transformaci
        target_x, target_y = self.transform_uv_to_xy_for_tracking(grid_center_uv)

        if target_x is not None and target_y is not None:
            safe_z_tracking = DEFAULT_SAFE_Z + 20 # Výše při sledování
            self.logger.info(f"Sledování: Přesun na střed mřížky XY: ({target_x:.1f}, {target_y:.1f}, Z:{safe_z_tracking})")
            self._unified_arm_command('go_to_position', x=target_x, y=target_y, z=safe_z_tracking, speed=MAX_SPEED // 4, wait=False)
        else:
            self.logger.warning("Sledování: Nepodařilo se transformovat souřadnice středu mřížky.")

    def transform_uv_to_xy_for_tracking(self, uv_coords):
        # Tato funkce je zjednodušená kopie logiky z get_cell_coordinates_from_yolo
        # pro transformaci libovolných UV souřadnic.
        if hasattr(self, 'calibration_data') and self.calibration_data and "uv_to_xy_matrix" in self.calibration_data:
            try:
                uv_to_xy_matrix = np.array(self.calibration_data["uv_to_xy_matrix"])
                uv_point = np.array([[uv_coords[0], uv_coords[1], 1.0]], dtype=np.float32).T
                xy_point = np.matmul(uv_to_xy_matrix, uv_point)
                if xy_point[2, 0] != 0:
                    return xy_point[0, 0] / xy_point[2, 0], xy_point[1, 0] / xy_point[2, 0]
            except Exception as e:
                self.logger.error(f"Chyba při transformaci UV pro sledování: {e}")
                return None, None

        # Fallback na jednodušší transformaci, pokud není matice
        self.logger.warning("Sledování: Chybí uv_to_xy_matrix, používám zjednodušenou transformaci.")
        # Zde by byla logika pro normalizované souřadnice a mapování na pracovní prostor ruky
        # Toto je velmi hrubý odhad a vyžaduje kalibraci:
        # Předpoklad: kamera vidí oblast cca 200x200mm kolem středu (200,0)
        # a obraz má např. 640x480px
        # frame_width_approx = 640
        # frame_height_approx = 480
        # arm_center_x = NEUTRAL_X
        # arm_center_y = NEUTRAL_Y
        # scale_x = 200 / frame_width_approx # mm/px
        # scale_y = 200 / frame_height_approx # mm/px
        #
        # target_x = arm_center_x + (uv_coords[0] - frame_width_approx/2) * scale_x
        # target_y = arm_center_y - (uv_coords[1] - frame_height_approx/2) * scale_y # Y osa kamery je často opačná
        # return target_x, target_y
        return None, None # Bez kalibrace je těžké toto správně implementovat


    def _unified_arm_command(self, command, *args, **kwargs):
        arm_thread_available, _ = self._check_arm_availability()
        if not arm_thread_available:
            raise RuntimeError(f"Arm command '{command}' failed: robotic arm is not available")

        # Použijeme ArmThread API
        if command == 'draw_o':
            success = self.arm_thread.draw_o(
                center_x=kwargs.get('x'),
                center_y=kwargs.get('y'),
                radius=kwargs.get('radius'),
                speed=kwargs.get('speed', DRAWING_SPEED)
            )
        elif command == 'draw_x':
            success = self.arm_thread.draw_x(
                center_x=kwargs.get('x'),
                center_y=kwargs.get('y'),
                size=kwargs.get('size'),
                speed=kwargs.get('speed', DRAWING_SPEED)
            )
        elif command == 'go_to_position':
            success = self.arm_thread.go_to_position(
                x=kwargs.get('x'),
                y=kwargs.get('y'),
                z=kwargs.get('z'),
                speed=kwargs.get('speed', MAX_SPEED),
                wait=kwargs.get('wait', True)
            )
        elif command == 'park':
            success = self.arm_thread.go_to_position(
                x=kwargs.get('x', PARK_X),
                y=kwargs.get('y', PARK_Y),
                z=kwargs.get('z', DEFAULT_SAFE_Z),
                speed=MAX_SPEED // 2,
                wait=kwargs.get('wait', True)
            )
        else:
            raise ValueError(f"Unknown arm command: {command}")

        if not success:
            raise RuntimeError(f"Arm command '{command}' failed to execute")

        self.logger.info(f"Příkaz pro ruku '{command}' proveden přes arm_thread, úspěch: {success}")
        return success


    def update_board_from_detection(self, detected_board_2d):
        # Tato metoda se volá POUZE pro vizuální aktualizaci desky v GUI na základě detekce.
        # Neměla by spouštět herní logiku.
        if not hasattr(self, 'board_widget') or not self.board_widget: return

        # Zde můžeme porovnat s self.board_widget.board a zvýraznit změny, pokud je to žádoucí.
        # Pro zjednodušení jen aktualizujeme.
        self.board_widget.update_board(detected_board_2d, self.board_widget.winning_line, highlight_changes=True)
        # self.logger.debug(f"GUI deska aktualizována z detekce: {detected_board_2d}")

    def handle_detected_game_state(self, detected_board_from_camera):
        # detected_board_from_camera je PŘEDPOKLADANĚ již 2D list nebo None
        if detected_board_from_camera is None:
            self.logger.debug("Detekována prázdná deska (None) z kamery.")
            return

        detected_board = self._convert_board_1d_to_2d(detected_board_from_camera)
        if not detected_board: # Pokud konverze selže nebo je výsledek stále None/False
            self.logger.warning("Nepodařilo se převést detekovanou desku na 2D formát.")
            return

        self.logger.debug(f"Detekovaný stav hry (po konverzi): {detected_board}")

        # POUZE aktualizuj vizuální zobrazení s tím, co YOLO skutečně detekoval
        self.update_board_from_detection(detected_board)

        if self.game_over:
            # Pokud je hra u konce, zkontrolujeme, zda není deska prázdná (signál pro novou hru)
            is_empty_now = all(cell == game_logic.EMPTY for row in detected_board for cell in row)
            if is_empty_now:
                self.logger.info("Detekována prázdná deska po konci hry - resetuji pro novou hru.")
                self.reset_game()
            return

        # ÚPLNĚ ODSTRANĚNÁ LOGIKA "VYMÝŠLENÍ" TAHŮ
        # Nyní pouze aktualizujeme board_widget.board s tím, co YOLO skutečně detekoval
        # a necháme _should_arm_play_now rozhodnout o tahu ruky na základě aktuálního stavu
        if hasattr(self, 'board_widget') and self.board_widget:
            self.board_widget.board = [row[:] for row in detected_board]

        # Zkontroluj end game podmínky na základě YOLO detekce
        self.check_game_end()

        # Aktualizuj turn management pouze na základě skutečného počtu symbolů z YOLO
        if not self.game_over:
            x_count, o_count, total_symbols = self._get_board_symbol_counts(detected_board)

            if total_symbols % 2 == 0:
                # Sudý počet = lidský hráč je na tahu
                if self.current_turn != self.human_player and not self.arm_move_in_progress:
                    self.current_turn = self.human_player
                    self.update_status("your_turn")
                    self.logger.debug(f"Turn switch: Human player turn (total symbols: {total_symbols})")
            else:
                # Lichý počet = AI/ruka je na tahu
                if self.current_turn != self.ai_player and not self.arm_move_in_progress:
                    self.current_turn = self.ai_player
                    self.update_status("arm_turn")
                    self.logger.debug(f"Turn switch: AI/arm turn (total symbols: {total_symbols})")

        # --- Hlavní logika pro rozhodnutí a spuštění tahu ruky ---
        if not self.game_over:
            should_play, arm_symbol_to_play = self._should_arm_play_now(detected_board)
            if should_play and arm_symbol_to_play:
                self.logger.info(f"ROZHODNUTÍ: Ruka by měla hrát symbolem {arm_symbol_to_play}.")
                # Nastavíme, že AI (ruka) hraje tímto symbolem
                self.ai_player = arm_symbol_to_play
                # Pokud člověk hrál X, AI je O. Pokud člověk hrál O, AI je X.
                # Toto by mělo být konzistentní s self.ai_player nastaveným na začátku.
                # Pro náš cíl "ruka hraje, když je lichý počet" je důležitější arm_symbol_to_play.

                self.current_turn = self.ai_player # Je na tahu ruka
                self.update_status("arm_moving") # Ruka se začne hýbat

                # Spustíme tah ruky
                # Předpokládáme, že self.ai_player byl správně nastaven na začátku (např. O)
                # a arm_symbol_to_play je ten, který má skutečně hrát
                self.make_arm_move_with_symbol(arm_symbol_to_play)
            elif self.current_turn == self.ai_player and not self.arm_move_in_progress and not self.waiting_for_detection:
                # Ruka by měla hrát, ale _should_arm_play_now vrátilo False (např. sudý počet, cooldown)
                # Pokud je current_turn stále ai_player, ale ruka nehraje, aktualizujeme status
                self.update_status("arm_turn") # Zobrazí "TAH RUKY", ale ruka čeká na vhodný okamžik
                self.logger.debug("Ruka je na tahu, ale podmínky pro hraní (_should_arm_play_now) nejsou splněny.")

    def _should_arm_play_now(self, current_board_state):
        self.logger.debug(f"Kontroluji, zda má ruka hrát. InProgress: {self.arm_move_in_progress}, Cooldown: {time.time() - self.last_arm_move_time < self.arm_move_cooldown}")
        if self.game_over or self.arm_move_in_progress or (time.time() - self.last_arm_move_time < self.arm_move_cooldown):
            return False, None

        # Kontrola validity mřížky
        _, game_state_obj = self._get_detection_data()
        grid_valid = False
        if game_state_obj and hasattr(game_state_obj, 'is_physical_grid_valid') and callable(game_state_obj.is_physical_grid_valid):
            grid_valid = game_state_obj.is_physical_grid_valid()

        if not grid_valid:
            self.logger.warning("Ruka nemůže hrát: mřížka není validní.")
            if not (hasattr(self, 'grid_warning_active') and self.grid_warning_active): # Aby se nezobrazovalo stále
                 self.update_status("grid_not_visible")
            return False, None

        x_count, o_count, total_symbols = self._get_board_symbol_counts(current_board_state)
        self.logger.debug(f"Analýza desky pro tah ruky: X={x_count}, O={o_count}, Celkem={total_symbols}")

        if total_symbols % 2 == 1: # Lichý počet symbolů => ruka má hrát
            # Ruka by měla hrát symbolem, kterého je na desce méně.
            # Pokud je jich stejně, a X začíná, pak O (AI) je na tahu.
            # Naše AI (ruka) je self.ai_player (např. O)
            # Symbol k zahrání by měl být self.ai_player.
            # Pokud by logika byla "ruka hraje symbol, kterého je méně":
            # arm_symbol_candidate = game_logic.PLAYER_X if x_count < o_count else game_logic.PLAYER_O
            # if x_count == o_count: # Pokud je jich stejně, a X začal, O je na tahu.
            #    arm_symbol_candidate = game_logic.PLAYER_O (pokud AI hraje za O)

            # Pro náš cíl "hrát když je lichý počet", symbolem AI (self.ai_player)
            self.logger.info(f"Lichý počet symbolů ({total_symbols}). Ruka by měla hrát za {self.ai_player}.")
            return True, self.ai_player
        else:
            self.logger.debug(f"Sudý počet symbolů ({total_symbols}). Ruka nehraje.")
            return False, None


    def make_arm_move_with_symbol(self, symbol_to_play):
        self.logger.info(f"Spouštím tah ruky se symbolem: {symbol_to_play}")
        if self.game_over or self.arm_move_in_progress: # Přidána kontrola arm_move_in_progress
            self.logger.warning("Tah ruky přerušen: hra skončila nebo ruka je již v pohybu.")
            return False

        # Získání aktuální desky z kamery pro strategii
        current_board_for_strategy = None
        if hasattr(self, 'camera_thread') and self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
            current_board_for_strategy = self._convert_board_1d_to_2d(self.camera_thread.last_board_state)

        if not current_board_for_strategy:
            self.logger.error("Nelze provést tah ruky: není dostupný aktuální stav desky pro strategii.")
            # Možná resetovat stav, aby se hra mohla obnovit?
            self.current_turn = self.human_player # Vrať tah hráči
            self.update_status("your_turn")
            return False

        # Nastavení příznaků PŘED zahájením pohybu
        self.arm_move_in_progress = True
        self.last_arm_move_time = time.time()
        self.update_status("arm_moving") # Zobraz, že se ruka pohybuje

        move = self.strategy_selector.get_move(current_board_for_strategy, symbol_to_play)
        if not move:
            self.logger.warning(f"Strategie nenašla platný tah pro symbol {symbol_to_play}.")
            self.arm_move_in_progress = False # Uvolnit ruku
            self.current_turn = self.human_player # Něco je špatně, vrátit tah hráči
            self.update_status("your_turn")
            return False

        row, col = move
        self.logger.info(f"AI strategie vybrala tah: ({row}, {col}) pro symbol {symbol_to_play}")

        # Uložit informace o očekávaném tahu pro detekci
        self.ai_move_row = row
        self.ai_move_col = col
        self.expected_symbol = symbol_to_play
        self.detection_wait_time = 0.0 # Resetovat časovač čekání na detekci
        self.ai_move_retry_count = 0   # Resetovat počítadlo opakování

        if self.draw_ai_symbol(row, col, symbol_to_play):
            self.logger.info(f"Symbol {symbol_to_play} úspěšně odeslán ke kreslení na ({row},{col}). Čekám na detekci.")
            self.waiting_for_detection = True
            # arm_move_in_progress zůstává True, dokud není detekce potvrzena (nebo timeout)
            return True
        else:
            self.logger.error(f"Nepodařilo se zahájit kreslení symbolu {symbol_to_play} na ({row},{col}).")
            self.arm_move_in_progress = False # Uvolnit ruku
            self.waiting_for_detection = False # Nebudeme čekat na detekci
            self.current_turn = self.human_player # Vrátit tah hráči
            self.update_status("your_turn")
            return False


    def update_game_state_machine(self): # Dříve update_game_state
        if self.game_paused or self.game_over : return # Pokud je hra pauznutá nebo skončila, nic nedělat

        if hasattr(self, 'grid_warning_active') and self.grid_warning_active:
            return # Pokud je problém s mřížkou, neaktualizovat logiku hry

        if self.waiting_for_detection:
            self.detection_wait_time += 0.1 # Timer je každých 100ms

            # Kontrola detekce symbolu
            board_after_arm_move = None
            if hasattr(self, 'camera_thread') and self.camera_thread and hasattr(self.camera_thread, 'last_board_state'):
                board_after_arm_move = self._convert_board_1d_to_2d(self.camera_thread.last_board_state)

            if board_after_arm_move and \
               self.ai_move_row is not None and self.ai_move_col is not None and \
               0 <= self.ai_move_row < 3 and 0 <= self.ai_move_col < 3 and \
               board_after_arm_move[self.ai_move_row][self.ai_move_col] == self.expected_symbol:

                self.logger.info(f"ÚSPĚŠNÁ DETEKCE: Symbol {self.expected_symbol} na ({self.ai_move_row},{self.ai_move_col}).")
                self.waiting_for_detection = False
                self.arm_move_in_progress = False # Ruka dokončila pohyb a byl detekován

                self.board_widget.board = [r[:] for r in board_after_arm_move] # Aktualizovat interní desku widgetu
                self.board_widget.update() # Překreslit GUI desku

                self.move_counter += 1
                self.check_game_end()

                if not self.game_over:
                    self.current_turn = self.human_player
                    self.update_status("your_turn")

                self.ai_move_row, self.ai_move_col, self.expected_symbol = None, None, None
                self.ai_move_retry_count = 0
                self.detection_wait_time = 0.0
                self.move_to_neutral_position() # Po úspěšné detekci do neutrálu

            elif self.detection_wait_time >= self.max_detection_wait_time:
                self.logger.warning(f"TIMEOUT DETEKCE: Symbol {self.expected_symbol} na ({self.ai_move_row},{self.ai_move_col}) nebyl detekován včas.")

                if self.ai_move_retry_count < self.max_retry_count:
                    self.ai_move_retry_count += 1
                    self.logger.info(f"Opakuji kreslení, pokus {self.ai_move_retry_count}/{self.max_retry_count}.")
                    self.update_status(self.tr("detection_attempt").format(self.ai_move_retry_count, self.max_retry_count), is_key=False)
                    self.detection_wait_time = 0.0 # Resetovat časovač pro další pokus
                    # Příznak arm_move_in_progress je stále True, takže nový tah se nespustí, dokud tento nedoběhne
                    # Musíme ho na chvíli uvolnit, aby se mohl spustit draw_ai_symbol
                    self.arm_move_in_progress = False
                    if not self.draw_ai_symbol(self.ai_move_row, self.ai_move_col, self.expected_symbol):
                        # Pokud ani opakované kreslení nelze zahájit
                        self.logger.error("Opakované kreslení selhalo.")
                        self.waiting_for_detection = False
                        self.arm_move_in_progress = False
                        self.current_turn = self.human_player
                        self.update_status("your_turn")
                    else:
                        self.arm_move_in_progress = True # Znovu nastavit, protože kreslení začalo
                else:
                    self.logger.error("Maximum pokusů o detekci dosaženo. Vzdávám tah ruky.")
                    self.waiting_for_detection = False
                    self.arm_move_in_progress = False
                    self.current_turn = self.human_player # Vzdát to a nechat hrát člověka
                    self.update_status("detection_failed")
                    QTimer.singleShot(2000, lambda: self.update_status("your_turn")) # Po chvíli vrátit na tah hráče

                self.ai_move_row, self.ai_move_col, self.expected_symbol = None, None, None # Po timeoutu/vyčerpání pokusů zapomenout
        # else:
            # Zde by mohla být logika pro případ, kdy není self.waiting_for_detection,
            # ale to se nyní řeší v handle_detected_game_state přes _should_arm_play_now
            # pass


    def draw_ai_symbol(self, row, col, symbol_to_draw):
        self.logger.info(f"Požadavek na kreslení {symbol_to_draw} na ({row},{col})")
        arm_available, _ = self._check_arm_availability()
        if not arm_available:
            raise RuntimeError(f"Cannot draw symbol {symbol_to_draw}: robotic arm is not available")

        # Získání souřadnic z YOLO a kalibrace
        target_x, target_y = self.get_cell_coordinates_from_yolo(row, col)
        if target_x is None or target_y is None:
            raise RuntimeError(f"Cannot get coordinates for drawing at ({row},{col})")

        self.logger.info(f"Kreslím {symbol_to_draw} na fyzické souřadnice ({target_x:.1f}, {target_y:.1f})")

        # Parametry pro kreslení z kalibrace nebo výchozí
        draw_z = self.calibration_data.get("draw_z", DEFAULT_DRAW_Z) if hasattr(self, 'calibration_data') else DEFAULT_DRAW_Z
        safe_z = self.calibration_data.get("safe_z", DEFAULT_SAFE_Z) if hasattr(self, 'calibration_data') else DEFAULT_SAFE_Z
        symbol_size = self.calibration_data.get("symbol_size_mm", DEFAULT_SYMBOL_SIZE_MM) if hasattr(self, 'calibration_data') else DEFAULT_SYMBOL_SIZE_MM

        # Sestavení příkazu pro _unified_arm_command
        # Předpokládáme, že ArmThread má metody draw_o a draw_x
        if symbol_to_draw == game_logic.PLAYER_O:
            success = self._unified_arm_command('draw_o', x=target_x, y=target_y, radius=symbol_size / 2,
                                              draw_z=draw_z, safe_z=safe_z, speed=DRAWING_SPEED)
        elif symbol_to_draw == game_logic.PLAYER_X:
            success = self._unified_arm_command('draw_x', x=target_x, y=target_y, size=symbol_size,
                                              draw_z=draw_z, safe_z=safe_z, speed=DRAWING_SPEED)
        else:
            self.logger.error(f"Neznámý symbol pro kreslení: {symbol_to_draw}")
            return False

        if success:
            self.logger.info(f"Příkaz ke kreslení {symbol_to_draw} odeslán.")
            # Přesun do neutrální pozice se děje až po detekci, ne hned po kreslení
            # self.move_to_neutral_position() # Toto se přesune
        else:
            self.logger.error(f"Odeslání příkazu ke kreslení {symbol_to_draw} selhalo.")

        return success


    def get_cell_coordinates_from_yolo(self, row, col):
        """
        Získá skutečné XY souřadnice robotické ruky pro danou buňku (row, col)
        na základě aktuální detekce mřížky a kalibrace.
        """
        # Získej aktuální stav detekce
        _, game_state_obj = self._get_detection_data()
        if not game_state_obj:
            raise RuntimeError(f"Cannot get detection state for cell ({row},{col})")

        # Získej UV souřadnice středu buňky z aktuální detekce
        uv_center = game_state_obj.get_cell_center_uv(row, col)
        if uv_center is None:
            raise RuntimeError(f"Cannot get UV center for cell ({row},{col}) from current detection")

        # Transformuj UV souřadnice na XY pomocí inverzní transformace
        if hasattr(self, 'calibration_data') and self.calibration_data:
            # Kalibrace obsahuje xy_to_uv matici, potřebujeme inverzní
            xy_to_uv_matrix = self.calibration_data.get("perspective_transform_matrix_xy_to_uv")
            if xy_to_uv_matrix:
                try:
                    # Inverze matice pro UV->XY transformaci
                    xy_to_uv_matrix = np.array(xy_to_uv_matrix, dtype=np.float32)
                    uv_to_xy_matrix = np.linalg.inv(xy_to_uv_matrix)

                    # Homogenní souřadnice pro transformaci
                    uv_point_homogeneous = np.array([uv_center[0], uv_center[1], 1.0], dtype=np.float32).reshape(3,1)
                    xy_transformed_homogeneous = np.dot(uv_to_xy_matrix, uv_point_homogeneous)

                    if xy_transformed_homogeneous[2,0] != 0:
                        arm_x = xy_transformed_homogeneous[0,0] / xy_transformed_homogeneous[2,0]
                        arm_y = xy_transformed_homogeneous[1,0] / xy_transformed_homogeneous[2,0]
                        self.logger.info(f"Transformované UV {uv_center} na XY ({arm_x:.1f}, {arm_y:.1f}) pro buňku ({row},{col})")
                        return arm_x, arm_y
                    else:
                        raise RuntimeError("Division by zero in UV->XY transformation")

                except Exception as e:
                    raise RuntimeError(f"Error in UV->XY transformation for ({row},{col}): {e}")
            else:
                raise RuntimeError("Missing transformation matrix in calibration data")
        else:
            raise RuntimeError("Missing calibration data")


    def check_game_end(self):
        if self.game_over: return True # Již vyhodnoceno

        # Kontrola výherce na základě aktuálního stavu GUI desky
        # Mělo by se ideálně kontrolovat na základě `camera_thread.last_board_state` pro přesnost,
        # ale pro GUI reakci může být `self.board_widget.board` dostačující, pokud je synchronizovaná.

        board_to_check = self.board_widget.board # Použijeme GUI desku pro konzistenci s tím, co vidí uživatel
        if hasattr(self, 'camera_thread') and self.camera_thread.last_board_state:
            # Pokud máme čerstvá data z kamery, preferujeme je
            board_from_cam = self._convert_board_1d_to_2d(self.camera_thread.last_board_state)
            if board_from_cam: board_to_check = board_from_cam

        self.winner = game_logic.check_winner(board_to_check)

        if self.winner:
            self.game_over = True
            self.logger.info(f"KONEC HRY! Vítěz: {self.winner}. Počet tahů: {self.move_counter}")

            # Resetovat příznaky ruky, protože hra skončila
            self.arm_move_in_progress = False
            self.waiting_for_detection = False

            if hasattr(self, '_celebration_triggered'): del self._celebration_triggered
            self.show_game_end_notification() # Zobrazí výsledek

            if self.winner != game_logic.TIE:
                self.board_widget.winning_line = game_logic.get_winning_line(board_to_check)
                self.board_widget.update()
                # Případná oslava ruky, pokud AI vyhrála
                if self.winner == self.ai_player:
                    arm_available, _ = self._check_arm_availability()
                    if arm_available:
                        self.logger.info("AI (ruka) vyhrála! Plánuji kreslení výherní čáry.")
                        QTimer.singleShot(1500, self.draw_winning_line) # Malé zpoždění pro efekt

            if self.winner == game_logic.TIE: self.update_status("draw")
            else: self.update_status("win") # Text se upraví v show_game_end_notification a update_status

            self.move_to_neutral_position() # Po konci hry do neutrálu
            return True

        # Kontrola, zda je deska plná (remíza, pokud ještě nebyl vítěz)
        if self.move_counter >= 9 and not self.winner : # Všechna políčka zaplněna
            self.game_over = True
            self.winner = game_logic.TIE # Explicitně nastavit remízu
            self.logger.info(f"KONEC HRY! Remíza. Počet tahů: {self.move_counter}")
            self.arm_move_in_progress = False
            self.waiting_for_detection = False
            if hasattr(self, '_celebration_triggered'): del self._celebration_triggered
            self.show_game_end_notification()
            self.update_status("draw")
            self.move_to_neutral_position()
            return True

        return False

    def show_game_end_notification(self):
        if hasattr(self, '_celebration_triggered'): return # Již zobrazeno
        self._celebration_triggered = True

        notification_widget = QWidget(self)
        notification_widget.setObjectName("game_end_notification")
        # ... (zbytek kódu pro styl a obsah notifikace zůstává stejný)
        notification_widget.setStyleSheet("""
            #game_end_notification {
                background-color: rgba(45, 45, 48, 0.95); /* Tmavší s průhledností */
                border-radius: 15px;
                border: 2px solid #0078D7; /* Modrý okraj */
            }
        """)
        layout = QVBoxLayout(notification_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        icon_text, message_text, color = "", "", ""
        if self.winner == game_logic.TIE:
            icon_text, message_text, color = "🤝", self.tr("draw"), "#f1c40f"
        elif self.winner == self.human_player:
            icon_text, message_text, color = "🏆", self.tr("win"), "#2ecc71"
        elif self.winner == self.ai_player:
            icon_text, message_text, color = "🤖", f"{self.tr('ai_turn')} {self.tr('win')}", "#3498db" # Použijeme AI barvu
        else: # Hra skončila, ale není jasný výherce (nemělo by nastat)
            icon_text, message_text, color = "🏁", self.tr("game_over"), "#95a5a6"

        icon_label = QLabel(icon_text)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"font-size: 60px; color: {color};")
        layout.addWidget(icon_label)

        message_label = QLabel(message_text.upper())
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color};")
        layout.addWidget(message_label)

        instruction_label = QLabel(self.tr("Pro novou hru vymažte hrací plochu nebo stiskněte Reset."))
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("font-size: 12px; color: #bdc3c7; margin-top: 10px;") # Světle šedá
        layout.addWidget(instruction_label)

        notification_widget.setFixedSize(350, 220)
        notification_widget.move(
            (self.width() - notification_widget.width()) // 2,
            (self.height() - notification_widget.height()) // 2
        )

        opacity_effect = QGraphicsOpacityEffect(notification_widget)
        notification_widget.setGraphicsEffect(opacity_effect)

        notification_widget.show()
        notification_widget.raise_()

        anim = QPropertyAnimation(opacity_effect, b"opacity")
        anim.setDuration(500)
        anim.setStartValue(0)
        anim.setEndValue(1)
        anim.start(QPropertyAnimation.DeleteWhenStopped) # Automaticky smaže animaci

        # Uložení reference na widget, aby nebyl smazán GC, a timer pro jeho skrytí
        self._active_notification = notification_widget
        QTimer.singleShot(4000, lambda: self._active_notification.hide() if hasattr(self, '_active_notification') and self._active_notification else None)


    def draw_winning_line(self):
        self.logger.info("Pokus o kreslení výherní čáry.")
        if not self.board_widget.winning_line or len(self.board_widget.winning_line) != 3:
            raise RuntimeError("Cannot draw winning line: missing or invalid data")

        arm_available, _ = self._check_arm_availability()
        if not arm_available:
            raise RuntimeError("Cannot draw winning line: robotic arm is not connected")

        start_pos_rc = self.board_widget.winning_line[0] # (row, col)
        end_pos_rc = self.board_widget.winning_line[2]   # (row, col)

        start_xy = self.get_cell_coordinates_from_yolo(start_pos_rc[0], start_pos_rc[1])
        end_xy = self.get_cell_coordinates_from_yolo(end_pos_rc[0], end_pos_rc[1])

        # Note: get_cell_coordinates_from_yolo now raises exceptions instead of returning None

        start_x, start_y = start_xy
        end_x, end_y = end_xy

        draw_z = self.calibration_data.get("draw_z", DEFAULT_DRAW_Z) if hasattr(self, 'calibration_data') else DEFAULT_DRAW_Z
        safe_z = self.calibration_data.get("safe_z", DEFAULT_SAFE_Z) if hasattr(self, 'calibration_data') else DEFAULT_SAFE_Z

        self.logger.info(f"Kreslím výherní čáru z ({start_x:.1f},{start_y:.1f}) do ({end_x:.1f},{end_y:.1f})")
        self.update_status(self.tr("Kreslím výherní čáru..."), is_key=False)

        # Sekvence pohybů
        # 1. Přesun nad start_xy v safe_z
        if not self._unified_arm_command('go_to_position', x=start_x, y=start_y, z=safe_z, speed=MAX_SPEED, wait=True):
            raise RuntimeError("Failed to move arm to start position")
        # 2. Spuštění na draw_z
        if not self._unified_arm_command('go_to_position', x=start_x, y=start_y, z=draw_z, speed=DRAWING_SPEED, wait=True):
            raise RuntimeError("Failed to lower arm to drawing position")
        # 3. Přesun na end_xy v draw_z (kreslení)
        if not self._unified_arm_command('go_to_position', x=end_x, y=end_y, z=draw_z, speed=DRAWING_SPEED, wait=True):
            raise RuntimeError("Failed to draw winning line")
        # 4. Zvednutí na safe_z
        if not self._unified_arm_command('go_to_position', x=end_x, y=end_y, z=safe_z, speed=MAX_SPEED, wait=True):
            raise RuntimeError("Failed to lift arm after drawing")

        self.logger.info("Výherní čára úspěšně nakreslena.")
        self.update_status(self.tr("Výherní čára nakreslena!"), is_key=False)
        QTimer.singleShot(1000, self.move_to_neutral_position) # Po chvíli do neutrálu
        return True


    def show_debug_window(self):
        if not self.debug_window:
            self.debug_window = DebugWindow(config=self.config, parent=self)
            # Připojení signálů pro debug okno
            if hasattr(self.camera_thread, 'fps_updated'): # FPS se nyní posílá přímo z camera_thread
                self.camera_thread.fps_updated.connect(self.debug_window.update_fps)
            # Pro aktualizaci desky v debug okně můžeme použít game_state_updated z camera_thread
            # nebo posílat data přímo z update_camera_view
            # self.camera_thread.game_state_updated.connect(lambda board_state: self.debug_window.update_board_state(self._convert_board_1d_to_2d(board_state)))
            # Přepínání kamery se řeší interně v DebugWindow, které volá self.handle_camera_changed
            if hasattr(self.debug_window, 'camera_changed_signal'): # Pokud DebugWindow emituje signál
                 self.debug_window.camera_changed_signal.connect(self.handle_camera_changed)

        self.debug_window.show()
        self.debug_window.activateWindow()


    def handle_camera_changed(self, camera_index):
        self.logger.info(f"Požadavek na změnu kamery na index: {camera_index}")
        if hasattr(self, 'camera_thread') and self.camera_thread:
            if self.camera_thread.camera_index == camera_index and self.camera_thread.isRunning():
                self.logger.info(f"Kamera {camera_index} je již aktivní.")
                return

            self.logger.info("Zastavuji stávající vlákno kamery...")
            self.camera_thread.stop() # Metoda stop by měla čistě ukončit vlákno
            self.camera_thread.wait(2000) # Počkat na doběhnutí
            if self.camera_thread.isRunning():
                self.logger.warning("Nepodařilo se čistě zastavit vlákno kamery, terminuji.")
                self.camera_thread.terminate() # Tvrdé ukončení, pokud stop selže
                self.camera_thread.wait(500)


        self.logger.info(f"Vytvářím nové vlákno kamery s indexem {camera_index}.")
        self.camera_thread = CameraThread(camera_index=camera_index)
        self.camera_thread.frame_ready.connect(self.update_camera_view)
        self.camera_thread.game_state_updated.connect(self.handle_detected_game_state)
        self.camera_thread.fps_updated.connect(self.update_fps_display)
        self.camera_thread.start()
        self.logger.info(f"Nové vlákno kamery pro index {camera_index} spuštěno.")


    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # Stylesheet (zkráceno pro přehlednost, předpokládáme původní)
        self.setStyleSheet(""" /* ... Váš původní stylesheet ... */
            QWidget { background-color: #2D2D30; color: #E0E0E0; font-family: 'Segoe UI', Arial, sans-serif; }
            QPushButton { background-color: #0078D7; color: white; border: none; padding: 8px 15px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #1084E3; } QPushButton:pressed { background-color: #0067B8; }
            QLabel { color: #E0E0E0; } QSlider::groove:horizontal { border: 1px solid #999999; height: 8px; background: #3D3D3D; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #0078D7; border: 1px solid #0078D7; width: 18px; margin: -6px 0; border-radius: 9px; }
        """)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20,20,20,20); main_layout.setSpacing(15)

        self.main_status_panel = QWidget()
        # Styl se nastavuje dynamicky v update_status přes _get_status_style
        status_layout = QVBoxLayout(self.main_status_panel)
        self.main_status_message = QLabel("START") # Výchozí zpráva
        self.main_status_message.setStyleSheet("color: #FFFFFF; font-size: 28px; font-weight: bold; padding: 12px;")
        self.main_status_message.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.main_status_message)
        main_layout.addWidget(self.main_status_panel)

        board_container = QWidget()
        board_container.setStyleSheet("background-color: #333740; border-radius: 10px; padding: 10px;")
        board_layout = QHBoxLayout(board_container)
        board_layout.addStretch(1)
        self.board_widget = TicTacToeBoard() # Předpokládá, že TicTacToeBoard je správně importován
        self.board_widget.cell_clicked.connect(self.handle_cell_clicked)
        self.board_widget.setMinimumSize(400, 400) # Mírně menší pro testování
        board_layout.addWidget(self.board_widget)
        board_layout.addStretch(1)
        main_layout.addWidget(board_container, 1) # Board zabere většinu místa

        # --- Ovládací panel ---
        controls_panel = QWidget()
        controls_panel.setStyleSheet("background-color: #333740; border-radius: 10px; padding: 15px;")
        controls_layout = QVBoxLayout(controls_panel)

        # Obtížnost
        difficulty_container = QWidget()
        difficulty_layout = QHBoxLayout(difficulty_container)
        self.difficulty_label = QLabel(self.tr("difficulty"))
        self.difficulty_slider = QSlider(Qt.Horizontal)
        self.difficulty_slider.setRange(0,10); self.difficulty_slider.setValue(DEFAULT_DIFFICULTY)
        self.difficulty_slider.valueChanged.connect(self.handle_difficulty_changed)
        self.difficulty_value_label = QLabel(f"{DEFAULT_DIFFICULTY}")
        difficulty_layout.addWidget(self.difficulty_label)
        difficulty_layout.addWidget(self.difficulty_slider,1)
        difficulty_layout.addWidget(self.difficulty_value_label)
        controls_layout.addWidget(difficulty_container)

        # Tlačítka
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        self.reset_button = QPushButton(self.tr("new_game"))
        self.reset_button.clicked.connect(self.handle_reset_button_click)
        self.reset_button.setStyleSheet("background-color: #27ae60; /* ... další styly ... */")

        self.language_button = QPushButton("🇨🇿")
        self.language_button.clicked.connect(self.change_language)
        self.language_button.setFixedSize(40,40)

        self.debug_button = QPushButton(self.tr("debug")) # Nebo ikona ⚙️
        self.debug_button.clicked.connect(self.handle_debug_button_click)
        self.debug_button.setFixedSize(40,40)

        self.track_checkbox = QCheckBox(self.tr("tracking")) # Text místo "Track"
        self.track_checkbox.stateChanged.connect(self.handle_track_checkbox_changed)

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.track_checkbox)
        button_layout.addWidget(self.language_button)
        button_layout.addWidget(self.debug_button)
        controls_layout.addWidget(button_container)
        main_layout.addWidget(controls_panel)

        # Starý status_label (skrytý, pro případnou zpětnou kompatibilitu s některými funkcemi)
        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        # main_layout.addWidget(self.status_label) # Ne přidávat do layoutu, pokud je skrytý

        self.reset_status_panel_style()  # Nastavit výchozí styl panelu



    def init_game_components(self):
        self.calibration_data = self.load_calibration()
        draw_z = self.calibration_data.get("draw_z", DEFAULT_DRAW_Z)
        safe_z = self.calibration_data.get("safe_z", DEFAULT_SAFE_Z)

        arm_port = self.config.arm_controller.port if hasattr(self.config, 'arm_controller') else None
        if not arm_port:
            self.logger.warning("Port pro ArmThread není konfigurován.")

        self.arm_thread = ArmThread(port=arm_port) # Může selhat, pokud port není None a nevalidní
        self.arm_thread.start() # Spustit vlákno pro zpracování příkazů

        # Připojení k ruce
        if self.arm_thread.connect(): # connect() by mělo vracet True/False
            self.logger.info("Robotická ruka úspěšně připojena přes ArmThread.")
            self.move_to_neutral_position()
        else:
            self.logger.error("Nepodařilo se připojit k robotické ruce přes ArmThread.")
            # Zde by se mohlo zobrazit varování uživateli

        self.arm_controller = ArmController(port=arm_port, draw_z=draw_z, safe_z=safe_z, speed=MAX_SPEED)
        self.arm_controller.connected = self.arm_thread.connected


    def load_calibration(self):
        if not os.path.exists(CALIBRATION_FILE):
            raise FileNotFoundError(f"Required calibration file not found: {CALIBRATION_FILE}")

        with open(CALIBRATION_FILE, 'r') as f:
            data = json.load(f)
        self.logger.info(f"Kalibrace úspěšně načtena z {CALIBRATION_FILE}.")
        return data

    def move_to_neutral_position(self):
        neutral_pos_cfg = self.calibration_data.get("neutral_position", {}) if hasattr(self, 'calibration_data') else {}
        x = neutral_pos_cfg.get("x", NEUTRAL_X)
        y = neutral_pos_cfg.get("y", NEUTRAL_Y)
        z = neutral_pos_cfg.get("z", NEUTRAL_Z)

        self.logger.info(f"Přesouvám ruku do neutrální pozice ({x}, {y}, {z})")
        # self.update_status(self.tr("move_to_neutral"), is_key=False) # Může být příliš časté
        success = self._unified_arm_command('go_to_position', x=x, y=y, z=z, speed=MAX_SPEED, wait=False) # wait=False pro rychlejší UI

        if success:
            self.logger.info("Ruka úspěšně odeslána do neutrální pozice.")
            # self.update_status(self.tr("move_success"), is_key=False)
            # QTimer.singleShot(2000, self.reset_status_panel_style)
        else:
            self.logger.warning("Nepodařilo se odeslat příkaz pro přesun do neutrální pozice.")
            # self.update_status(self.tr("move_failed"), is_key=False)
        return success


    def closeEvent(self, event):
        self.logger.info("Zavírám aplikaci...")
        if hasattr(self, 'tracking_timer'): self.tracking_timer.stop()
        if hasattr(self, 'update_timer'): self.update_timer.stop()

        if hasattr(self, 'camera_thread') and self.camera_thread:
            self.logger.info("Zastavuji vlákno kamery...")
            self.camera_thread.stop()
            self.camera_thread.wait(1000)

        if hasattr(self, 'arm_thread') and self.arm_thread:
            self.logger.info("Parkuji a odpojuji ruku...")
            self.park_arm() # Počká na dokončení
            self.arm_thread.disconnect()
            self.arm_thread.stop() # Pokud má ArmThread vlastní stop metodu
            self.arm_thread.wait(500)


        if hasattr(self, 'debug_window') and self.debug_window:
            self.debug_window.close()

        self.logger.info("Aplikace ukončena.")
        event.accept()


if __name__ == "__main__":
    # Základní konfigurace loggeru, pokud není nastavena jinde
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'  # noqa: E501
        )

    app = QApplication(sys.argv)
    # Můžete předat vlastní AppConfig() instanci, pokud je potřeba
    # default_config = AppConfig()
    window = TicTacToeApp(config=None)  # Použije AppConfig() interně
    window.show()
    sys.exit(app.exec_())