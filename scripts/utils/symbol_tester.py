# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
# test_symbols.py
import logging
import sys  # Vrácen import pro sys.exit(1)
import cv2
from pynput import keyboard
from typing import Optional, Tuple, Dict
import numpy as np
import torch
from ultralytics import YOLO
import json

# --- Import ArmController ---
try:
    # Předpokládáme, že je ve stejném adresáři nebo v PYTHONPATH
    from arm_controller import ArmController
except ImportError:
    print("ERROR: arm_controller.py not found.")
    print("       Ensure it's in the same directory or accessible.")
    sys.exit(1)

# --- Konfigurace (převzato/upraveno z calibration.py) ---
POSE_MODEL_PATH = "weights/best_pose.pt"  # !!! UPRAVTE CESTU dle potřeby !!!
CALIBRATION_FILE = "hand_eye_calibration.json"  # Soubor s kalibrací
CAM_INDEX = 0  # Index kamery
WINDOW_NAME = "Symbol Test"
# Prahové hodnoty pro detekci mřížky
POSE_CONF_THRESHOLD = 0.8
KEYPOINT_VISIBLE_THRESHOLD = 0.5
MIN_POINTS_FOR_RANSAC = 6  # Minimum bodů pro RANSAC korekci mřížky
GRID_POINTS_COUNT = 16  # Očekávaný počet bodů mřížky

# Barvy a font pro kreslení
GRID_COLOR = (0, 255, 0)  # Zelená pro mřížku
CENTER_CELL_COLOR = (0, 0, 255)  # Červená pro středovou buňku (pro ladění)
SYMBOL_COLOR = (255, 0, 0)  # Modrá pro symbol X/O
SYMBOL_FONT = cv2.FONT_HERSHEY_SIMPLEX
SYMBOL_SCALE = 2
SYMBOL_THICKNESS = 3
POINT_RADIUS = 3
SYMBOL_SIZE_MM = 40.0  # Velikost kresleného symbolu v mm (LADIT!)

# Klávesy
KEY_DRAW_X = 'x'
KEY_DRAW_O = 'o'
KEY_QUIT = keyboard.Key.esc

# --- Globální Stav ---
pose_model: Optional[YOLO] = None
device: Optional[str] = None
cap: Optional[cv2.VideoCapture] = None
running = True
# symbol_to_draw: Optional[str] = None # Nahrazeno draw_request
draw_request: Optional[str] = None  # 'X' nebo 'O' - požadavek na kreslení
center_cell_uv: Optional[Tuple[int, int]] = None  # UV souřadnice středu
# XY souřadnice robota pro střed
center_cell_xy: Optional[Tuple[float, float]] = None
# Transformační matice UV -> XY
transform_uv_to_xy: Optional[np.ndarray] = None
safe_z: Optional[float] = None  # Bezpečná Z výška z kalibrace
draw_z: Optional[float] = None  # Kreslící Z výška z kalibrace
controller: Optional[ArmController] = None  # Instance ArmController

# --- Funkce pro korekci bodů (zkopírováno z calibration.py) ---


def correct_grid_points_homography(
        predicted_kpts_data,
        confidence_threshold=KEYPOINT_VISIBLE_THRESHOLD,
        min_points_for_ransac=MIN_POINTS_FOR_RANSAC):
    """Opraví pozice keypointů mřížky pomocí odhadu homografie a RANSAC."""
    correction_logger = logging.getLogger(
        f"{__name__}.correct_grid_homography"
    )
    valid_mask = predicted_kpts_data[:, 2] > confidence_threshold
    valid_indices = np.where(valid_mask)[0]
    valid_predicted_pts = predicted_kpts_data[valid_mask, :2].astype(
        np.float32)
    num_valid = len(valid_indices)
    correction_logger.debug(
        f"Nalezeno {num_valid} validních bodů mřížky nad prahem "
        f"{confidence_threshold}."
    )
    min_req_points = max(4, min_points_for_ransac)
    if num_valid < min_req_points:
        correction_logger.warning(
            f"Nedostatek validních bodů ({num_valid} < {min_req_points}) "
            f"pro Homografii mřížky. Nelze opravit."
        )
        return None
    # Ideální mřížka 4x4 body (indexy 0-15)
    ideal_grid_all = np.array([(i % 4, i // 4) for i in range(GRID_POINTS_COUNT)],
                              dtype=np.float32)
    valid_ideal_pts = ideal_grid_all[valid_indices]
    try:
        homography_matrix, ransac_mask = cv2.findHomography(
            valid_ideal_pts, valid_predicted_pts, method=cv2.RANSAC,
            ransacReprojThreshold=10.0)  # Použijeme stejný práh jako v kalibraci
        if homography_matrix is None:
            correction_logger.warning("RANSAC selhal při hledání "
                                      "homografie mřížky.")
            return None
        num_inliers = np.sum(ransac_mask) if ransac_mask is not None else 0
        correction_logger.debug(
            f"Homografie mřížky RANSAC nalezla matici s {num_inliers} inliery."
        )
        if num_inliers < min_req_points:
            correction_logger.warning(
                f"Homografie mřížky RANSAC nalezla příliš málo inlierů "
                f"({num_inliers})."
            )
            return None
    except cv2.error as e:
        correction_logger.error(
            f"OpenCV chyba findHomography pro mřížku: {e}.")
        return None
    except Exception as e:
        correction_logger.exception("Neočekávaná chyba findHomography "
                                    f"pro mřížku: {e}.")
        return None
    # Aplikujeme transformaci na všechny ideální body
    ideal_grid_all_reshaped = ideal_grid_all.reshape(-1, 1, 2)
    corrected_pts_xy = cv2.perspectiveTransform(
        ideal_grid_all_reshaped, homography_matrix
    ).reshape(-1, 2)  # Oprava reshape na správný tvar (N, 2)
    correction_logger.debug("Korekce bodů mřížky homografií úspěšná.")
    # Funkce by měla nyní vždy vracet (16, 2) nebo None, pokud selhala dříve
    return corrected_pts_xy

# --- Funkce pro načtení kalibrace ---


def load_calibration(filename: str) -> Optional[Dict]:
    """Načte kalibrační data ze JSON souboru."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        logging.info("Kalibrace úspěšně načtena z %s", filename)
        # Ověření potřebných klíčů
        required_keys = [
            "perspective_transform_matrix_xy_to_uv",
            "touch_z",
            "safe_z",
            "calibration_points_raw"]
        if not all(key in data for key in required_keys):
            logging.error("Chybí potřebné klíče v kalibračním souboru.")
            return None
        # Převedení matice zpět na numpy array
        data["perspective_transform_matrix_xy_to_uv"] = np.array(
            data["perspective_transform_matrix_xy_to_uv"])
        return data
    except FileNotFoundError:
        logging.error("CHYBA: Kalibrační soubor '%s' nenalezen!", filename)
        logging.error("Spusťte nejprve calibration.py")
        return None
    except json.JSONDecodeError:
        logging.error(
            "CHYBA: Nepodařilo se zpracovat '%s'. Neplatný JSON?", filename)
        return None
    except Exception as e:
        logging.error("CHYBA: Neočekávaná chyba při načítání kalibrace: %s", e)
        return None

# --- Funkce pro výpočet inverzní transformace ---


def calculate_uv_to_xy_transform(
        calibration_data: Dict) -> Optional[np.ndarray]:
    """Vypočítá transformační matici UV -> XY z kalibračních bodů."""
    if not calibration_data or "calibration_points_raw" not in calibration_data:
        logging.error("Chybí data pro výpočet UV->XY transformace.")
        return None

    raw_points = calibration_data["calibration_points_raw"]
    min_points = 4  # Stejné minimum jako pro kalibraci
    if len(raw_points) < min_points:
        logging.error(
            f"Nedostatek bodů ({len(raw_points)} < {min_points}) "
            f"v kalibraci pro výpočet UV->XY.")
        return None

    # Zdrojové body: UV souřadnice obrazu
    src_pts_uv = np.array([p["target_uv"]
                          for p in raw_points], dtype=np.float32)
    # Cílové body: XY souřadnice ramene
    dst_pts_xy = np.array([p["robot_xyz"][:2]
                          for p in raw_points], dtype=np.float32)

    # Výpočet homografie UV -> XY
    try:
        transform_matrix, inliers = cv2.findHomography(
            src_pts_uv, dst_pts_xy,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0)  # Stejný RANSAC práh

        if transform_matrix is None:
            logging.error(
                "Nepodařilo se vypočítat UV->XY transformaci (findHomography selhal).")
            return None

        num_inliers = np.sum(inliers) if inliers is not None else 0
        logging.info(
            "UV->XY transformace vypočtena s %d / %d inliery.", num_inliers, len(raw_points))

        if num_inliers < min_points:
            logging.warning(
                "Nízký počet inlierů (%d) pro UV->XY transformaci.", num_inliers)

        return transform_matrix

    except cv2.error as e:
        logging.error("OpenCV chyba při výpočtu UV->XY transformace: %s", e)
        return None
    except Exception as e:
        logging.exception(
            "Neočekávaná chyba při výpočtu UV->XY transformace: %s", e)
        return None

# --- Callback pro Klávesnici ---


def on_press(key):
    """Zpracuje stisk klávesy."""
    global running, draw_request  # Používáme draw_request
    try:
        char_key = key.char.lower()
        if char_key == KEY_DRAW_X:
            print("Požadavek na kreslení: X")
            draw_request = 'X'  # Nastavíme požadavek
        elif char_key == KEY_DRAW_O:
            print("Požadavek na kreslení: O")
            draw_request = 'O'  # Nastavíme požadavek
    except AttributeError:
        if key == KEY_QUIT:
            print("Ukončuji...")
            running = False

# --- Hlavní Funkce ---


def main():
    global pose_model, device, cap, running, center_cell_uv, center_cell_xy, \
        transform_uv_to_xy, safe_z, draw_z, controller, draw_request

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(levelname)s-%(name)s: %(message)s')
    logger = logging.getLogger(__name__)

    # --- Načtení Modelu ---
    logger.info("Načítání Pose modelu...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Používám zařízení: {device}")
        pose_model = YOLO(POSE_MODEL_PATH)
        pose_model.to(device)
        logger.info("Pose model úspěšně načten.")
    except FileNotFoundError:
        logger.error(
            f"!!!! CHYBA: Pose model nenalezen: {POSE_MODEL_PATH} !!!!")
        return
    except Exception as e:
        logger.error(f"Chyba při načítání Pose modelu: {e}")
        return

    # --- Načtení Kalibrace ---
    logger.info(f"Načítání kalibrace z {CALIBRATION_FILE}...")
    calibration_data = load_calibration(CALIBRATION_FILE)
    if not calibration_data:
        return  # Chyba už byla zalogována
    safe_z = calibration_data["safe_z"]
    draw_z = calibration_data["touch_z"]  # V kalibraci je to 'touch_z'
    # Výpočet inverzní transformace
    transform_uv_to_xy = calculate_uv_to_xy_transform(calibration_data)
    if transform_uv_to_xy is None:
        logger.error("Nepodařilo se vypočítat UV->XY transformaci. Ukončuji.")
        return

    # --- Inicializace ArmController ---
    logger.info("Inicializace ArmControlleru...")
    controller = ArmController(port=None, safe_z=safe_z, draw_z=draw_z)
    logger.info(f"Používám Safe Z: {safe_z}, Draw Z: {draw_z}")
    if not controller.connect():
        logger.error("Nepodařilo se připojit k rameni. Ukončuji.")
        # Cleanup pro model a kameru
        if pose_model:
            del pose_model
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        return
    logger.info("Rameno připojeno. Přesun na bezpečnou výšku.")
    # Přesun na výchozí bezpečnou pozici (volitelné, ale dobré)
    initial_pos = controller.get_position()
    if initial_pos:
        controller.go_to_position(
            x=initial_pos[0],
            y=initial_pos[1],
            z=safe_z,
            wait=True)
    else:
        controller.go_to_position(z=safe_z, wait=True)  # Alespoň Z

    # --- Inicializace Kamery ---
    logger.info(f"Otevírání kamery s indexem {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        logger.error(f"Nepodařilo se otevřít kameru {CAM_INDEX}.")
        if pose_model:
            del pose_model  # Uvolnění paměti
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Kamera otevřena ({frame_width}x{frame_height}).")
    cv2.namedWindow(WINDOW_NAME)

    # --- Start Keyboard Listener ---
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    logger.info("Listener klávesnice spuštěn (Stiskni 'X', 'O' nebo 'Esc').")

    # --- Hlavní Smyčka ---
    while running:
        ret, frame = cap.read()
        if not ret:
            logger.error("Nepodařilo se přečíst snímek z kamery.")
            break

        display_frame = frame.copy()
        detected_kpts = None

        # --- Detekce Mřížky ---
        try:
            pose_results = pose_model(
                frame, verbose=False, conf=POSE_CONF_THRESHOLD)
            if pose_results and pose_results[0].keypoints and \
               len(pose_results[0].boxes.data) > 0:
                # Kontrola jistoty detekce bounding boxu mřížky
                grid_box_conf = pose_results[0].boxes.data[0][4].item()
                if grid_box_conf > POSE_CONF_THRESHOLD:
                    kpts_data_raw = pose_results[0].keypoints.data[0].cpu(
                    ).numpy()
                    # Korekce bodů
                    corrected_kpts = correct_grid_points_homography(
                        kpts_data_raw)
                    if corrected_kpts is not None and corrected_kpts.shape[0] == GRID_POINTS_COUNT:
                        detected_kpts = corrected_kpts  # Máme opravené body
                    else:
                        logger.warning(
                            "Korekce mřížky selhala nebo vrátila špatný počet bodů.")
                        detected_kpts = None  # Zajistíme reset
                else:
                    logger.debug("Detekce mřížky (box) pod prahem jistoty.")
                    detected_kpts = None  # Správné odsazení
            else:
                logger.debug(
                    "V obraze nenalezena žádná mřížka (keypoints/boxes).")
                detected_kpts = None  # Správné odsazení

        except Exception as e:
            logger.exception(
                f"Neočekávaná chyba při detekci/korekci mřížky: {e}")
            detected_kpts = None  # Pro jistotu resetujeme

        # --- Zpracování a Kreslení na Obrazovku ---
        center_cell_uv = None  # Resetujeme UV pro každý snímek
        center_cell_xy = None  # Resetujeme XY pro každý snímek
        if detected_kpts is not None:
            # Vykreslení detekované/opravené mřížky
            for i, pt in enumerate(detected_kpts):
                u, v = int(pt[0]), int(pt[1])
                cv2.circle(display_frame, (u, v), POINT_RADIUS, GRID_COLOR, -1)

            # Výpočet středu prostřední buňky (indexy 5, 6, 9, 10)
            center_indices = [5, 6, 9, 10]
            if all(idx < len(detected_kpts) for idx in center_indices):
                center_points_uv = detected_kpts[center_indices]
                center_mean_uv = np.mean(center_points_uv, axis=0)
                center_cell_uv = (
                    int(center_mean_uv[0]), int(center_mean_uv[1]))
                # Vykreslení středu buňky pro ladění
                cv2.circle(
                    display_frame,
                    center_cell_uv,
                    POINT_RADIUS + 2,
                    CENTER_CELL_COLOR,
                    2)

                # Převod středu UV na XY robota
                if transform_uv_to_xy is not None:
                    center_uv_np = np.array(
                        [[center_cell_uv]], dtype=np.float32)  # Tvar (1, 1, 2)
                    transformed_xy = cv2.perspectiveTransform(
                        center_uv_np, transform_uv_to_xy)
                    if transformed_xy is not None and transformed_xy.shape == (
                            1, 1, 2):
                        center_cell_xy = tuple(transformed_xy[0, 0])
                        # logger.debug(f"Střed buňky UV: {center_cell_uv} ->
                        # XY: {center_cell_xy}") # Pro ladění
                    else:
                        logger.warning(
                            f"Nepodařilo se transformovat střed buňky UV->XY (výsledek: {transformed_xy})")
            else:
                logger.warning(
                    "Detekované body nemají očekávané indexy pro střední buňku.")

        # --- Fyzické Kreslení Ramenem ---
        if draw_request and center_cell_xy and controller:
            symbol = draw_request
            target_x, target_y = center_cell_xy
            logger.info(
                f"Přijat požadavek na kreslení '{symbol}' na XY: "
                f"({target_x:.1f}, {target_y:.1f})")
            draw_success = False
            if symbol == 'X':
                draw_success = controller.draw_x(
                    target_x, target_y, SYMBOL_SIZE_MM)
            elif symbol == 'O':
                draw_success = controller.draw_o(
                    target_x, target_y, SYMBOL_SIZE_MM / 2.0)

            if draw_success:
                logger.info(f"Symbol '{symbol}' úspěšně nakreslen.")
            else:
                logger.error(f"Nepodařilo se nakreslit symbol '{symbol}'.")
                # Můžeme zkusit vrátit rameno do bezpečné výšky
                controller.go_to_position(z=safe_z, wait=False)

            draw_request = None  # Resetujeme požadavek

        # Kreslení symbolu X nebo O na obrazovku (pokud byl požadavek) - pro
        # vizualizaci
        if draw_request and center_cell_uv:
            symbol_visual = draw_request  # Přejmenováno pro jasnost
            # Zjistíme velikost textu pro centrování
            (text_w, text_h), _ = cv2.getTextSize(
                symbol_visual, SYMBOL_FONT, SYMBOL_SCALE, SYMBOL_THICKNESS)
            # Vypočítáme pozici levého dolního rohu textu pro centrování
            org_x = center_cell_uv[0] - text_w // 2
            org_y = center_cell_uv[1] + text_h // 2
            cv2.putText(
                display_frame,
                symbol_visual,
                (org_x,
                 org_y),
                SYMBOL_FONT,
                SYMBOL_SCALE,
                SYMBOL_COLOR,
                SYMBOL_THICKNESS,
                cv2.LINE_AA)

        # --- Zobrazení Výsledku ---
        cv2.imshow(WINDOW_NAME, display_frame)

        # --- Zpracování Událostí Okna ---
        key_cv = cv2.waitKey(1) & 0xFF
        if key_cv == 27:  # Klávesa ESC v okně OpenCV také ukončí
            running = False
            logger.info("Ukončeno klávesou ESC v okně OpenCV.")

    # --- Cleanup ---
    logger.info("Zastavování listeneru klávesnice...")
    if listener.is_alive():
        listener.stop()
        # listener.join() # Může blokovat, pokud se listener nezastaví korektně

    logger.info("Uvolňování kamery...")
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    logger.info("Zavírání oken GUI.")

    # Odpojení ramene
    if controller:
        logger.info("Odpojování ramene...")
        controller.disconnect()
        logger.info("Rameno odpojeno.")

    # Uvolnění modelu z paměti GPU/CPU
    if pose_model:
        del pose_model
        if device == 'cuda':
            torch.cuda.empty_cache()
    logger.info("Model uvolněn.")

    print("=" * 50)
    print("Testovací skript dokončen.")
    print("=" * 50)


if __name__ == "__main__":
    main()
