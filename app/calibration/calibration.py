# @generated [partially] Claude Code 2025-01-01: AI-assisted code review
"""
Camera calibration module for TicTacToe robot arm.
"""
# pylint: disable=line-too-long,superfluous-parens,invalid-name,too-many-locals
# pylint: disable=too-many-function-args,no-else-return,consider-using-f-string
# pylint: disable=too-many-return-statements,global-variable-not-assigned
# pylint: disable=global-statement
# pylint: disable=no-member,consider-using-in,broad-exception-caught
# pylint: disable=too-many-nested-blocks
# pylint: disable=unsubscriptable-object,unspecified-encoding,too-many-branches
# pylint: disable=too-many-statements
import json
import logging
import sys
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2  # pylint: disable=no-member
import numpy as np
import torch  # Přidáno pro YOLO
from pynput import keyboard
from ultralytics import YOLO  # Přidáno pro YOLO

# --- Import ArmController and Path Utils ---
try:
    # Importujeme z app.main modulu
    from app.main.arm_controller import ArmController
    from app.main.path_utils import (
        get_calibration_file_path,
        get_pose_model_path
    )
except ImportError:
    print("ERROR: ArmController or path_utils not found in app.main.")
    print("       Ensure the path is correct.")
    sys.exit(1)

# --- Konfigurace ---
# Relativní cesta pro ukládání kalibračního souboru
OUTPUT_FILE = str(get_calibration_file_path())
CAM_INDEX = 0  # Index kamery
WINDOW_NAME = "Hand-Eye Calibration"
# Cesta k YOLO modelu pro detekci pozice mřížky
POSE_MODEL_PATH = str(get_pose_model_path())
# Prahové hodnoty pro detekci mřížky
POSE_CONF_THRESHOLD = 0.5
KEYPOINT_VISIBLE_THRESHOLD = 0.5
MIN_POINTS_FOR_RANSAC = 6  # Minimum bodů pro RANSAC korekci mřížky
# TARGET_POINTS_UV bylo odstraněno
MIN_POINTS_FOR_TRANSFORM = 4  # Minimum bodů pro výpočet finální transformace
# Barvy a velikosti pro kreslení v GUI
TARGET_COLOR = (0, 0, 255)    # Červená pro aktuální cíl
CONFIRMED_COLOR = (0, 255, 0)  # Zelená pro potvrzené
POINT_RADIUS = 5
TEXT_COLOR = (255, 255, 255)  # Bílá pro text
TARGET_TEXT_COLOR = (0, 0, 255)   # Červená pro text u cíle
FONT_SIZE = 0.8  # Velikost písma (zvětšeno)
FONT_THICKNESS = 2  # Tloušťka písma (zvětšeno)
INSTRUCTION_FONT_SIZE = 1.0  # Velikost písma pro instrukce
INSTRUCTION_FONT_THICKNESS = 2  # Tloušťka písma pro instrukce

# Krok pohybu (vráceno)
FINE_STEP_XY = 1.0
FINE_STEP_Z = 0.5
COARSE_STEP_XY = 5.0
COARSE_STEP_Z = 2.0
# Rychlost pohybu
ARM_SPEED = 100000  # Maximální rychlost pro přesné pohyby
# Klávesy (Prohozeno A/D a W/S)
KEY_LEFT = 'w'           # -X
KEY_RIGHT = 's'          # +X
KEY_FORWARD = 'a'        # -Y (Blíž k tělu)
KEY_BACKWARD = 'd'       # +Y (Dál od těla)
KEY_UP = 'r'             # +Z
KEY_DOWN = ''           # -Z
KEY_CONFIRM = keyboard.Key.enter
KEY_QUIT = keyboard.Key.esc
KEY_MODIFIER = keyboard.Key.shift  # Hold Shift for coarse movement

# --- Globální Stav ---
controller: Optional[ArmController] = None
cap: Optional[cv2.VideoCapture] = None  # pylint: disable=no-member
pose_model: Optional[YOLO] = None  # Přidáno pro model
device: Optional[str] = None  # Přidáno pro zařízení (cpu/cuda)
current_frame: Optional[np.ndarray] = None
# Uloží detekované a opravené body mřížky
detected_grid_kpts_uv: Optional[np.ndarray] = None
# Ukládá páry: (cílové_uv z detekované mřížky, potvrzené_xyz)
calibration_points: List[
    Tuple[
        Tuple[int, int],
        Tuple[float, float, float]
    ]
] = []
# Ukládá potvrzené Z-hodnoty (jen Z souřadnice)
calibrated_z: Dict[str, float] = {}
# Ukládá neutrální pozici
neutral_position: Optional[Dict[str, float]] = None
# Původní proměnné pro ovládání (vráceno)
current_target_pos: Optional[Dict[str, float]] = None  # Last position *sent*
move_request: Optional[Dict[str, float]] = None  # Relative move requested
modifier_pressed = False
# Store the last confirmed position from the arm
# Actual pos
last_confirmed_pos_arm: Optional[Tuple[float, float, float]] = None
running = True
current_target_index = 0  # Index v detekovaných bodech
# Stages: "detect_grid", "target_0"..."touch_z", "safe_z", "neutral",
# "done", "error"
current_stage = "detect_grid"  # Začneme detekcí mřížky


# --- Funkce ---


# Přidána funkce pro korekci bodů (z tictactoe_mirror.py, mírně upraveno)
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
        "Nalezeno %d validních bodů mřížky nad prahem %s.",
        num_valid, confidence_threshold
    )
    min_req_points = max(4, min_points_for_ransac)
    if num_valid < min_req_points:
        correction_logger.warning(
            "Nedostatek bodů (%d < %d) pro Homografii. Nelze opravit.",
            num_valid, min_req_points
        )
        return None
    # Ideální mřížka 4x4 body (indexy 0-15)
    ideal_grid_all = np.array([(i % 4, i // 4) for i in range(16)],
                              dtype=np.float32)
    valid_ideal_pts = ideal_grid_all[valid_indices]
    try:
        # pylint: disable=no-member
        homography_matrix, ransac_mask = cv2.findHomography(
            valid_ideal_pts, valid_predicted_pts, method=cv2.RANSAC,
            ransacReprojThreshold=10.0)
        if homography_matrix is None:
            correction_logger.warning("RANSAC selhal při hledání "
                                      "homografie mřížky.")
            return None
        num_inliers = np.sum(
            ransac_mask) if ransac_mask is not None else 0  # Kontrola None
        correction_logger.debug(
            "Homografie mřížky RANSAC nalezla matici s %d inliery.",
            num_inliers
        )
        if num_inliers < min_req_points:
            correction_logger.warning(
                "Homografie mřížky RANSAC nalezla příliš málo inlierů (%d).",
                num_inliers
            )
            return None
    except (RuntimeError, ValueError) as e:  # cv2.error is not recognized by pylint
        correction_logger.error(
            "OpenCV chyba findHomography pro mřížku: %s", e
        )
        return None
    # Aplikujeme transformaci na všechny ideální body
    ideal_grid_all_reshaped = ideal_grid_all.reshape(-1, 1, 2)
    corrected_pts_xy = cv2.perspectiveTransform(  # pylint: disable=no-member
        ideal_grid_all_reshaped.astype(np.float32), homography_matrix
    ).reshape(-1, 2)
    correction_logger.debug("Korekce bodů mřížky homografií úspěšná.")
    return corrected_pts_xy


def print_instructions(stage: str):
    """Vypíše instrukce podle aktuální fáze."""
    print("\n" + "=" * 50)
    print("       Hand-Eye Calibration Utility")
    print("=" * 50)
    print("Ovládání (Aktuální mapování):")
    # Updated instructions for swapped mapping
    print(f"  {KEY_LEFT.upper()}/{KEY_RIGHT.upper()}: Left/Right (-X/+X)")
    print(f"  {KEY_FORWARD.upper()}/{KEY_BACKWARD.upper()}: Fwd/Back (-Y/+Y)")
    print(f"  {KEY_UP.upper()}/{KEY_DOWN.upper()}:   Up/Down   (+Z/-Z)")
    print(f"  Hold [Shift]: Hrubě ({COARSE_STEP_XY}mm XY, {COARSE_STEP_Z}mm Z)")
    print(f"  [Shift] Off : Jemně ({FINE_STEP_XY}mm XY, {FINE_STEP_Z}mm Z)")
    # Upřesněná instrukce
    print("  POHYBUJTE RAMENEM tak, aby se HROT PERA (nebo koncový bod)")
    print("    přesně kryl s ČERVENÝM cílem v okně kamery.")
    print("  Stiskněte [Enter] pro potvrzení pozice.")
    print("  Stiskněte [Esc] pro ukončení kalibrace.")
    print("-" * 50)
    print(f"Aktuální fáze: {get_stage_prompt(stage)}")
    print("=" * 50)


def get_stage_prompt(stage: str) -> str:
    """Vrátí textový popis aktuální fáze."""
    if stage == "detect_grid":
        return ("Umistete kalibracni mrizku do pohledu kamery a stisknete "
                "[Enter] pro detekci bodu.")
    elif stage.startswith("target_"):
        idx = int(stage.split('_')[1])
        # Počet cílů je nyní 16 (body mřížky)
        total_targets = 16
        # Prompt upraven pro detekované body
        return ("Cil %d/%d: Najedte HROTEM PERA pomoci "
                "WASDRF (+Shift) na CERVENY bod mrizky, pak [Enter]" % (idx + 1, total_targets))
    elif stage == "touch_z":
        # Upřesněný prompt
        return ("Z-Kalibrace (Dotek): Pomoci R/F (+Shift) nastavte HROT PERA "
                "tak, aby se DOTYKAL papiru, pak [Enter]")
    elif stage == "safe_z":
        # Upřesněný prompt
        return ("Z-Kalibrace (Bezpecna): Pomoci R/F (+Shift) nastavte "
                "HROT PERA do BEZPECNE vysky pro prejezdy, pak [Enter]")
    elif stage == "neutral":
        # Prompt pro neutrální pozici
        return (
            "Nastaveni neutralni pozice: Pomoci WASDRF (+Shift) nastavte "
            "HROT PERA do NEUTRALNI pozice, kam se bude vracet po kazdem tahu, "
            "pak [Enter]")
    elif stage == "calculating":
        return "Vypocet transformace..."
    elif stage == "done":
        return "Kalibrace dokoncena! Ukladani dat..."
    elif stage == "error":
        return "Chyba kalibrace!"
    else:
        return "Neznama faze: %s" % stage


def calculate_transformation() -> Optional[np.ndarray]:
    """Vypočítá perspektivní transformační matici (homografii)."""
    global calibration_points
    if len(calibration_points) < MIN_POINTS_FOR_TRANSFORM:
        print("ERROR: Nedostatek bodů pro výpočet transformace (potřeba %d, máme %d)." % (MIN_POINTS_FOR_TRANSFORM, len(calibration_points)))
        return None

    # Připravíme pole bodů pro OpenCV
    # Zdrojové body: XY souřadnice ramene
    # Cílové body: UV souřadnice obrazu
    src_pts = np.array([p[1][:2] for p in calibration_points],
                       dtype=np.float32)  # Pouze X, Y
    dst_pts = np.array([p[0] for p in calibration_points],
                       dtype=np.float32)  # UV

    print("Zdrojové body (Robot XY):")
    print(src_pts)
    print("Cílové body (Image UV):")
    print(dst_pts)

    # Výpočet perspektivní transformace (homografie) (3x3 matice)
    # Používá RANSAC pro robustnost vůči odlehlým hodnotám
    # pylint: disable=no-member
    transform_matrix, inliers = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0)  # Prah pro RANSAC

    if transform_matrix is None:
        print("ERROR: Nepodařilo se vypočítat perspektivní transformaci "
              "(homografii).")
        return None

    num_inliers = np.sum(
        inliers) if inliers is not None else 0  # Kontrola None
    print("Perspektivní transformace (homografie) vypočtena s %d / %d inliery." % (num_inliers, len(src_pts)))

    if num_inliers < MIN_POINTS_FOR_TRANSFORM:
        print("WARNING: Počet inlierů (%d) je menší než minimum %d)." % (num_inliers, MIN_POINTS_FOR_TRANSFORM))

    print("Vypočtená perspektivní transformační matice "
          "(Robot XY -> Image UV):")
    print(transform_matrix)
    return transform_matrix


def on_press(key):
    """Callback pro stisk klávesy."""
    global running, current_stage, current_target_index
    global calibration_points, calibrated_z
    # Vrácené globální proměnné pro ovládání
    global move_request, modifier_pressed, current_target_pos, \
        last_confirmed_pos_arm  # Renamed from last_confirmed_pos

    # Výpočet kroku (vráceno)
    step_xy = COARSE_STEP_XY if modifier_pressed else FINE_STEP_XY
    step_z = COARSE_STEP_Z if modifier_pressed else FINE_STEP_Z
    new_move = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    triggered = False  # Flag, zda byla stisknuta relevantní klávesa

    try:
        char_key = key.char.lower()
        # Updated key handling for swapped mapping
        if char_key == KEY_LEFT:    # W
            new_move['x'] = -step_xy
        elif char_key == KEY_RIGHT:   # S
            new_move['x'] = step_xy
        elif char_key == KEY_FORWARD:  # A
            new_move['y'] = -step_xy  # Blíž k tělu
        elif char_key == KEY_BACKWARD:  # D
            new_move['y'] = step_xy   # Dál od těla
        elif char_key == KEY_UP:      # R
            new_move['z'] = step_z
        elif char_key == KEY_DOWN:    # F
            new_move['z'] = -step_z
        else:
            # Není to pohybová klávesa (W, S, A, D, R, F)
            return
        triggered = True
    except AttributeError:
        # Speciální klávesy (Shift, Enter, Esc)
        if key == KEY_MODIFIER or key == keyboard.Key.shift_r:
            if not modifier_pressed:
                print("[Hrubý režim ZAPNUT]")
            modifier_pressed = True
            triggered = True
        elif key == KEY_CONFIRM:
            # --- Zpracování Enter podle fáze ---
            if current_stage == "detect_grid":
                if current_frame is None or pose_model is None:
                    print("ERROR: Snímek nebo model nejsou k dispozici "
                          "pro detekci.")
                    return
                print("Detekuji mřížku...")
                try:
                    pose_results = pose_model(current_frame, verbose=False,
                                              conf=POSE_CONF_THRESHOLD)
                    if pose_results and pose_results[0].keypoints and \
                       len(pose_results[0].boxes.data) > 0:
                        # Check confidence of the detected grid box itself
                        grid_box_conf = pose_results[0].boxes.data[0][4].item()
                        if grid_box_conf > POSE_CONF_THRESHOLD:
                            kpts_data_raw = (
                                pose_results[0].keypoints.data[0].cpu().numpy()
                            )
                            corrected_kpts = correct_grid_points_homography(
                                kpts_data_raw
                            )
                            if corrected_kpts is not None and \
                               corrected_kpts.shape[0] == 16:
                                global detected_grid_kpts_uv
                                detected_grid_kpts_uv = corrected_kpts
                                print("Mřížka detekována a opravena "
                                      "(%d bodů)." % len(detected_grid_kpts_uv))
                                current_stage = "target_0"  # Přejdeme na první cíl
                            else:
                                print("ERROR: Nepodařilo se opravit dostatek "
                                      "bodů mřížky.")
                                # Zůstaneme ve fázi detekce
                        else:
                            print("ERROR: Detekce mřížky nemá "
                                  "dostatečnou jistotu.")
                    else:
                        print("ERROR: V obraze nebyla nalezena žádná mřížka.")
                except Exception as e:
                    print("ERROR: Neočekávaná chyba při detekci mřížky: "
                          "%s" % e)
                    # Zůstaneme ve fázi detekce

            elif current_stage.startswith("target_"):
                # Logika potvrzení pro target fáze
                if detected_grid_kpts_uv is None:
                    print("ERROR: Body mřížky nebyly detekovány.")
                    return
                if not current_target_pos:
                    print("ERROR: Aktuální cílová pozice není známa.")
                    return

                confirmed_x = current_target_pos['x']
                confirmed_y = current_target_pos['y']
                confirmed_z = current_target_pos['z']
                confirmed_xyz = (confirmed_x, confirmed_y, confirmed_z)

                # Získání UV souřadnic z DETEKOVANÉ mřížky a převod na Python
                # int
                uv_point = detected_grid_kpts_uv[current_target_index]
                target_uv_tuple = tuple(map(int, uv_point))

                # Uložíme pár (UV detekovaný bod, XYZ CÍLOVÁ pozice)
                calibration_points.append((target_uv_tuple, confirmed_xyz))
                print("Uložen kalibrační bod %d: "
                      "Cíl_UV=%s, Cíl_XYZ=%s" % (current_target_index + 1, target_uv_tuple, confirmed_xyz))

                # Získání skutečné pozice pro info
                actual_pos_arm = controller.get_position(cached=False)
                if actual_pos_arm:
                    last_confirmed_pos_arm = actual_pos_arm
                    print("    (Skutečná pozice: X=%f, Y=%f, Z=%f)" % (actual_pos_arm[0], actual_pos_arm[1], actual_pos_arm[2]))
                else:
                    last_confirmed_pos_arm = None
                    print("    (Skutečnou pozici se nepodařilo získat)")

                # Přechod na další cíl nebo Z kalibraci
                current_target_index += 1
                if current_target_index < 16:  # Máme 16 bodů mřížky (0-15)
                    current_stage = f"target_{current_target_index}"
                else:
                    current_stage = "touch_z"

            elif current_stage == "touch_z":
                if not current_target_pos:  # Safety check
                    return
                confirmed_z = current_target_pos['z']
                calibrated_z["touch_z"] = confirmed_z
                current_stage = "safe_z"

            elif current_stage == "safe_z":
                if not current_target_pos:  # Safety check
                    return
                confirmed_z = current_target_pos['z']
                calibrated_z["safe_z"] = confirmed_z
                current_stage = "neutral"

            elif current_stage == "neutral":
                if not current_target_pos:  # Safety check
                    return
                # Uložíme aktuální pozici jako neutrální
                global neutral_position
                neutral_position = {
                    "x": current_target_pos['x'],
                    "y": current_target_pos['y'],
                    "z": current_target_pos['z']
                }
                print("Neutrální pozice nastavena: X=%f, Y=%f, Z=%f" % (neutral_position['x'], neutral_position['y'], neutral_position['z']))

                # Přejdeme na výpočet transformace
                current_stage = "calculating"
                print("\n%s" % get_stage_prompt(current_stage))
                transform = calculate_transformation()
                if transform is not None:
                    current_stage = "done"
                else:
                    current_stage = "error"
                running = False

            # Společné pro všechna úspěšná Enter stisky
            # (kromě detect_grid error, kde se fáze nemění)
            triggered = True
            print_instructions(current_stage)

        elif key == KEY_QUIT:
            running = False
            triggered = True  # Potřebujeme trigger
            print("\nKalibrace přerušena uživatelem.")

    # Aktualizace move_request (vráceno)
    if triggered and running:
        if new_move != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
            move_request = new_move
        # Jinak (Enter, Esc, Shift) nechte move_request být None


def on_release(key):
    """Callback pro uvolnění klávesy."""
    global modifier_pressed
    if key == KEY_MODIFIER or key == keyboard.Key.shift_r:
        modifier_pressed = False
        print("[Jemný režim ZAPNUT]")


def save_calibration_data(transform_matrix: Optional[np.ndarray]):
    """Uloží vypočtenou transformaci a Z hodnoty do JSON."""
    global calibrated_z, neutral_position
    # Kontrola, zda máme všechna potřebná data
    required_z_keys = ["touch_z", "safe_z"]
    all_z_done = all(k in calibrated_z for k in required_z_keys)

    if transform_matrix is None:
        print("ERROR: Chybí transformační matice. Data nebudou uložena.")
        return
    if not all_z_done:
        print("ERROR: Chybí některé Z-kalibrační hodnoty. "
              "Data nebudou uložena.")
        missing_z = [k for k in required_z_keys if k not in calibrated_z]
        if missing_z:
            print("  Chybějící Z hodnoty: %s" % missing_z)
        return

    # Připravíme data pro JSON
    # Matice NumPy musí být převedena na list pro serializaci
    data_to_save = {
        # Změna názvu klíče pro homografii
        "perspective_transform_matrix_xy_to_uv": transform_matrix.tolist(),
        "touch_z": calibrated_z["touch_z"],
        "safe_z": calibrated_z["safe_z"],
        "calibration_points_raw": [  # Uložení i surových dat pro ladění
            {"target_uv": cp[0], "robot_xyz": cp[1]}
            for cp in calibration_points
        ]
        # Můžeme přidat další metadata, např. datum, rozlišení kamery...
    }

    # Přidáme neutrální pozici, pokud byla nastavena
    if neutral_position:
        data_to_save["neutral_position"] = neutral_position
        print("Neutrální pozice uložena: X=%f, Y=%f, Z=%f" % (neutral_position['x'], neutral_position['y'], neutral_position['z']))
    else:
        print("VAROVÁNÍ: Neutrální pozice nebyla nastavena!")

    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print("Kalibrační data úspěšně uložena do %s" % OUTPUT_FILE)
    except Exception as e:
        print("ERROR: Nepodařilo se uložit kalibrační data do "
              "%s: %s" % (OUTPUT_FILE, e))


def calibration_main():
    """Hlavní funkce kalibračního skriptu."""
    global controller, cap, current_frame, running, current_stage
    global calibration_points, move_request, current_target_pos  # Vráceno
    global last_confirmed_pos_arm  # Přejmenováno
    # Přidáno pro model a detekci
    global pose_model, device, detected_grid_kpts_uv

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(levelname)s-%(name)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    # --- Načtení Pose modelu ---
    logger.info("Načítání Pose modelu...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Používám zařízení: %s", device)
        pose_model = YOLO(POSE_MODEL_PATH)
        pose_model.to(device)
        logger.info("Pose model úspěšně načten.")
    except FileNotFoundError:
        logger.error(
            "!!!! CHYBA: Pose model nenalezen: %s !!!!", POSE_MODEL_PATH
        )
        return
    except Exception as e:
        logger.error("Chyba při načítání Pose modelu: %s", e)
        return
    # ------------------------------

    # --- Inicializace kamery ---
    logger.info("Otevírání kamery s indexem %s...", CAM_INDEX)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        logger.error("Nepodařilo se otevřít kameru %s.", CAM_INDEX)
        return
    # Zkusíme nastavit rozlišení (pokud je potřeba, jinak zakomentujte)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Kamera otevřena (%sx%s).", frame_width, frame_height)
    cv2.namedWindow(WINDOW_NAME)

    # --- Inicializace ArmController ---
    logger.info("Inicializace ArmController...")
    controller = ArmController(port=None, speed=ARM_SPEED, draw_z=5.0, safe_z=15.0)  # Přidány parametry jako v hlavní aplikaci
    if not controller.connect():
        logger.error("Nepodařilo se připojit k rameni. Ukončuji.")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        return
    logger.info("Rameno připojeno.")

    # Nastavíme rychlostní faktor jako v hlavní aplikaci
    if hasattr(controller, 'swift') and controller.swift:
        controller.swift.set_speed_factor(2)  # Stejný faktor jako v hlavní aplikaci
        logger.info("Speed factor set to 2")

    # Získáme výchozí pozici pro informaci a nastavíme current_target_pos
    initial_pos_tuple = controller.get_position(cached=False)  # Přidán parametr cached=False
    if initial_pos_tuple:
        # Nastavíme počáteční CÍLOVOU pozici
        start_x, start_y, _ = initial_pos_tuple  # Původní Z ignorujeme
        target_z_init = 10.0  # Cílová Z výška
        logger.info("Přesun na počáteční výšku Z=%f...", target_z_init)
        move_init_ok = controller.go_to_position(x=start_x, y=start_y,
                                                 z=target_z_init, speed=ARM_SPEED, wait=True)

        if move_init_ok:
            # Znovu získáme pozici PO přesunu na Z=10
            actual_pos_after_init = controller.get_position(cached=False)
            if actual_pos_after_init:
                current_target_pos = {'x': actual_pos_after_init[0],
                                      'y': actual_pos_after_init[1],
                                      'z': actual_pos_after_init[2]}
                last_confirmed_pos_arm = actual_pos_after_init  # Zobrazíme skutečnou
                logger.info("Rameno na počáteční výšce. Aktuální cíl: %s", current_target_pos)
            else:
                logger.error("Nepodařilo se získat pozici po přesunu na Z=10. "
                             "Ukončuji.")
                controller.disconnect()
                if cap:
                    cap.release()
                cv2.destroyAllWindows()
                return
        else:
            logger.error("Nepodařilo se přesunout rameno na počáteční "
                         "výšku Z=10. Ukončuji.")
            controller.disconnect()
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            return
    else:
        logger.error("Nepodařilo se získat výchozí pozici ramene. Ukončuji.")
        controller.disconnect()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        return

    # --- Start Keyboard Listener ---
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    logger.info("Listener klávesnice spuštěn.")

    print_instructions(current_stage)

    # Hlavní smyčka GUI a logiky
    transform_matrix_result = None
    while running:
        # --- Zpracování pohybu (vráceno) ---
        if move_request:
            req = move_request
            move_request = None  # Spotřebujeme požadavek

            # Vypočítáme nový cíl relativně k AKTUÁLNÍMU CÍLI
            target_x = current_target_pos['x'] + req['x']
            target_y = current_target_pos['y'] + req['y']
            target_z = current_target_pos['z'] + req['z']

            # Omezení Z (např.)
            target_z = max(0, min(250, target_z))

            print("-> Pohyb na CÍL: X=%.1f Y=%.1f Z=%.1f" % (target_x, target_y, target_z))
            print("   Aktuální pozice před pohybem: X=%.1f Y=%.1f Z=%.1f" % (current_target_pos['x'], current_target_pos['y'], current_target_pos['z']))

            # Pošleme příkaz k pohybu s
            # explicitní rychlostí a wait=True pro okamžitou odezvu
            move_ok = controller.go_to_position(x=target_x, y=target_y,
                                                z=target_z, speed=ARM_SPEED, wait=True)
            print(f"   Výsledek pohybu: {'ÚSPĚCH' if move_ok else 'SELHÁNÍ'}")

            if move_ok:
                # Aktualizujeme CÍLOVOU pozici HNED
                current_target_pos = {'x': target_x, 'y': target_y,
                                      'z': target_z}
            else:
                print("WARNING: Příkaz k pohybu selhal (Limit?)")
                # Nepokoušíme se vrátit, jen logujeme
                # Zkusíme resynchronizovat CÍL se skutečnou pozicí
                print("   Resynchronizuji CÍL se skutečnou pozicí...")
                actual_pos = controller.get_position(cached=False)
                if actual_pos:
                    current_target_pos = {'x': actual_pos[0],
                                          'y': actual_pos[1],
                                          'z': actual_pos[2]}
                    print("   CÍL resynchronizován na: %s" % current_target_pos)
                else:
                    print("   WARNING: Nelze resynchronizovat CÍL.")

        # --- Čtení kamery a GUI --- (Zbytek smyčky zůstává podobný)
        ret, frame = cap.read()
        if not ret:
            logger.error("Nepodařilo se přečíst snímek z kamery.")
            running = False
            current_stage = "error"
            break
        current_frame = frame.copy()  # Uložíme kopii pro případné použití

        # --- Vykreslení do GUI ---
        display_frame = frame.copy()

        # Vykreslení již potvrzených bodů (pouze jejich cílové UV)
        for i, (target_uv, _) in enumerate(calibration_points):
            cv2.circle(display_frame, target_uv, POINT_RADIUS,
                       CONFIRMED_COLOR, -1)
            cv2.putText(display_frame, str(i + 1),
                        (target_uv[0] + 5, target_uv[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, CONFIRMED_COLOR, 1)

        # Vykreslení detekované mřížky (pokud existuje)
        if detected_grid_kpts_uv is not None:
            for idx, pt in enumerate(detected_grid_kpts_uv):
                u, v = int(pt[0]), int(pt[1])
                # Všechny body mřížky malým zeleným kroužkem
                cv2.circle(display_frame, (u, v), POINT_RADIUS - 1,
                           CONFIRMED_COLOR, 1)

                # Zvýraznění aktuálního cíle (pokud jsme v target fázi)
                if current_stage.startswith("target_") and \
                   idx == current_target_index:
                    # Aktuální cíl větším červeným kruhem
                    cv2.circle(display_frame, (u, v), POINT_RADIUS + 2,
                               TARGET_COLOR, 2)

        # Zobrazení aktuálního promptu v obraze - větším písmem
        prompt = get_stage_prompt(current_stage)
        # Rozdělíme prompt na více řádků, pokud je příliš dlouhý
        max_chars_per_line = 50
        prompt_lines = []

        # Rozdělení textu na řádky
        words = prompt.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                prompt_lines.append(current_line)
                current_line = word
        if current_line:
            prompt_lines.append(current_line)

        # Vykreslení textu s černým obrysem pro lepší čitelnost
        y_pos = frame_height - 20 - (len(prompt_lines) - 1) * 40
        for line in prompt_lines:
            # Černý obrys textu
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(display_frame, line, (10 + dx, y_pos + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, INSTRUCTION_FONT_SIZE,
                            (0, 0, 0), INSTRUCTION_FONT_THICKNESS + 1, cv2.LINE_AA)
            # Samotný text
            cv2.putText(display_frame, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, INSTRUCTION_FONT_SIZE,
                        TEXT_COLOR, INSTRUCTION_FONT_THICKNESS, cv2.LINE_AA)
            y_pos += 40

        # Zobrazení poslední SKUTEČNĚ POTVRZENÉ XYZ pozice (pokud existuje)
        if last_confirmed_pos_arm:
            pos_text = ("Potvrzena pozice: X=%f Y=%f Z=%f" % (last_confirmed_pos_arm[0], last_confirmed_pos_arm[1], last_confirmed_pos_arm[2]))
            # Černý obrys textu
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(display_frame, pos_text, (10 + dx, 30 + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE,
                            (0, 0, 0), FONT_THICKNESS + 1, cv2.LINE_AA)
            # Samotný text
            cv2.putText(display_frame, pos_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR,
                        FONT_THICKNESS, cv2.LINE_AA)

        # Zobrazení AKTUÁLNÍ CÍLOVÉ XYZ pozice
        if current_target_pos:
            target_pos_text = "Aktualni cil: X=%f Y=%f Z=%f" % (current_target_pos['x'], current_target_pos['y'], current_target_pos['z'])
            # Černý obrys textu
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(display_frame, target_pos_text, (10 + dx, 70 + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE,
                            (0, 0, 0), FONT_THICKNESS + 1, cv2.LINE_AA)
            # Samotný text
            cv2.putText(display_frame, target_pos_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 0),
                        FONT_THICKNESS, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display_frame)

        # Zpracování událostí okna
        key_cv = cv2.waitKey(20) & 0xFF  # Kratší čekání (20ms)
        if key_cv == 27:  # Klávesa ESC v okně OpenCV také ukončí
            running = False
            logger.info("Ukončeno klávesou ESC v okně OpenCV.")

    # --- Konec hlavní smyčky - Výpočet a Ukládání ---
    transform_matrix_result = None  # Reset pro jistotu
    if current_stage == "done":
        # Výpočet už proběhl v on_press
        # Potřebujeme ale získat výsledek
        if len(calibration_points) >= MIN_POINTS_FOR_TRANSFORM:
            print("Přepočítávání transformace pro uložení...")  # Změna logu
            transform_matrix_result = calculate_transformation()
            if transform_matrix_result is not None:
                save_calibration_data(transform_matrix_result)
            else:
                print("Chyba při finálním výpočtu transformace. "
                      "Data neuložena.")
                current_stage = "error"  # Označíme chybu
        else:
            print("Nedostatek bodů (%d) pro finální "
                  "výpočet. Data neuložena." % len(calibration_points))
            current_stage = "error"

    elif current_stage == "error":
        print("Kalibrace selhala nebo byla přerušena před výpočtem.")
    else:  # Přerušeno uživatelem
        print("Kalibrace přerušena, data nebudou uložena.")

    # --- Cleanup ---
    logger.info("Zastavování listeneru klávesnice...")
    listener.stop()
    listener.join()  # Počkáme na ukončení vlákna listeneru

    logger.info("Uvolňování kamery...")
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    logger.info("Zavírání oken GUI.")

    logger.info("Odpojování ramene...")
    if controller:
        controller.disconnect()
    logger.info("Rameno odpojeno.")

    print("=" * 50)
    print("Kalibrační skript dokončen.")
    print("=" * 50)


if __name__ == "__main__":
    calibration_main()
