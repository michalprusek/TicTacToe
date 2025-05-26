import cv2
import os
import datetime
import numpy as np
import time
import torch
from ultralytics import YOLO
import logging
import sys
# --- Visualization Imports ---
# import torchvision.transforms.functional as TF # Není potřeba pro OpenCV
# kreslení

# --- Konfigurace ---
# ... (Cesty k modelům, výstupní adresář atd. zůstávají stejné) ...
# !!! UPRAVTE CESTU !!!
DETECT_MODEL_PATH = "/Users/michalprusek/PycharmProjects/TicTacToe/weights/best_detection.pt"
# !!! UPRAVTE CESTU !!!
POSE_MODEL_PATH = "/Users/michalprusek/PycharmProjects/TicTacToe/weights/best_pose.pt"
OUTPUT_DIR = "captured_frames_with_preds"
SAVE_FRAMES = False
BBOX_CONF_THRESHOLD = 0.5
POSE_CONF_THRESHOLD = 0.5
KEYPOINT_VISIBLE_THRESHOLD = 0.5  # Prah pro body použité pro fitování homografie
GRID_LINE_COLOR = (0, 255, 255)  # Žlutá pro mřížku
KPT_COLOR_CORRECTED = (0, 255, 0)  # Zelená pro opravené body
KPT_COLOR_RAW = (255, 0, 0)      # Modrá pro původní detekce (volitelné)
KPT_RADIUS = 4
CAM_INDEX = 0
DISABLE_AUTOFOCUS = True

# --- Setup Logging ---
# ... (Logging setup zůstává stejný) ...
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(
            sys.stdout)],
    force=True)
logger = logging.getLogger(__name__)
logger.info("Logger initialized for webcam script.")

# --- Funkce pro korekci Keypointů (Homografie pomocí RANSAC) ---


def correct_grid_points_homography(
        predicted_kpts_data,
        confidence_threshold=0.5,
        min_points_for_ransac=4):
    """
    Opraví pozice keypointů pomocí odhadu homografie a RANSAC.
    Args:
        predicted_kpts_data (np.ndarray): Pole tvaru [16, 3] s [x, y, conf/vis].
        confidence_threshold (float): Minimální jistota/viditelnost pro zařazení bodu do fitování.
        min_points_for_ransac (int): Minimální počet validních bodů pro pokus o RANSAC (pro homografii jsou potřeba 4).
    Returns:
        np.ndarray or None: Opravené pole keypointů [16, 2] (pouze xy) nebo None, pokud korekce selže.
    """
    correction_logger = logging.getLogger(
        f"{__name__}.correct_grid_homography")

    # 1. Získání validních bodů
    valid_mask = predicted_kpts_data[:, 2] > confidence_threshold
    valid_indices = np.where(valid_mask)[0]
    # Potřebujeme tvary [N_valid, 2] pro OpenCV
    valid_predicted_pts = predicted_kpts_data[valid_mask, :2].astype(
        np.float32)

    num_valid = len(valid_indices)
    correction_logger.debug(
        f"Found {num_valid} valid points above threshold {confidence_threshold}.")

    # Pro homografii potřebujeme alespoň 4 body
    min_req_points = max(4, min_points_for_ransac)
    if num_valid < min_req_points:
        correction_logger.debug(
            f"Not enough valid points ({num_valid} < {min_req_points}) for Homography RANSAC. Cannot correct.")
        return None  # Vracíme None, když nemůžeme opravit

    # 2. Definice ideální mřížky (např. 0-3 pro řádky/sloupce) a výběr odpovídajících bodů
    # Důležité: Pořadí musí odpovídat indexům 0-15 z YOLO modelu
    ideal_grid_all = np.array(
        [(i % 4, i // 4) for i in range(16)], dtype=np.float32)  # Shape [16, 2]
    valid_ideal_pts = ideal_grid_all[valid_indices]  # Shape [N_valid, 2]

    # 3. Odhad Homografie pomocí RANSAC
    try:
        # Najde matici H takovou, že predicted_pts ~= H * ideal_pts
        homography_matrix, ransac_mask = cv2.findHomography(
            valid_ideal_pts,  # Zdrojové body (ideální mřížka)
            valid_predicted_pts,  # Cílové body (detekované)
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            # Max povolená reprojekční chyba v pixelech (LADIT!)
            # maxIters=2000, # Výchozí bývá dostatečné
            # confidence=0.99 # Výchozí bývá dostatečné
        )

        if homography_matrix is None:
            correction_logger.warning(
                "RANSAC failed to find a homography. Cannot correct.")
            return None

        num_inliers = np.sum(ransac_mask) if ransac_mask is not None else 0
        correction_logger.debug(
            f"Homography RANSAC found matrix with {num_inliers} inliers.")

        # Volitelný check: Dostatek inlierů?
        if num_inliers < min_req_points:
            correction_logger.warning(
                f"Homography RANSAC found too few inliers ({num_inliers}). Cannot correct reliably.")
            return None

    except cv2.error as e:
        correction_logger.error(
            f"OpenCV error during findHomography: {e}. Cannot correct.")
        return None
    except Exception as e:
        correction_logger.exception(
            f"Unexpected error during findHomography: {e}. Cannot correct.")
        return None

    # 4. Aplikace homografie na VŠECHNY ideální body
    # Potřebujeme tvar [N, 1, 2] pro cv2.perspectiveTransform
    ideal_grid_all_reshaped = ideal_grid_all.reshape(-1, 1, 2)
    corrected_pts_xy = cv2.perspectiveTransform(
        ideal_grid_all_reshaped, homography_matrix).reshape(-1, 2)  # Shape [16, 2]

    correction_logger.debug("Homography correction successful.")
    return corrected_pts_xy  # Vracíme jen opravené xy souřadnice


# --- Pomocná funkce pro kreslení (Upravená pro homografii) ---
def draw_predictions(
        frame,
        detect_results,
        pose_results,
        box_threshold=0.5,
        pose_threshold=0.5,
        kpt_threshold=0.5,
        apply_correction=False):
    vis_logger = logging.getLogger(f"{__name__}.draw_predictions")
    frame_h, frame_w = frame.shape[:2]

    # --- Vykreslení detekcí (X/O) ---
    # ... (stejné jako předtím) ...
    if detect_results and detect_results[0].boxes and len(
            detect_results[0].boxes.data) > 0:
        boxes = detect_results[0].boxes
        for box in boxes.data:
            conf = box[4].item()
            if conf > box_threshold:
                x1, y1, x2, y2 = map(int, box[:4])
                cls_id = int(box[5].item())
                label = detect_results[0].names.get(cls_id, f"CLS{cls_id}")
                text = f"{label}: {conf:.2f}"
                color = (0, 255, 0) if label == "O" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- Vykreslení Pose (Mřížka a Keypointy) ---
    if pose_results and pose_results[0].boxes and pose_results[0].keypoints and len(
            pose_results[0].boxes.data) > 0:
        boxes_pose = pose_results[0].boxes
        keypoints_pose = pose_results[0].keypoints
        vis_logger.debug("Found %d raw pose instances.", len(boxes_pose.data))

        for i in range(len(boxes_pose.data)):  # Pro každou detekovanou mřížku
            box_pose = boxes_pose.data[i]
            conf_pose = box_pose[4].item()

            if conf_pose > pose_threshold:
                vis_logger.debug(
                    "  Processing pose instance %d (conf: %.2f)", i, conf_pose)
                kpts_data_raw = keypoints_pose.data[i].cpu().numpy()  # [16, 3]

                corrected_kpts_xy = None  # Inicializace
                if apply_correction:
                    corrected_kpts_xy = correct_grid_points_homography(
                        kpts_data_raw, kpt_threshold, min_points_for_ransac=6)
                    if corrected_kpts_xy is None:
                        vis_logger.debug(
                            "Correction failed, drawing raw points and no grid.")
                        # Pokud korekce selže, nebudeme kreslit mřížku, jen
                        # body
                    else:
                        vis_logger.debug(
                            "Using corrected keypoints for grid drawing.")

                # --- Kreslení bodů (vždy kreslíme, i když korekce selže) ---
                drawn_points_count = 0
                for k_idx, kpt_raw in enumerate(kpts_data_raw):
                    x_raw, y_raw, conf = kpt_raw
                    if conf > kpt_threshold and 0 <= x_raw < frame_w and 0 <= y_raw < frame_h:
                        # Vykreslíme původní detekovaný bod (modře)
                        cv2.circle(frame, (int(x_raw), int(y_raw)),
                                   KPT_RADIUS, KPT_COLOR_RAW, -1)
                        drawn_points_count += 1
                vis_logger.debug(
                    f"    Drew {drawn_points_count} raw keypoints above threshold.")

                # --- Kreslení opravené mřížky (POUZE pokud korekce uspěla) ---
                if apply_correction and corrected_kpts_xy is not None:
                    # Máme opravené pozice všech 16 bodů v corrected_kpts_xy [16, 2]
                    # Můžeme je spojit rovnou podle indexů

                    # Vykreslení opravených bodů (zeleně)
                    for k_idx in range(16):
                        x_corr, y_corr = corrected_kpts_xy[k_idx]
                        if 0 <= x_corr < frame_w and 0 <= y_corr < frame_h:
                            cv2.circle(
                                frame, (int(x_corr), int(y_corr)), KPT_RADIUS - 1, KPT_COLOR_CORRECTED, -1)

                    # Horizontální čáry
                    for row in range(4):
                        for col in range(3):
                            idx1 = row * 4 + col
                            idx2 = row * 4 + (col + 1)
                            pt1 = tuple(corrected_kpts_xy[idx1].astype(int))
                            pt2 = tuple(corrected_kpts_xy[idx2].astype(int))
                            # Jednoduchá kontrola, zda nejsou body příliš
                            # daleko (mimo obraz)
                            if 0 <= pt1[0] < frame_w and 0 <= pt1[1] < frame_h and \
                               0 <= pt2[0] < frame_w and 0 <= pt2[1] < frame_h:
                                cv2.line(frame, pt1, pt2, GRID_LINE_COLOR, 1)

                    # Vertikální čáry
                    for col in range(4):
                        for row in range(3):
                            idx1 = row * 4 + col
                            idx2 = (row + 1) * 4 + col
                            pt1 = tuple(corrected_kpts_xy[idx1].astype(int))
                            pt2 = tuple(corrected_kpts_xy[idx2].astype(int))
                            if 0 <= pt1[0] < frame_w and 0 <= pt1[1] < frame_h and \
                               0 <= pt2[0] < frame_w and 0 <= pt2[1] < frame_h:
                                cv2.line(frame, pt1, pt2, GRID_LINE_COLOR, 1)
                    vis_logger.debug("    Drew corrected grid lines.")

    return frame

# --- Hlavní funkce ---
# ... (main funkce zůstává téměř stejná, jen volá upravenou draw_predictions) ...


def main():
    global logger
    if SAVE_FRAMES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("Ukládání snímků: %s", OUTPUT_DIR)
    logger.info("Načítání modelů...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Používám zařízení: %s", device)
        detect_model = YOLO(DETECT_MODEL_PATH)
        pose_model = YOLO(POSE_MODEL_PATH)
        detect_model.to(device)
        pose_model.to(device)
        logger.info("Modely načteny.")
    except FileNotFoundError:
        logger.error(
            "!!!! Chyba: Model(y) nenalezen(y): %s, %s !!!!", DETECT_MODEL_PATH, POSE_MODEL_PATH)
        return
    except Exception as e:
        logger.error("Chyba při načítání modelů: %s", e)
        return
    logger.info("Inicializace kamery (index %d)...", CAM_INDEX)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        logger.error("Nepodařilo se otevřít kameru.")
        return
    if DISABLE_AUTOFOCUS:
        logger.info("Vypínám autofocus...")
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(2)
        logger.info("  Autofocus aktuálně: %s", cap.get(cv2.CAP_PROP_AUTOFOCUS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        f"Rozlišení kamery: {frame_width}x{frame_height}, FPS: {
            fps_cam:.2f}")

    frame_counter = 0
    session_id = None
    session_dir = None
    if SAVE_FRAMES:
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(OUTPUT_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        logger.info("Session directory: %s", session_dir)

    print("\n--- Spuštěno snímání a detekce ---")
    print("Stiskněte 'Q' pro ukončení.")
    if SAVE_FRAMES:
        print("Stiskněte 'S' pro uložení snímku.")
    apply_correction = False  # Výchozí stav korekce
    print(
        "Korekce bodů mřížky (Homografie): {} (Stiskni 'C' pro přepnutí)".format(
            'Zapnuto' if apply_correction else 'Vypnuto'))

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Nepodařilo se získat snímek.")
            break
        frame_display = frame.copy()
        start_time = time.time()
        try:
            detect_results = detect_model(
                frame, verbose=False, conf=BBOX_CONF_THRESHOLD)
            pose_results = pose_model(
                frame, verbose=False, conf=POSE_CONF_THRESHOLD)
        except Exception as e:
            logger.error("Chyba během inference: %s", e)
            continue
        inference_time = time.time() - start_time
        try:
            # Voláme s přepínačem apply_correction
            frame_with_preds = draw_predictions(
                frame_display,
                detect_results,
                pose_results,
                box_threshold=BBOX_CONF_THRESHOLD,
                pose_threshold=POSE_CONF_THRESHOLD,
                kpt_threshold=KEYPOINT_VISIBLE_THRESHOLD,
                apply_correction=apply_correction)
        except Exception as e:
            logger.error("Chyba během vykreslování: %s", e)
            frame_with_preds = frame_display
        fps_display = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(
            frame_with_preds, "FPS: {:.1f}".format(fps_display), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('YOLOv8 Detection & Pose', frame_with_preds)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            logger.info("Ukončuji program.")
            break
        if SAVE_FRAMES and (key == ord('s') or key == ord('S')):
            filename_pred = os.path.join(
                session_dir, f"frame_{
                    frame_counter:06d}_preds.png")
            try:
                cv2.imwrite(filename_pred, frame_with_preds)
                logger.info("Snímek uložen: %s", filename_pred)
                frame_counter += 1
            except Exception as e:
                logger.error("Chyba při ukládání snímku: %s", e)
        if key == ord('c') or key == ord('C'):
            apply_correction = not apply_correction
            print(
                "Korekce bodů mřížky (Homografie): {}".format(
                    'Zapnuto' if apply_correction else 'Vypnuto'))

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Kamera uvolněna, okna zavřena.")


if __name__ == "__main__":
    try:
        logger.info("Python Version: %s", sys.version)
        logger.info("PyTorch Version: %s", torch.__version__)
        try:
            import torchvision
            logger.info("Torchvision Version: %s", torchvision.__version__)
        except ImportError:
            logger.warning("Torchvision není nainstalováno.")
        ultralytics_ver_str = ultralytics_version if 'ultralytics_version' in locals() else 'N/A'
        logger.info("Ultralytics Version: %s", ultralytics_ver_str)
        logger.info("OpenCV Version: %s", cv2.__version__)
    except Exception as e:
        logger.error("Chyba při zjišťování verzí: %s", e)
    main()
