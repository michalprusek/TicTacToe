import os
import shutil
import cv2
import torch
# import numpy as np # Implicitly used
import logging
# import sys # Not used
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ultralytics import YOLO
from typing import List, Dict, Any

# --- Configuration --- #
INPUT_DIR = "unique_frames"                # Source directory (unique images)
# Target directory (batches & annotations)
BATCH_OUTPUT_DIR = "batched_unique_frames"
BATCH_SIZE = 1000                          # Images per batch folder
DETECT_MODEL_PATH = "weights/best_detection.pt"  # Path to symbol detection model
POSE_MODEL_PATH = "weights/best_pose.pt"   # Path to grid pose estimation model
BBOX_CONF_THRESHOLD = 0.5                  # Bbox detection confidence threshold
POSE_CONF_THRESHOLD = 0.5                  # Pose estimation object conf threshold
KEYPOINT_VISIBLE_THRESHOLD = 0.5           # Individual keypoint visibility conf
# Output annotation filename per batch
OUTPUT_XML_FILENAME = "annotations.xml"
LABELS = {
    "box": ["X", "O"],
    "points": ["grid"]
}

# --- Setup Logging --- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_batches(input_dir: str, output_dir: str, batch_size: int):
    """Copies images from input_dir to numbered batch subdirs in output_dir."""
    logger.info("Creating batches from '%s' to '%s'...", input_dir, output_dir)
    if not os.path.isdir(input_dir):
        logger.error("Input directory '%s' not found.", input_dir)
        return False

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ])

    if not image_files:
        logger.warning("No image files found in '%s'.", input_dir)
        return False

    num_batches = (len(image_files) + batch_size - 1) // batch_size
    logger.info(
        "Found %d imgs, creating %d batches.", len(image_files), num_batches)

    for i in range(num_batches):
        batch_num = i + 1
        batch_dir_name = f"batch_{batch_num:03d}"
        batch_path = os.path.join(output_dir, batch_dir_name)
        os.makedirs(batch_path, exist_ok=True)

        start_index = i * batch_size
        end_index = start_index + batch_size
        batch_files = image_files[start_index:end_index]

        logger.info("Copying %d files to '%s'...", len(batch_files), batch_path)
        for filename in batch_files:
            src_path = os.path.join(input_dir, filename)
            dest_path = os.path.join(batch_path, filename)
            try:
                shutil.copy2(src_path, dest_path)
            except Exception as e:
                logger.error("Failed to copy %s to %s: %s", src_path, dest_path, e)

    logger.info("Finished creating batches.")
    return True


def generate_cvat_xml(image_data: List[Dict[str, Any]],
                      labels: Dict[str, List[str]],
                      output_xml_path: str):
    """Generates a CVAT XML 1.1 annotation file."""
    logger.info(f"Generating CVAT XML: {os.path.basename(output_xml_path)}")
    annotations = ET.Element('annotations')
    version = ET.SubElement(annotations, 'version')
    version.text = '1.1'

    # Meta section with labels
    meta = ET.SubElement(annotations, 'meta')
    task = ET.SubElement(meta, 'task')
    labels_elem = ET.SubElement(task, 'labels')
    # Add box labels
    for label_name in labels.get("box", []):
        label_elem = ET.SubElement(labels_elem, 'label')
        name_elem = ET.SubElement(label_elem, 'name')
        name_elem.text = label_name
        color_elem = ET.SubElement(label_elem, 'color')
        color = ("#FF0000" if label_name == "X" else
                 "#00FF00" if label_name == "O" else "#0000FF")
        color_elem.text = color
        ET.SubElement(label_elem, 'attributes')  # Required, even if empty
    # Add points labels
    for label_name in labels.get("points", []):
        label_elem = ET.SubElement(labels_elem, 'label')
        name_elem = ET.SubElement(label_elem, 'name')
        name_elem.text = label_name
        color_elem = ET.SubElement(label_elem, 'color')
        color_elem.text = "#FFFF00"  # Yellow for grid points
        type_elem = ET.SubElement(label_elem, 'type')
        type_elem.text = 'points'
        ET.SubElement(label_elem, 'attributes')  # Required, even if empty

    # Add image annotations
    for img_info in image_data:
        image_elem = ET.SubElement(
            annotations, 'image', id=str(
                img_info['id']), name=img_info['name'], width=str(
                img_info['width']), height=str(
                img_info['height']))

        # Add bounding boxes
        for box in img_info.get('boxes', []):
            ET.SubElement(image_elem, 'box', label=box['label'], occluded='0',
                          xtl=f"{box['xtl']:.2f}", ytl=f"{box['ytl']:.2f}",
                          xbr=f"{box['xbr']:.2f}", ybr=f"{box['ybr']:.2f}")

        # Add keypoints
        for points in img_info.get('points', []):
            visible_kpts = [
                f"{p[0]:.2f},{p[1]:.2f}"
                for p in points['coords'] if p[2] >= KEYPOINT_VISIBLE_THRESHOLD
            ]
            if visible_kpts:
                points_str = ";".join(visible_kpts)
                ET.SubElement(image_elem, 'points', label=points['label'],
                              occluded='0', points=points_str)

    # Pretty print XML
    xml_str = ET.tostring(annotations, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent='  ')

    try:
        with open(output_xml_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml_str)
        logger.info(f"Successfully saved annotations to '{output_xml_path}'.")
    except Exception as e:
        logger.error(f"Failed to write XML file '{output_xml_path}': {e}")


# --- Main Function --- #
def main():
    # 1. Create Batches
    if not create_batches(INPUT_DIR, BATCH_OUTPUT_DIR, BATCH_SIZE):
        logger.error("Failed to create image batches. Exiting.")
        return

    # 2. Determine Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # 3. Load Models
    logger.info("Loading YOLO models...")
    try:
        detect_model = YOLO(DETECT_MODEL_PATH)
        pose_model = YOLO(POSE_MODEL_PATH)
        detect_model.to(device)
        pose_model.to(device)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}. Exiting.")
        return

    # 4. Process Each Batch
    batch_dirs = sorted([d for d in os.listdir(BATCH_OUTPUT_DIR) if os.path.isdir(
        os.path.join(BATCH_OUTPUT_DIR, d)) and d.startswith('batch_')])

    if not batch_dirs:
        logger.warning(f"No batch directories found in '{BATCH_OUTPUT_DIR}'.")
        return

    total_images_processed = 0
    for batch_idx, batch_dir_name in enumerate(batch_dirs):
        batch_path = os.path.join(BATCH_OUTPUT_DIR, batch_dir_name)
        logger.info(
            f"--- Processing {batch_dir_name} ({batch_idx + 1}/{len(batch_dirs)}) ---")

        image_files = sorted([
            f for f in os.listdir(batch_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ])

        if not image_files:
            logger.warning(f"No images found in '{batch_path}'. Skipping.")
            continue

        batch_image_data = []  # Store annotation data for images in this batch

        for img_idx, filename in enumerate(image_files):
            image_path = os.path.join(batch_path, filename)
            # Global ID for XML consistency across batches
            current_image_id = (batch_idx * BATCH_SIZE) + img_idx
            logger.debug(
                f"Processing img {img_idx + 1}/{len(image_files)}: {filename} (ID: {current_image_id})")

            try:
                # Read image for dimensions and processing
                frame = cv2.imread(image_path)
                if frame is None:
                    logger.warning(
                        f"Could not read image: {image_path}. Skip.")
                    continue
                img_height, img_width = frame.shape[:2]

                # Run Inference
                detect_results = detect_model(
                    frame, verbose=False, conf=BBOX_CONF_THRESHOLD, device=device)
                pose_results = pose_model(
                    frame,
                    verbose=False,
                    conf=POSE_CONF_THRESHOLD,
                    device=device)

                # --- Extract Annotations ---
                img_annotations = {
                    'id': current_image_id,
                    # Relative path from the XML file location (batch folder)
                    'name': filename,
                    'width': img_width,
                    'height': img_height,
                    'boxes': [],
                    'points': []
                }

                # Extract BBoxes
                if detect_results and isinstance(detect_results, list):
                    for result in detect_results:  # Should be only one
                        if hasattr(result,
                                   'boxes') and result.boxes is not None:
                            boxes_data = result.boxes.data.cpu().numpy()
                            # {0: 'O', 1: 'X'} or similar
                            class_names = result.names
                            for box_data in boxes_data:
                                if len(box_data) >= 6:
                                    x1, y1, x2, y2, score, class_id_float = box_data[:6]
                                    class_id = int(class_id_float)
                                    label_name = class_names.get(
                                        class_id, f"CLS_{class_id}")
                                    # Ensure label is expected
                                    if label_name in LABELS["box"]:
                                        img_annotations['boxes'].append({
                                            'label': label_name,
                                            'xtl': x1, 'ytl': y1,
                                            'xbr': x2, 'ybr': y2
                                        })

                # Extract Keypoints
                pose_data_valid = (
                    pose_results and pose_results[0].keypoints and pose_results[0].boxes and len(
                        pose_results[0].boxes.data) > 0)
                if pose_data_valid:
                    grid_box_conf = pose_results[0].boxes.data[0][4].item()
                    # Check overall pose detection confidence
                    if grid_box_conf > POSE_CONF_THRESHOLD:
                        kpts_data = pose_results[0].keypoints.data.cpu(
                        ).numpy()
                        # Expect only one instance of the grid
                        if kpts_data.shape[0] > 0:
                            # Shape (N_kpts, 3) -> (u, v, conf)
                            kpts_instance = kpts_data[0]
                            img_annotations['points'].append({
                                'label': 'grid',  # Label for the points group
                                'coords': kpts_instance
                            })

                batch_image_data.append(img_annotations)
                total_images_processed += 1

            except Exception as e:
                # Use logger.exception for traceback
                logger.exception(f"ERROR processing image {image_path}: {e}")

        # 5. Generate XML for the completed batch
        if batch_image_data:  # Only generate if we processed images
            output_xml_path = os.path.join(batch_path, OUTPUT_XML_FILENAME)
            generate_cvat_xml(batch_image_data, LABELS, output_xml_path)
        else:
            logger.warning(
                f"No images successfully processed in {batch_dir_name}, skipping XML generation.")

    logger.info(f"\n--- Annotation process finished ---")
    logger.info(f"Total images processed: {total_images_processed}")
    logger.info(
        f"Annotations saved in respective batch folders within '{BATCH_OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
