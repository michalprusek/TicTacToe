# @generated [partially] Claude Code 2025-01-01: AI-assisted code review and pylint fixes
import cv2
import os

# --- Configuration ---
DATASET_DIR = "dataset_frames"  # Main directory to store session folders
CAMERA_INDEX = 0  # Index of the camera (0 = default built-in webcam)
FILENAME_PREFIX = "frame"  # Prefix for saved image files
IMAGE_FORMAT = ".png"  # Format for saved images (e.g., .jpg, .png)
EXIT_KEY = ord('q')  # Key to press to exit the program
TOGGLE_SAVE_KEY = ord('r')  # Key to press to toggle saving

# --- State Variables ---
is_saving = False
session_counter = 0
frame_counter = 0
current_save_path = ""


# --- Functions ---
def get_next_session_dir(base_dir: str) -> str:
    """Finds the next available session directory name."""
    counter = 1
    while True:
        session_dir = os.path.join(base_dir, f"session_{counter}")
        if not os.path.exists(session_dir):
            return session_dir
        counter += 1


# --- Main Script ---

# Create the main dataset directory if it doesn't exist
os.makedirs(DATASET_DIR, exist_ok=True)
print(f"Saving sessions to directory: '{DATASET_DIR}'")

# Initialize video capture
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
    exit()

print(f"Camera {CAMERA_INDEX} opened successfully.")
print(f"Press '{chr(TOGGLE_SAVE_KEY)}' to start/stop saving frames.")
print(f"Press '{chr(EXIT_KEY)}' to quit.")

cv2.namedWindow("Camera Feed")

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # --- Frame Processing and Display ---
    display_frame = frame.copy()  # Work on a copy to draw status text

    # Add status text to the frame
    status_text = "Status: Idle"
    color = (0, 0, 255)  # Red for Idle
    if is_saving:
        status_text = (f"Status: SAVING to "
                       f"{os.path.basename(current_save_path)} | "
                       f"Frame: {frame_counter}")
        color = (0, 255, 0)  # Green for Saving
    cv2.putText(display_frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Create the second status text separately for line length
    press_key_text = (f"Press '{chr(TOGGLE_SAVE_KEY).upper()}'=Toggle Save, "
                      f"'{chr(EXIT_KEY).upper()}'=Quit")
    cv2.putText(display_frame, press_key_text,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Camera Feed", display_frame)

    # --- Handle Key Presses ---
    key = cv2.waitKey(1) & 0xFF  # Use waitKey(1) for a responsive feed

    if key == EXIT_KEY:
        print("Exit key pressed. Quitting...")
        break
    elif key == TOGGLE_SAVE_KEY:
        if not is_saving:
            # Start saving
            is_saving = True
            current_save_path = get_next_session_dir(DATASET_DIR)
            os.makedirs(current_save_path, exist_ok=True)
            frame_counter = 0
            print(f"\n[INFO] Started saving session to: {current_save_path}")
        else:
            # Stop saving
            is_saving = False
            print(f"[INFO] Stopped saving. Saved {frame_counter} frames in "
                  f"{current_save_path}")
            current_save_path = ""  # Reset path

    # --- Save Frame ---
    if is_saving:
        # Format frame number with leading zeros for better sorting
        filename = f"{FILENAME_PREFIX}_{frame_counter:05d}{IMAGE_FORMAT}"
        save_path_full = os.path.join(current_save_path, filename)
        try:
            cv2.imwrite(save_path_full, frame)
            frame_counter += 1
        except Exception as e:
            print(f"Error saving frame {filename} to {current_save_path}: {e}")
            # Optionally stop saving on error or just log it
            # is_saving = False
            # print("[ERROR] Stopping saving due to write error.")


# --- Cleanup ---
if is_saving:
    print(f"[INFO] Finalizing. Saved {frame_counter} frames in the last "
          f"session: {current_save_path}")

cap.release()
cv2.destroyAllWindows()
print("Resources released. Script finished.")
