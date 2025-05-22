import os
import shutil
from PIL import Image
import numpy as np

# Global counter for unique image filenames
unique_image_counter = 0


def compute_dhash(image, hash_size=8):
    """Compute the difference hash of an image"""
    # Resize the input image, adding a single column (width) so we can compute
    # the gradient
    resized = image.convert('L').resize((hash_size + 1, hash_size))
    pixels = np.array(resized)
    # compute differences between adjacent column pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert the binary array to a numerical hash value
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])


def process_directory(src_dir, dest_root, processed_hashes):
    """Process directory, copy unique images to flat destination."""
    global unique_image_counter  # Use the global counter

    # Get all image files in the directory
    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not files:
        return 0, 0  # Return counts of unique and duplicates found in this dir

    unique_in_dir = 0
    duplicate_in_dir = 0

    # Process each image
    for filename in files:
        filepath = os.path.join(src_dir, filename)
        try:
            with Image.open(filepath) as img:
                # Compute the hash
                img_hash = compute_dhash(img)

                # Check if we've seen this hash before ACROSS ALL DIRS
                if img_hash not in processed_hashes:
                    # New unique image
                    processed_hashes.add(img_hash)
                    unique_image_counter += 1  # Increment global counter
                    # Generate new filename (e.g., 000001.png)
                    # Let's keep the original extension
                    _, ext = os.path.splitext(filename)
                    if not ext:  # Handle cases with no extension if necessary
                        ext = '.png'  # Default extension
                    new_filename = f"{unique_image_counter:06d}{ext}"
                    dest_path = os.path.join(dest_root, new_filename)
                    shutil.copy2(filepath, dest_path)
                    unique_in_dir += 1
                else:
                    # Duplicate image (found either in this dir or a previous
                    # one)
                    duplicate_in_dir += 1
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    return unique_in_dir, duplicate_in_dir


def main():
    # Define paths
    input_root = "/Users/michalprusek/PycharmProjects/TicTacToe/dataset_frames"
    output_root = "/Users/michalprusek/PycharmProjects/TicTacToe/unique_frames"

    # Handle output directory existence
    if os.path.exists(output_root):
        print(f"Output dir exists: {os.path.basename(output_root)}. Reusing.")
        # Optional: Clear the output directory before starting
        # Be careful with this! Uncomment only if sure.
        # print(f"Output directory {output_root} exists. Clearing it.")
        # for item in os.listdir(output_root):
        #     item_path = os.path.join(output_root, item)
        #     if os.path.isfile(item_path) or os.path.islink(item_path):
        #         os.unlink(item_path)
        #     elif os.path.isdir(item_path):
        #         shutil.rmtree(item_path)
        pass  # Currently does nothing if exists, change if needed
    else:
        os.makedirs(output_root, exist_ok=True)

    # Get all subdirectories to process from input_root
    subdirs = [
        os.path.join(input_root, d) for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ]

    # Set to track processed image hashes across all directories
    processed_hashes = set()

    # Process each subdirectory
    total_processed_files = 0
    total_duplicates_found = 0  # Counts duplicates *within* the run

    print(f"Starting processing from: {input_root}")
    print(f"Saving unique images to: {output_root}")

    for subdir in subdirs:
        print(f"Processing directory: {subdir}...")
        unique_in_dir, duplicate_in_dir = process_directory(
            subdir, output_root, processed_hashes
        )
        total_processed_files += (unique_in_dir + duplicate_in_dir)
        total_duplicates_found += duplicate_in_dir
        print(f"  Found {unique_in_dir} new unique images, "
              f"{duplicate_in_dir} duplicates in this directory.")

    print("\nCompleted processing all directories.")
    print(f"Total files processed: {total_processed_files}")
    # unique_image_counter holds the final count of unique images saved
    print(f"Total unique images saved: {unique_image_counter}")
    print(f"Total duplicate images found: {total_duplicates_found}")
    print(f"Unique images saved to: {output_root}")


if __name__ == "__main__":
    main()
