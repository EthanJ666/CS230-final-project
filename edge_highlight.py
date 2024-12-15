import os
import cv2

def batch_edge_highlight(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is an image (e.g., .jpg, .png)
        if os.path.isfile(input_path) and filename.lower().endswith('.jpg'):
            # Read the image in grayscale
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Apply edge highlighting (Canny edge detection in this case)
            edges = cv2.Canny(img, threshold1=100, threshold2=200)

            # Save the edge-highlighted image with the same filename
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, edges)

            print(f"Processed and saved: {output_path}")

# Example usage
input_folder = "./drone_dataset/test/images"
output_folder = "./drone_dataset_edge/test/images"
batch_edge_highlight(input_folder, output_folder)
