import os
import cv2
import numpy as np

def enhance_high_res(input_folder, output_folder):
    """
    Enhance images to simulate higher resolution while keeping the size constant.

    Args:
    - input_folder: Path to the folder with input images.
    - output_folder: Path to save the processed images.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Step 1: Read the image
            img = cv2.imread(input_path)

            # Step 2: Enhance contrast using CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)  # Apply CLAHE to the L channel
            enhanced_lab = cv2.merge((l, a, b))
            contrast_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Step 3: Apply gentle sharpening
            gaussian_blur = cv2.GaussianBlur(contrast_img, (3, 3), sigmaX=1)
            sharpened_img = cv2.addWeighted(contrast_img, 1.5, gaussian_blur, -0.5, 0)

            # Save the processed image with the same filename in the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, sharpened_img)

            print(f"Processed and saved: {output_path}")

# Example usage
input_folder = "./drone_dataset/test/images"
output_folder = "./drone_dataset_contrast/test/images"
enhance_high_res(input_folder, output_folder)
