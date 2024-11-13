import os
import cv2
import random
import numpy as np


# Augmentation functions
def change_contrast(image, alpha_range=(0.6, 1.6)):
    """Adjust contrast randomly within the given range."""
    alpha = random.uniform(*alpha_range)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def change_brightness(image, beta_range=(-30, 30)):
    """Adjust brightness randomly within the given range."""
    beta = random.randint(*beta_range)
    return cv2.convertScaleAbs(image, alpha=1, beta=beta)


def flip_horizontal(image, labels):
    """Flip image horizontally and adjust labels."""
    flipped_image = cv2.flip(image, 1)
    new_labels = []
    for label in labels:
        label[1] = 1 - label[1]
        new_labels.append(label)
    return flipped_image, new_labels

"""
def color_shift(image, shift_range=(-20, 20)):
    #shift color
    shift_values = np.random.randint(*shift_range, size=(3,))
    shifted_image = image.copy()
    for i in range(3):  # Apply shift to each channel
        shifted_image[:, :, i] = np.clip(image[:, :, i] + shift_values[i], 0, 255)
    return shifted_image
"""


def gaussian_blur(image, kernel_range=(3, 9)):
    # Choose a random odd kernel size within the range
    kernel_size = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def add_noise(image, noise_level=30):
    """Add Gaussian noise to the image."""
    h, w, c = image.shape
    noise = np.random.normal(0, noise_level, (h, w, c)).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


# Read label file
def read_labels(label_path):
    with open(label_path, 'r') as f:
        labels = [
            [int(parts[0])] + list(map(float, parts[1:]))
            for parts in (line.split() for line in f)
        ]
    return labels


# Write label file
def write_labels(label_path, labels):
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(' '.join(map(str, label)) + '\n')


# Augment dataset
def augment_dataset(input_folder, output_folder, n_augments=20, target_negatives=3000):
    image_folder = os.path.join(input_folder, 'images')
    label_folder = os.path.join(input_folder, 'labels')

    output_image_folder = os.path.join(output_folder, 'images')
    output_label_folder = os.path.join(output_folder, 'labels')

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    negative_count = 0
    positive_files = []
    negative_files = []

    # Separate positive and negative examples
    for label_file in os.listdir(label_folder):
        label_path = os.path.join(label_folder, label_file)
        labels = read_labels(label_path)
        if all(label[0] == 1 for label in labels):  # All labels are negative
            negative_files.append(label_file)
        else:  # At least one positive label
            positive_files.append(label_file)

    # Copy positive examples
    for label_file in positive_files:
        image_file = label_file.replace('.txt', '.jpg')
        cv2.imwrite(os.path.join(output_image_folder, image_file), cv2.imread(os.path.join(image_folder, image_file)))
        write_labels(os.path.join(output_label_folder, label_file), read_labels(os.path.join(label_folder, label_file)))

    # Augment negative examples
    for label_file in negative_files:
        if negative_count >= target_negatives:
            break
        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)

        image = cv2.imread(image_path)
        labels = read_labels(label_path)

        # Generate one horizontal flip
        flipped_image, flipped_labels = flip_horizontal(image, labels)
        aug_image_file = f"{os.path.splitext(image_file)[0]}_flip.jpg"
        aug_label_file = f"{os.path.splitext(label_file)[0]}_flip.txt"
        cv2.imwrite(os.path.join(output_image_folder, aug_image_file), flipped_image)
        write_labels(os.path.join(output_label_folder, aug_label_file), flipped_labels)
        negative_count += 1

        # Generate random augmentations
        for i in range(n_augments - 1):  # Exclude one slot for flip
            if negative_count >= target_negatives:
                break

            # Randomly choose an augmentation
            aug_type = random.choice(['contrast', 'brightness', 'gaussian_blur', 'noise'])
            if aug_type == 'contrast':
                aug_image = change_contrast(image)
            elif aug_type == 'brightness':
                aug_image = change_brightness(image)
            elif aug_type == 'gaussian_blur':
                aug_image = gaussian_blur(image)
            elif aug_type == 'noise':
                aug_image = add_noise(image)

            # Save augmented image and label
            aug_image_file = f"{os.path.splitext(image_file)[0]}_aug_{negative_count}.jpg"
            aug_label_file = f"{os.path.splitext(label_file)[0]}_aug_{negative_count}.txt"

            cv2.imwrite(os.path.join(output_image_folder, aug_image_file), aug_image)
            write_labels(os.path.join(output_label_folder, aug_label_file), labels)

            negative_count += 1

            #if i == (n_augments - 2):
            #    print("finished processing one negative sample!")


# Example usage
input_folder = 'drone_dataset/train'
output_folder = 'drone_augmented/train'
augment_dataset(input_folder, output_folder, n_augments=20, target_negatives=3000)
