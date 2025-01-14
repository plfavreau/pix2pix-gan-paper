import os

import cv2
import matplotlib.pyplot as plt


def load_image(image_path):
    return cv2.imread(image_path)


def split_image(image):
    w = image.shape[1] // 2
    image_real = image[:, :w, :]
    image_cond = image[:, w:, :]
    return image_real, image_cond


def display_images(images, titles, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def display_multiple_images(image_paths, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    for i, img_path in enumerate(image_paths):
        img = load_image(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"Image {os.path.basename(img_path)}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    path = "facades/train/"
    image_path = f"{path}91.jpg"
    image = load_image(image_path)
    print("Shape of the image: ", image.shape)
    image_real, image_cond = split_image(image)

    display_images([image_real, image_cond], ["Real", "Condition"], figsize=(18, 6))

    image_names = ["91.jpg", "92.jpg", "93.jpg", "94.jpg", "95.jpg"]
    image_paths = [f"{path}{img_name}" for img_name in image_names]
    display_multiple_images(image_paths)


if __name__ == "__main__":
    main()
