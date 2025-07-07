import os
import random
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np


class ImageAugmentor:
    def __init__(self):
        pass

    def augment(self, image):
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Random operations
        if random.random() < 0.5:
            image_cv = self.random_rotate(image_cv)
        if random.random() < 0.5:
            image_cv = self.random_blur(image_cv)
        if random.random() < 0.5:
            image_cv = self.random_translate(image_cv)
        if random.random() < 0.5:
            image_cv = self.random_skew(image_cv)

        # Convert back to PIL
        image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        if random.random() < 0.5:
            image = ImageOps.invert(image)
        if random.random() < 0.5:
            image = self.change_brightness(image)
        if random.random() < 0.5:
            image = self.change_contrast(image)

        return image

    def random_rotate(self, image):
        angle = random.uniform(-30, 30)
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))

    def random_blur(self, image):
        ksize = random.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def random_translate(self, image):
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    def random_skew(self, image):
        rows, cols, ch = image.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32(
            [
                [50 + random.randint(-20, 20), 50 + random.randint(-20, 20)],
                [200 + random.randint(-20, 20), 50 + random.randint(-20, 20)],
                [50 + random.randint(-20, 20), 200 + random.randint(-20, 20)],
            ]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(image, M, (cols, rows))

    def change_brightness(self, image):
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.5, 1.5)
        return enhancer.enhance(factor)

    def change_contrast(self, image):
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.5, 1.5)
        return enhancer.enhance(factor)


def augment_images_in_directory(
    input_dir,
    output_dir,
    augmentor,
    image_extensions={".jpg", ".jpeg", ".png", ".bmp", ".tiff"},
):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_dir, filename)
            try:
                image = Image.open(input_path).convert("RGB")
                augmented_image = augmentor.augment(image)
                output_path = os.path.join(output_dir, f"aug_{filename}")
                augmented_image.save(output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":

    resources_dir = os.getenv("RESOURCES_DIR", "image_augmentation/resources")
    input_directory = os.path.join(resources_dir, "images")
    output_directory = os.path.join(resources_dir, "augmented_images")
    augmentor = ImageAugmentor()

    augment_images_in_directory(input_directory, output_directory, augmentor)
    print("Image augmentation completed.")
