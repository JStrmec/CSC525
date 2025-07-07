import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import unittest
import urllib.request
from PIL import Image
import shutil
from image_augmentation.image_augmentor import (
    ImageAugmentor,
    augment_images_in_directory,
)

class TestImageAugmentation(unittest.TestCase):
    def setUp(self):
        self.resources_dir = os.getenv("RESOURCES_DIR", "image_augmentation/resources")
        self.input_dir = os.path.join(self.resources_dir, "images")
        self.output_dir = os.path.join(self.resources_dir, "augmented_images_test")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.sample_images = [
            (
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/640px-PNG_transparency_demonstration_1.png",
                "sample1.png",
            ),
            (
                "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Example.jpg/640px-Example.jpg",
                "sample2.jpg",
            ),
        ]

        for url, filename in self.sample_images:
            file_path = os.path.join(self.input_dir, filename)
            if not os.path.exists(file_path):
                urllib.request.urlretrieve(url, file_path)

    def test_valid_augmentation(self):
        augmentor = ImageAugmentor()
        augment_images_in_directory(self.input_dir, self.output_dir, augmentor)
        files = os.listdir(self.output_dir)
        self.assertGreaterEqual(
            len(files), 2, "Expected at least two augmented images."
        )

        for file in files:
            try:
                img = Image.open(os.path.join(self.output_dir, file))
                img.verify()  # Confirms it's a valid image
            except Exception as e:
                self.fail(f"Augmented image '{file}' is invalid: {e}")

    def test_empty_input_directory(self):
        empty_dir = os.path.join(self.resources_dir, "empty")
        output_dir = os.path.join(self.resources_dir, "empty_augmented")
        os.makedirs(empty_dir, exist_ok=True)
        augment_images_in_directory(empty_dir, output_dir, ImageAugmentor())
        self.assertEqual(
            len(os.listdir(output_dir)),
            0,
            "No images should be processed in an empty directory.",
        )
        shutil.rmtree(empty_dir)

    def test_invalid_image_file(self):
        # Create a fake image file
        bad_file_path = os.path.join(self.input_dir, "fake_image.jpg")
        with open(bad_file_path, "w") as f:
            f.write("This is not a real image")

        augmentor = ImageAugmentor()
        augment_images_in_directory(self.input_dir, self.output_dir, augmentor)

        # Should only augment the valid images, and skip the corrupted one
        augmented = os.listdir(self.output_dir)
        self.assertTrue(
            any(f.startswith("aug_") for f in augmented),
            "Should still process valid images.",
        )
        self.assertNotIn(
            "aug_fake_image.jpg", augmented, "Corrupted image should not be processed."
        )

    def test_non_image_files_ignored(self):
        txt_file = os.path.join(self.input_dir, "notes.txt")
        with open(txt_file, "w") as f:
            f.write("Not an image.")

        augmentor = ImageAugmentor()
        augment_images_in_directory(self.input_dir, self.output_dir, augmentor)

        augmented_files = os.listdir(self.output_dir)
        self.assertTrue(
            all(
                f.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
                for f in augmented_files
            ),
            "Output should only include images.",
        )

    def tearDown(self):
        #shutil.rmtree(self.input_dir, ignore_errors=True)
        #shutil.rmtree(self.output_dir, ignore_errors=True)
        pass
