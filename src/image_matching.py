
from PIL import Image, ImageOps
import imagehash


def calculate_image_similarity(image1_path, image2_path):

    # Open images and correct orientation
    image1 = ImageOps.exif_transpose(
        Image.open(image1_path)
    ).convert("RGB")

    image2 = ImageOps.exif_transpose(
        Image.open(image2_path)
    ).convert("RGB")

    # Generate hashes for multiple angles
    angles = [-15, -10, -5, 0, 5, 10, 15]

    hashes1 = []
    hashes2 = []

    for angle in angles:

        rotated1 = image1.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True
        )

        rotated2 = image2.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True
        )

        hashes1.append(imagehash.phash(rotated1))
        hashes2.append(imagehash.phash(rotated2))

    # Compare all angle combinations
    total_bits = hashes1[0].hash.size
    best_difference = min(
        hash1 - hash2
        for hash1 in hashes1
        for hash2 in hashes2
    )

    similarity = (
        1 - best_difference / total_bits
    ) * 100

    return round(max(0, similarity), 2)
