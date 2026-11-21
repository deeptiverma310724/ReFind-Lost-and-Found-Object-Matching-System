
import sys
from pathlib import Path

# Project root ko Python path mein add karna
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.image_matching import calculate_image_similarity

image_folder = project_root / "data" / "processed" / "images"

images = list(image_folder.glob("*"))

if len(images) < 2:
    print("Please upload at least 2 images.")
else:
    image1 = str(images[0])
    image2 = str(images[1])

    print("Image 1:", image1)
    print("Image 2:", image2)

    score = calculate_image_similarity(image1, image2)

    print("Image Similarity:", score, "%")
