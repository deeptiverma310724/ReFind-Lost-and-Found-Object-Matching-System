
from src.image_matching import calculate_image_similarity

image1 = "data/processed/images/data/processed/images/6d0ba849-1c1e-481c-8121-7eca49fe33d1.png"
image2 = "data/processed/images/data/processed/images/b56f7184-12c6-4256-a931-d58855971be2.png"

score = calculate_image_similarity(image1, image2)

print("Image Similarity:", score, "%")
