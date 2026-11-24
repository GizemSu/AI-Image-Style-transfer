
from PIL import Image

# Open the image
image = Image.open("images/input.png")

# Display image information
print("Image size:", image.size)
print("Image format:", image.format)

# Resize the image
resized_image = image.resize((512, 512))

# Save the resized image
resized_image.save("images/resized.png")

print("Image successfully resized!")


import numpy as np

# Convert image to RGB
rgb_image = resized_image.convert("RGB")

# Convert image to NumPy array
pixels = np.array(rgb_image)

print("Array shape:", pixels.shape)
print("First pixel:", pixels[0, 0])


# Convert RGB image to grayscale
gray = (
    0.299 * pixels[:, :, 0] +
    0.587 * pixels[:, :, 1] +
    0.114 * pixels[:, :, 2]
)

# Convert values to integers
gray = gray.astype(np.uint8)

# Save grayscale image
gray_image = Image.fromarray(gray)
gray_image.save("images/grayscale.png")

print("Grayscale image saved!")

import matplotlib.pyplot as plt

# Create comparison
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title("Original Image")
plt.axis("off")

# Grayscale image
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

#save comparison
plt.tight_layout()
plt.savefig("images/comparison.png")
plt.show()


