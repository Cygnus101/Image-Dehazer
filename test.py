import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Define paths
TEST_DATASET_HAZY_PATH = r'C:\Users\rajat\Downloads\final_dataset (train+val)\val'
TEST_DATASET_OUTPUT_PATH = r'C:\Users\rajat\Downloads\final_dataset (train+val)\Output'

# Make sure output directory exists
if not os.path.exists(TEST_DATASET_OUTPUT_PATH):
    os.makedirs(TEST_DATASET_OUTPUT_PATH)

# List all files in the hazy images directory
input_images = os.listdir(TEST_DATASET_HAZY_PATH)
num = len(input_images)
output_images = []

# Prepare full paths for input and output
for i in range(num):
    output_images.append(os.path.join(TEST_DATASET_OUTPUT_PATH, input_images[i]))
    input_images[i] = os.path.join(TEST_DATASET_HAZY_PATH, input_images[i])

# Load your trained dehazing generator
model = load_model(r'C:\Users\rajat\Downloads\generator.h5')  # Adjust path if necessary

for i in range(num):
    # Load each image, process it and save the dehazed version
    img = Image.open(input_images[i])
    img = img.resize((256, 256))  # Resize to the required input size of the generator
    img = np.array(img) / 255.0  # Normalize the image to 0-1
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the dehazed image
    dehazed_image = model.predict(img)

    # Convert back to image
    dehazed_image = np.clip(dehazed_image[0], 0, 1)  # Clip values to ensure they are between 0 and 1
    dehazed_image = (dehazed_image * 255).astype(np.uint8)  # Scale back up to 0-255
    dehazed_image = Image.fromarray(dehazed_image)

    # Save the dehazed image
    dehazed_image.save(output_images[i])

print("Dehazing complete. Processed images saved to", TEST_DATASET_OUTPUT_PATH)
