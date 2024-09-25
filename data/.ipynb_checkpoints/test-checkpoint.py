from PIL import Image
import os

def resize_images(directory, size):
    """
    Resize all images in the given directory to the specified size.

    Args:
    directory (str): The path to the directory containing the images.
    size (tuple): The new size for the images as a (width, height) tuple.
    """
    # Walk through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Open the image file
            with Image.open(file_path) as image:
                # Resize the image
                resized_image = image.resize(size)
                # Save the resized image
                resized_image.save(file_path)

# Call the function with the directory path and desired size
resize_images("./walk/walk/",(432,240))
# resize_images("./walk_mask/walk_mask/",(432,240))