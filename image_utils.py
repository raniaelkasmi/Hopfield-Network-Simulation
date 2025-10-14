import cv2
import numpy as np
from sklearn.preprocessing import Binarizer

def load_and_binarize_image(image_path):
    """
    Load an image, binarize it, and convert pixel values to +1 and -1.

    Parameters:
    -----------
    image_path : str
        Path to the image file to be loaded.

    Returns:
    --------
    binarized_image : numpy.ndarray
        A 100x100 array with pixel values binarized to +1 and -1.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    _, binarized_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binarized_image = (binarized_image / 255).astype(int)
    binarized_image = 2 * binarized_image - 1
    
    return binarized_image
