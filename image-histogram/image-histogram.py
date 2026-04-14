import numpy as np
def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    return list(np.bincount(np.array(image).reshape(-1), minlength=256))