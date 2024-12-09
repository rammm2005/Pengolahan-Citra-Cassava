
def resize_image(img, max_width=800):
    """
    Resizes the image to a maximum width while maintaining the aspect ratio.

    Args:
        img (ndarray): The original image.
        max_width (int): The maximum width for the resized image.

    Returns:
        ndarray: The resized image.
    """
    height, width = img.shape[:2]
    if width > max_width:
        scaling_factor = max_width / float(width)
        new_dim = (max_width, int(height * scaling_factor))
        return cv2.resize(img, new_dim)
    return img
