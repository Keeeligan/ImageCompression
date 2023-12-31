import numpy as np
import pandas as pd


def run_simple_algorithm(img: np.ndarray):
    """
    Applies the simple algorithm to compress an input image.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        np.ndarray: The compressed image values.
    """
    img = colour_compress(img)
    img = run_length_enc(img)

    return img


def colour_compress(img: np.ndarray):
    """
    Compresses the image by shifting all the RGB values of each pixel to the closest multiple of four.

    This compression technique aims to improve the effectiveness of the Run Length Encoding (RLE) algorithm
    at the cost of reducing the color depth.

    Args:
        img (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: Compressed image as a NumPy array.

    """
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            rgb = []
            # Append each floored value of RGB to the RGB list
            for i in range(len(img[y, x])):
                rgb.append(img[y, x][i] - (img[y, x][i] % 4))
            img[y, x] = np.array(rgb)
    return img


def run_length_enc(img: np.ndarray) -> list:
    """
    Performs Run Length Encoding (RLE) on the image to compress its pixel data.

    Args:
        img (np.ndarray): Input image as a NumPy array.

    Returns:
        list: RLE-encoded representation of the image, where each pixel value is represented as a tuple
        containing the RGB values and the count of consecutive occurrences.

    """
    img_rle = []

    for y in range(img.shape[0]):
        row = []
        count = 1
        for x in range(img.shape[1] - 1):
            # If the RGB values are identical, add to the counter
            if img[y, x][0] == img[y, x + 1][0] and \
                    img[y, x][1] == img[y, x + 1][1] and \
                    img[y, x][2] == img[y, x + 1][2]:
                count += 1
            else:
                # Append the last count to the row and reset the counter
                row.append((img[y, x].tolist(), count))
                count = 1
        row.append((img[y, x].tolist(), count))
        img_rle.append(row)

    return img_rle


def build_image(image_name: str, directory="images/STORE/"):
    """
    Builds an image based on the Run Length Encoded (RLE) data.

    Args:
        image_name (str): The name of the image without the file extension.
        directory (str, optional): The directory where the RLE data is stored. Defaults to "images/STORE/".

    Returns:
        np.ndarray: The reconstructed image as a NumPy array.

    """
    # Open the compressed data
    rle = pd.read_pickle(f'{directory}{image_name}')

    # Initialize the array with the original resolution
    res = []
    for y in range(len(rle)):
        row = []
        for x in range(len(rle[y])):
            # Append the value to the list the counter amount
            for i in range(rle[y][x][1]):
                row.append(rle[y][x][0])
        res.append(row)
    return np.array(res, dtype=np.uint8)
