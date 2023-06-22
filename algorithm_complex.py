from PIL import Image
import numpy as np
import pandas as pd

import json


def test_algorithm(img: np.ndarray):
    height, width = img.shape[0], img.shape[1]
    img.astype(int)
    img = colour_space_conversion(img)

    img = img.tolist()

    # img = chrominance_downsample(img)
    # print(f"{len(img[1])}x{len(img[1][0])}")

    # img = discrete_cosine_transform(img[0])

    return img



def build_image(image_name: str, directory="images/STORE/"):

    # Open the data
    image_name = image_name.strip(".png")
    img = pd.read_pickle(f'{directory}ew_{image_name}.npy')

    # Convert the values back to RGB
    img = colour_space_conversion(img)

    # Resize the chrominance layers

    return img


def colour_space_conversion(img:np.ndarray, from_rgb=True):
    """
    Converts all the RGB values of an image to the YCbCr color space.

    The converted color space consists of Luminance (Y), Blue Chrominance (Cb), and Red Chrominance (Cr).
    The Luminance represents the brightness, and the Blue- and Red Chrominance make up the colors.

    Note:
    Using this conversion, the Luminance can serve as a base layer. By downsampling the two Chrominance layers,
    it is possible to retain a good amount of detail/precision while reducing the amount of information.

    Args:
        img (np.ndarray): The image that needs to be converted.

    Returns:
        np.ndarray: NumPy array containing the converted pixel values.

    """
    res = []
    # Convert each rgb value to the YCbCr colour space.
    for y in range(img.shape[0]):
        row = []
        for x in range(img.shape[1]):
            if from_rgb:
                row.append(rgb_to_ycbcr(img[y, x]))
            else:
                row.append(ycbcr_to_rgb(img[y, x]))
        res.append(row)

    return np.array(res, dtype=int)


def rgb_to_ycbcr(rgb: tuple):
    """
    Converts the rbg values of a single pixel to the YCbCr colour space.
    Source: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdprfx/b550d1b5-f7d9-4a0c-9141-b3dca9d7f525

    Args:
        rgb (Tuple[int, int, int]): Tuple with the RGB values of a pixel.

    Returns:
        Tuple[float, float, float]: Tuple with the YCbCr values of the pixel.
    """

    # Shift the RGB values down by 128.
    r, g, b = [max(min(x, 127), -128) for x in rgb]

    # Create the YCbCr values based on the rgb values of the image.
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) / (2 * (1 - 0.114))
    cr = (r - y) / (2 * (1 - 0.299))
    return round(y), round(cb), round(cr)

def ycbcr_to_rgb(ycbcr: tuple):
    """
    Converts the YCbCr values of a single pixel to the RGB color space.

    Args:
        ycbcr (Tuple[float, float, float]): Tuple with the YCbCr values of a pixel.

    Returns:
        Tuple[int, int, int]: Tuple with the RGB values of the pixel.

    """
    y, cb, cr = ycbcr

    # Shift the YCbCr values up by 128.
    cb_shifted = cb * (2 * (1 - 0.114))
    cr_shifted = cr * (2 * (1 - 0.299))

    # Calculate the RGB values based on the YCbCr values.
    r = y + 1.402 * cr_shifted
    g = y - 0.344 * cb_shifted - 0.714 * cr_shifted
    b = y + 1.772 * cb_shifted

    # Round the RGB values and clip them to the valid range [0, 255].
    r = max(0, min(255, round(r)))
    g = max(0, min(255, round(g)))
    b = max(0, min(255, round(b)))

    return r, g, b


def chrominance_downsample(img: np.ndarray):
    """
    Takes the YCbCr values of an image and creates 2x2 blocks, averaging the values in each block to
    create one pixel.

    Args:
        img (np.ndarray): NumPy array with the YCbCr values of an image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the luminance layer and the two downsampled
        chrominance layers.

    """
    # Initialize the layers with the correct shape.
    luminance = np.ndarray(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
    blue_chrominance = np.ndarray(shape=(img.shape[0] // 2, img.shape[1] // 2), dtype=np.float32)
    red_chrominance = np.ndarray(shape=(img.shape[0] // 2, img.shape[1] // 2), dtype=np.float32)

    # Fill the luminance layer with the same values.
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            luminance[y, x] = img[y, x, 0]

    # Fill the chrominance layers that are 1/4 of the size.
    for y in range(0, img.shape[0]-1, 2):
        for x in range(0, img.shape[1]-1, 2):
            blue_chrominance[y // 2, x // 2] = np.average([img[y, x, 1], img[y, x + 1, 1], img[y + 1, x, 1], img[y + 1, x + 1, 1]])
            red_chrominance[y // 2, x // 2] = np.average([img[y, x, 2], img[y, x + 1, 2], img[y + 1, x, 2], img[y + 1, x + 1, 2]])

    return luminance, blue_chrominance, red_chrominance


def discrete_cosine_transform(img: np.ndarray, block_size=8):
    """
    Applies the Discrete Cosine Transform (DCT) to the input image.

    Args:
        img (np.ndarray): Input image as a 2D NumPy array.
        block_size (int, optional): Size of the blocks to divide the image into for compression. Default is 8.

    Returns:
        list: List of compressed DCT blocks.

    """
    # Set the compression ratio
    compression_ratio = 0.1  # Keep only the top 10% of coefficients

    # Divide the image into blocks (e.g., 8x8 blocks)
    height, width = img.shape
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size

    # Compress each block
    compressed_blocks = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            dct_block = dct(block)

            # Determine the number of coefficients to keep
            num_coeffs = int(dct_block.size * compression_ratio)
            sorted_coeffs = np.abs(dct_block).ravel().argsort()[::-1][:num_coeffs]

            # Set the remaining coefficients to zero
            mask = np.zeros_like(dct_block)
            mask.ravel()[sorted_coeffs] = 1
            compressed_dct_block = dct_block * mask

            compressed_blocks.append(compressed_dct_block)

    return compressed_blocks


def discrete_cosine_transform_reconstruct(img: np.ndarray, compressed_blocks, block_size=8):
    """
    Reconstructs the compressed image using the compressed DCT blocks.

    Args:
        img (np.ndarray): Compressed image as a 2D NumPy array.
        compressed_blocks (list): List of compressed DCT blocks.
        block_size (int, optional): Size of the blocks used for compression. Default is 8.

    Returns:
        np.ndarray: Reconstructed image as a 2D NumPy array.

    """
    height, width = img.shape
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size

    # Reconstruct the compressed image
    compressed_image = np.zeros_like(img)
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            compressed_dct_block = compressed_blocks[i * num_blocks_w + j]
            block = inv_dct(compressed_dct_block)
            compressed_image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block

    # Convert back to uint8 format (0-255)
    compressed_image = np.uint8(compressed_image)
    return compressed_image


def dct(block):
    """
    Applies the Discrete Cosine Transform (DCT) to a given block.

    Args:
        block (np.ndarray): Input block as a 2D NumPy array.

    Returns:
        np.ndarray: Compressed DCT block.

    """
    N = block.shape[0]
    M = block.shape[1]
    alpha = np.ones_like(block) * np.sqrt(2/N)
    alpha[0, :] = np.sqrt(1/N)

    dct_block = np.zeros_like(block, dtype=float)

    for u in range(N):
        for v in range(M):
            for i in range(N):
                for j in range(M):
                    dct_block[u, v] += block[i, j] * np.cos((2*i + 1) * u * np.pi / (2*N)) * np.cos((2*j + 1) * v * np.pi / (2*M))
            dct_block[u, v] *= alpha[u, v]

    return dct_block


def inv_dct(dct_block):
    """
    Applies the inverse Discrete Cosine Transform (IDCT) to a given DCT block.

    Args:
        dct_block (np.ndarray): Compressed DCT block as a 2D NumPy array.

    Returns:
        np.ndarray: Reconstructed block as a 2D NumPy array.

    """
    N = dct_block.shape[0]
    M = dct_block.shape[1]
    alpha = np.ones_like(dct_block) * np.sqrt(2/N)
    alpha[0, :] = np.sqrt(1/N)

    block = np.zeros_like(dct_block, dtype=float)

    for i in range(N):
        for j in range(M):
            for u in range(N):
                for v in range(M):
                    block[i, j] += alpha[u, v] * dct_block[u, v] * np.cos((2*i + 1) * u * np.pi / (2*N)) * np.cos((2*j + 1) * v * np.pi / (2*M))

    return block


def test():
    # open_image("logitech_mouse_1.png")
    # open_image("logitech_keyboard_1.png")
    print(rgb_to_ycbcr((23, 57, 203)))
    lst = [
        [[23, 79, 32], [24, 83, 35]],
        [[74, 104, 204], [250, 230, 23]]
    ]
    lst = np.array(lst)
    print(colour_space_conversion(lst))
    pass



if __name__ == "__main__":
    # test_algorithm()
    test()