from PIL import Image
import numpy as np
import pandas as pd

import json

quantization_table = np.array([
    [4,  3,  4,  4,  4,  6,  11,  15],
    [3,  3,  3,  4,  5,  8,  14,  19],
    [3,  4,  4,  5,  8,  12,  16,  19],
    [4,  5,  6,  7,  12,  14,  18, 20],
    [6,  6,  9,  11,  14, 17, 21,  23],
    [9,  12,  12,  18,  23, 22, 25, 21],
    [11,  13,  15,  17, 21, 23, 25, 21],
    [13,  12,  12,  13, 16, 19, 21,  21]
])


def test_algorithm(img: np.ndarray):
    # height, width = img.shape[0], img.shape[1]
    img.astype(int)
    img = colour_space_conversion(img)

    # Use chrominance downsampling
    lum, blue_chr, red_chr = chrominance_downsample(img)

    # Apply DCT
    lum_dct = discrete_cosine_transform(lum)
    blue_chr_dct = np.round(discrete_cosine_transform(blue_chr)).astype(int)
    red_chr_dct = discrete_cosine_transform(red_chr)

    # Compress the DCT arrays with RLE
    lum_rle = run_length_encoding(lum_dct)
    blue_chr_rle = run_length_encoding(blue_chr_dct)
    red_chr_rle = run_length_encoding(red_chr_dct)

    # print(f"Lum length: {len(lum)}")
    # print(f"Blue_chr length: {len(blue_chr)}")
    # print(f"Red_chr length: {len(red_chr)}")
    #
    # print(f"Lum_dct length: {len(lum_dct)}")
    # print(f"Blue_chr_dct length: {len(blue_chr_dct)}")
    # print(f"Red_chr_dct length: {len(red_chr_dct)}")

    print(f"Lum_rle length: {len(lum_rle)}")
    print(f"Blue_chr_rle length: {len(blue_chr_rle)}")
    print(f"Red_chr_rle length: {len(red_chr_rle)}")


    # with open(f'images/STORE/test1.json', 'w') as f:
    #     json.dump(blue_chr_rle, f)

    return [lum_rle, blue_chr_rle, red_chr_rle]


def build_image(image_name: str, directory="images/STORE/"):

    # Open the data
    image_name = image_name.strip(".png")
    img = np.load(f'{directory}ew_{image_name}.npy')

    # Run length decoding


    # Dequantize and invert DCT


    # Resize the chrominance layers


    # Convert the values back to RGB
    img = colour_space_conversion(img, from_rgb=False)

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


def chrominance_rescale(img:np.ndarray):
    """
    Rescales the chrominance channels of an image to 4 times the size.

    Args:
        img (np.ndarray): An input image represented as a NumPy array in YCbCr color space.
            The image should have three channels: luminance (Y), blue-difference chrominance (Cb),
            and red-difference chrominance (Cr).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the rescaled image channels in the same order:
            luminance (Y), rescaled blue-difference chrominance (Cb_res), and rescaled red-difference chrominance (Cr_res).
            The rescaling is performed by duplicating each pixel value in the Cb and Cr channels.
    """
    lum, cb, cr = img

    cb_res = []
    cr_res = []
    for y in range(len(cb)):
        cb_row = []
        cr_row = []
        for x in range(len(cr)):
            cb_row.append(cb[y][x])
            cb_row.append(cb[y][x])
            cr_row.append(cb[y][x])
            cr_row.append(cb[y][x])
        cb_res.append(cb_row)
        cb_res.append(cb_row)
        cr_res.append(cr_row)
        cr_res.append(cr_row)

    return lum, cb_res, cr_res


def discrete_cosine_transform(img: np.ndarray, block_size=8, quantization_table=None):
    """
    Applies the Discrete Cosine Transform (DCT) to the input image.

    Args:
        img (np.ndarray): Input image as a 2D NumPy array.
        block_size (int, optional): Size of the blocks to divide the image into for compression. Default is 8.
        quantization_table (np.ndarray, optional): Quantization table as a 2D NumPy array. Default is None.

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

            # Apply quantization if a quantization table is provided
            if quantization_table is not None:
                quantized_dct_block = quantize_dct_block(compressed_dct_block, quantization_table)
                compressed_blocks.append(quantized_dct_block)
            else:
                compressed_blocks.append(compressed_dct_block)

    return compressed_blocks


def discrete_cosine_transform_reconstruct(img: np.ndarray, compressed_blocks, block_size=8, quantization_table=None):
    """
    Reconstructs the compressed image using the compressed DCT blocks.

    Args:
        img (np.ndarray): Compressed image as a 2D NumPy array.
        compressed_blocks (list): List of compressed DCT blocks.
        block_size (int, optional): Size of the blocks used for compression. Default is 8.
        quantization_table (np.ndarray, optional): Quantization table as a 2D NumPy array. Default is None.

    Returns:
        np.ndarray: Reconstructed image.

    """
    # Set the compression ratio
    compression_ratio = 0.1  # Keep only the top 10% of coefficients

    # Determine the dimensions of the reconstructed image
    num_blocks_h = img.shape[0] // block_size
    num_blocks_w = img.shape[1] // block_size
    height = num_blocks_h * block_size
    width = num_blocks_w * block_size

    # Reconstruct the image block by block
    reconstructed_img = np.zeros((height, width))
    block_index = 0
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Get the compressed DCT block
            compressed_dct_block = compressed_blocks[block_index]

            # Apply dequantization if a quantization table is provided
            if quantization_table is not None:
                quantized_dct_block = dequantize_dct_block(compressed_dct_block, quantization_table)
                compressed_dct_block = quantized_dct_block

            # Reconstruct the block by applying the inverse DCT
            reconstructed_block = inv_dct(compressed_dct_block)

            # Place the reconstructed block in the image
            start_i = i * block_size
            start_j = j * block_size
            reconstructed_img[start_i:start_i + block_size, start_j:start_j + block_size] = reconstructed_block

            block_index += 1

    return reconstructed_img


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


def quantize_dct_block(dct_block, quantization_table):
    """
    Quantizes the given DCT block using the specified quantization table.

    Args:
        dct_block (np.ndarray): DCT block as a 2D NumPy array.
        quantization_table (np.ndarray): Quantization table as a 2D NumPy array.

    Returns:
        np.ndarray: Quantized DCT block.

    """
    return np.round(dct_block / quantization_table).astype(int)
    # return np.rint(dct_block / quantization_table).astype(int)


def dequantize_dct_block(quantized_dct_block, quantization_table):
    """
    Dequantizes the given quantized DCT block using the specified quantization table.

    Args:
        quantized_dct_block (np.ndarray): Quantized DCT block as a 2D NumPy array.
        quantization_table (np.ndarray): Quantization table as a 2D NumPy array.

    Returns:
        np.ndarray: Dequantized DCT block.

    """
    return quantized_dct_block * quantization_table


def run_length_encoding(q_blocks):
    """
    Applies run-length encoding (RLE) for each quantized block.

    Args:
        q_blocks (np.ndarray): Quantized blocks as a 2D NumPy array.

    Returns:
        list: List of run-length pairs representing the compressed block.

    """
    rle_blocks = []

    for q_block in q_blocks:
        rle_block = []
        for y in range(len(q_block)):
            row = []
            count = 1
            for x in range(len(q_block[y])-1):
                if q_block[y][x] == q_block[y][x+1]:
                    count += 1
                else:
                    row.append((int(q_block[y, x]), count))
                    count = 1

            row.append((int(q_block[y, x]), count))
            rle_block.append(row)
        rle_blocks.append(rle_block)

    return rle_blocks


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