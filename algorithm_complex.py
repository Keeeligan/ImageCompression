import numpy as np
import pandas as pd

quantization_table = np.array([
    [4, 3, 4, 4, 4, 6, 11, 15],
    [3, 3, 3, 4, 5, 8, 14, 19],
    [3, 4, 4, 5, 8, 12, 16, 19],
    [4, 5, 6, 7, 12, 14, 18, 20],
    [6, 6, 9, 11, 14, 17, 21, 23],
    [9, 12, 12, 18, 23, 22, 25, 21],
    [11, 13, 15, 17, 21, 23, 25, 21],
    [13, 12, 12, 13, 16, 19, 21, 21]
])


def test_algorithm(img: np.ndarray):
    # height, width = img.shape[0], img.shape[1]
    img.astype(int)

    print("Converting colour space...")
    img = colour_space_conversion(img)

    print("Down sampling the chrominance channels...")
    # Use chrominance downsampling
    lum, blue_chr, red_chr = chrominance_downsample(img)

    print("Starting DCT and quantization...")
    # Apply DCT
    lum_dct = discrete_cosine_transform(lum)
    blue_chr_dct = discrete_cosine_transform(blue_chr)
    red_chr_dct = discrete_cosine_transform(red_chr)

    print("Compressing the data with RLE...")
    # Compress the DCT arrays with RLE
    lum_rle = run_length_encoding(lum_dct)
    blue_chr_rle = run_length_encoding(blue_chr_dct)
    red_chr_rle = run_length_encoding(red_chr_dct)

    return [lum_rle, blue_chr_rle, red_chr_rle]


def build_image(image_name: str, directory="images/STORE/"):
    # Open the data
    rle = pd.read_pickle(f'{directory}{image_name}')

    print("Decoding the RLE...")
    # Run length decoding
    print(f"Lum shape:{len(rle[0])}x{len(rle[0][0])}")
    lum_dct = run_length_decoding(rle[0])
    blue_chr_dct = run_length_decoding(rle[1])
    red_chr_dct = run_length_decoding(rle[2])
    print(f"Lum shape, after RLD:{len(lum_dct)}x{len(lum_dct[0])}")

    print("Dequantizing and inverting the DCT...")
    # Dequantize and invert DCT
    lum = inv_discrete_cosine_transform(lum_dct)
    blue_chr = inv_discrete_cosine_transform(blue_chr_dct)
    red_chr = inv_discrete_cosine_transform(red_chr_dct)

    print("Resizing the chrominance layers to fit the original size...")
    # Resize the chrominance layers
    img = chrominance_rescale(lum, blue_chr, red_chr)

    print("Converting the YCbCr values back to RGB...")
    # Convert the values back to RGB
    img = colour_space_conversion(img, from_rgb=False)

    return img


def colour_space_conversion(img: np.ndarray, from_rgb=True):
    """
    Converts all the RGB values of an image to the YCbCr color space.

    The converted color space consists of Luminance (Y), Blue Chrominance (Cb), and Red Chrominance (Cr).
    The Luminance represents the brightness, and the Blue- and Red Chrominance make up the colors.

    Note:
    Using this conversion, the Luminance can serve as a base layer. By downsampling the two Chrominance layers,
    it is possible to retain a good amount of detail/precision while reducing the amount of information.

    Args:
        img (np.ndarray): The image that needs to be converted.
        from_rgb (bool): Switch the conversion way.

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

    return np.array(res, dtype=np.uint8)


def rgb_to_ycbcr(rgb: tuple):
    """
    Converts the rbg values of a single pixel to the YCbCr colour space.
    Source: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdprfx/b550d1b5-f7d9-4a0c-9141-b3dca9d7f525

    Args:
        rgb (Tuple[int, int, int]): Tuple with the RGB values of a pixel.

    Returns:
        Tuple[float, float, float]: Tuple with the YCbCr values of the pixel.
    """
    r, g, b = rgb

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

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

    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)

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
    for y in range(0, img.shape[0] - 1, 2):
        for x in range(0, img.shape[1] - 1, 2):
            blue_chrominance[y // 2, x // 2] = np.average(
                [img[y, x, 1], img[y, x + 1, 1], img[y + 1, x, 1], img[y + 1, x + 1, 1]])
            red_chrominance[y // 2, x // 2] = np.average(
                [img[y, x, 2], img[y, x + 1, 2], img[y + 1, x, 2], img[y + 1, x + 1, 2]])

    return luminance, blue_chrominance, red_chrominance


def chrominance_rescale(lum, cb, cr):
    """
    Rescales the chrominance channels of an image to 4 times the size.

    Args:
        lum (np.ndarray): Luminance channel (Y) of the input image represented as a NumPy array.
        cb (np.ndarray): Blue-difference chrominance channel (Cb) of the input image represented as a NumPy array.
        cr (np.ndarray): Red-difference chrominance channel (Cr) of the input image represented as a NumPy array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the rescaled image channels in the same order:
            luminance (Y), rescaled blue-difference chrominance (Cb_res), and rescaled red-difference chrominance (Cr_res).
            The rescaling is performed by duplicating each pixel value in the Cb and Cr channels.
    """
    # Get the dimensions of the input channels
    height, width = lum.shape

    # Rescale the chrominance channels to the correct size
    cb_res = np.repeat(np.repeat(cb, 4, axis=1), 4, axis=0)[:height, :width]
    cr_res = np.repeat(np.repeat(cr, 4, axis=1), 4, axis=0)[:height, :width]

    # Create the rescaled image array
    img = np.empty(shape=(height, width, 3), dtype=np.uint8)

    # Fill the channels
    img[:, :, 0] = lum
    img[:, :, 1] = cb_res
    img[:, :, 2] = cr_res

    return img


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

    print(f"{num_blocks_w * num_blocks_h} blocks to compute...")
    # Compress each block
    compressed_blocks = np.empty((num_blocks_h, num_blocks_w, block_size, block_size), dtype=int)

    for y in range(num_blocks_h):
        for x in range(num_blocks_w):
            # Take the 8x8 block and perform DCT on that block
            block = img[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size]
            dct_block = dct(block)

            # Determine the number of coefficients to keep
            num_coeffs = int(dct_block.size * compression_ratio)
            sorted_coeffs = np.abs(dct_block).ravel().argsort()[::-1][:num_coeffs]

            # Set the remaining coefficients to zero
            mask = np.zeros_like(dct_block)
            mask.ravel()[sorted_coeffs] = 1
            compressed_dct_block = dct_block * mask

            # Apply quantization
            quantized_dct_block = quantize_dct_block(compressed_dct_block)
            compressed_blocks[y, x] = quantized_dct_block

    return compressed_blocks


def inv_discrete_cosine_transform(compressed_blocks, block_size=8):
    """
    Applies the inverse Discrete Cosine Transform (IDCT) to the compressed blocks for image decompression.

    Args:
        compressed_blocks (np.ndarray): Compressed DCT blocks as a 4D NumPy array.
        block_size (int, optional): Size of the blocks used for compression. Default is 8.

    Returns:
        np.ndarray: Reconstructed decompressed image.

    """
    # Compute the dimensions of the decompressed image
    num_blocks_h, num_blocks_w = compressed_blocks.shape[:2]
    height = num_blocks_h * block_size
    width = num_blocks_w * block_size

    # Initialize the decompressed image
    decompressed_image = np.zeros((height, width), dtype=float)

    print(f"{num_blocks_w * num_blocks_h} blocks to reconstruct...")
    # Iterate over each block and perform IDCT and dequantization
    for y in range(num_blocks_h):
        for x in range(num_blocks_w):
            quantized_dct_block = compressed_blocks[y, x]
            # Apply dequantization
            decompressed_dct_block = dequantize_dct_block(quantized_dct_block)

            # Perform IDCT on the block
            idct_block = inv_dct(decompressed_dct_block)

            # Place the IDCT block in the decompressed image
            decompressed_image[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = idct_block

    return decompressed_image


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

    # Create the alpha matrix
    alpha_p = np.ones_like(block) * np.sqrt(2 / M)
    alpha_p[0, :] = 1 / np.sqrt(M)
    alpha_q = np.ones_like(block) * np.sqrt(2 / M)
    alpha_q[:, 0] = 1 / np.sqrt(N)

    dct_block = np.zeros_like(block, dtype=float)

    # Precompute constants
    constant_N = np.pi / (2 * N)
    constant_M = np.pi / (2 * M)

    for n in range(N):
        for m in range(M):
            constant_n = (2 * n + 1) * constant_N
            constant_m = (2 * m + 1) * constant_M

            for q in range(N):
                for p in range(M):
                    # Apply precomputed constants
                    cos_q = np.cos(constant_n * q)
                    cos_p = np.cos(constant_m * p)

                    dct_block[q, p] += block[n, m] * cos_p * cos_q

    # Multiply with alpha for normalisation
    dct_block *= alpha_p * alpha_q
    dct_block = np.round(dct_block)
    return dct_block


def inv_dct(dct_block):
    """
    Applies the inverse Discrete Cosine Transform (IDCT or DCT-III) to a given DCT block.

    Args:
        dct_block (np.ndarray): DCT block as a 2D NumPy array.

    Returns:
        np.ndarray: Reconstructed block as a 2D NumPy array.

    """
    N = dct_block.shape[0]
    M = dct_block.shape[1]

    block = np.zeros_like(dct_block, dtype=float)

    for q in range(N):
        for p in range(M):
            for n in range(N):
                for m in range(M):
                    cos_p = np.cos((np.pi * (2 * m + 1) * p) / (2 * M))
                    cos_q = np.cos((np.pi * (2 * n + 1) * q) / (2 * N))

                    # Calculate the alpha for normalisation
                    if p == 0:
                        alpha_p = 1 / np.sqrt(M)
                    else:
                        alpha_p = np.sqrt(2 / M)

                    if q == 0:
                        alpha_q = 1 / np.sqrt(N)
                    else:
                        alpha_q = np.sqrt(2 / N)

                    # Sum to the block
                    block[n, m] += alpha_p * alpha_q * dct_block[q, p] * cos_p * cos_q

    block = np.round(block)
    return block.astype(np.uint8)


def quantize_dct_block(dct_block):
    """
    Quantizes the given DCT block using the specified quantization table.

    Args:
        dct_block (np.ndarray): DCT block as a 2D NumPy array.

    Returns:
        np.ndarray: Quantized DCT block.

    """
    return np.round(dct_block / quantization_table).astype(int)


def dequantize_dct_block(quantized_dct_block):
    """
    Dequantizes the given quantized DCT block using the specified quantization table.

    Args:
        quantized_dct_block (np.ndarray): Quantized DCT block as a 2D NumPy array.

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
    # For each row
    for q_block_row in q_blocks:
        rle_row = []

        # For each block within that row
        for block in q_block_row:
            # Flatten the block and use rle
            rle_block = []
            block = block.flatten()
            count = 1

            # Up counter if the next element is the same, else append the data + counter and reset the counter
            for i in range(len(block) - 1):
                if block[i] == block[i + 1]:
                    count += 1
                else:
                    rle_block.append((int(block[i]), count))
                    count = 1
            rle_block.append((int(block[i]), count))

            # Append the rle_block to the row
            rle_row.append(rle_block)
        # Append the row to the base
        rle_blocks.append(rle_row)

    return rle_blocks


def run_length_decoding(rle_blocks, block_size=8):
    """
    Decodes the run-length encoded blocks back to the original quantized blocks.

    Args:
        rle_blocks (list): List of run-length pairs representing the compressed block.
        block_size (tuple): Shape of the original block. Defaults to 8.

    Returns:
        np.ndarray: Decoded quantized blocks as a 2D NumPy array.

    """
    height = len(rle_blocks)
    width = len(rle_blocks[0])
    decoded_blocks = np.empty(shape=(height, width, block_size, block_size), dtype=int)

    # Reconstruct the list
    for y in range(len(rle_blocks)):
        for x in range(len(rle_blocks[y])):
            decoded_block = []
            # Paste the values the amount of times the count is on
            for value, count in rle_blocks[y][x]:
                decoded_block.extend([value] * count)

            # Reshape the 64 element list to a 8x8 array
            decoded_blocks[y, x] = np.reshape(decoded_block, (block_size, block_size))

    return decoded_blocks


def test():
    # Testfunction, irrelevant for when running the GUI
    n_row = [42] * 8
    n = np.array([n_row] * 8, dtype=float)
    n[:, 3] = 255
    n[:, 4] = 0.

    print("block before DCT:\n", n)
    block = dct(n)
    print("Block after dct:\n", block)
    print("inverted block:\n", inv_dct(block))


if __name__ == "__main__":
    test()
