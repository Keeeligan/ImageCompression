from PIL import Image
import colorsys
import numpy as np




def test_algorithm(img):

    img = colour_space_conversion(img)

    img = chrominance_downsample(img)




def jpeg_algorithm():
    pass





def colour_space_conversion(img:np.ndarray):
    """
    This function converts all the RGB values of an image to a different colour space.
    The converted colour space consists of Luminance (Y), Blue Chrominance (Cb) and Red Chrominance (Cr).
    The Luminance represents the brightness, and the Blue- and Red Chrominance make out the colours.

    Note:
    Using this conversion you can use the Luminance as a base layer, now if we downsample the two Chrominance
    layers we're able to keep a good amount of detail/precision when we deleted quite a bit of information.

    :param img: The image that needs converting.
    :return: np.ndarray containing the converted pixel values.
    """

    # Convert each rgb value to the YCbCr colour space.
    for y in range(1, img.shape[0]):
        for x in range(1, img.shape[1]):
            img[y, x] = rgb_to_ycbcr(img[y, x])

    return img


def rgb_to_ycbcr(rgb: tuple):
    """
    Converts the rbg values of a single pixel to the YCbCr colour space.
    Source: https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdprfx/b550d1b5-f7d9-4a0c-9141-b3dca9d7f525

    :param rgb: Tuple with the RBG values of a pixel.
    :return: Tuple with the YCbCr of the pixel.
    """

    # Shift the RGB values down by 128.
    r, g, b = [x-128 for x in rgb]


    # Create the YCbCr values based on the rgb values of the image.
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) / (2 * (1 - 0.114))
    cr = (r - y) / (2 * (1 - 0.299))
    return y, cb, cr


def chrominance_downsample(img: np.ndarray):
    """
    Takes the YCbCr values of an img and creates 2x2 blocks and makes it one pixel based on the average
    of the values in the 2x2 block.

    :param img: np.ndarray with the YCbCr values of an img
    :return: Tuple with 3 ndarrays containing the Luminance layer, and the two down sampled chrominance layers.
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
    for y in range(0, img.shape[0], 2):
        for x in range(0, img.shape[1], 2):
            blue_chrominance[y // 2, x // 2] = np.average([img[y, x, 1], img[y, x + 1, 1], img[y + 1, x, 1], img[y + 1, x + 1, 1]])
            red_chrominance[y // 2, x // 2] = np.average([img[y, x, 2], img[y, x + 1, 2], img[y + 1, x, 2], img[y + 1, x + 1, 2]])

    return luminance, blue_chrominance, red_chrominance


def discrete_cosine_transform(img: np.ndarray, block_size=8):
    # Set the compression ratio
    compression_ratio = 0.1  # Keep only the top 10% of coefficients

    # Divide the image into blocks (e.g., 8x8 blocks)
    height, width = img.size
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
    height, width = img.size
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

    pass



if __name__ == "__main__":
    # test_algorithm()
    test()