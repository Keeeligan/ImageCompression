from PIL import Image
import colorsys
import numpy as np




def test_algorithm(image_name: str):
    img = open_image(image_name)
    if img is None:
        print("stopping compression")
    img = colour_space_conversion(img)

    img = chrominance_downsample(img)




def jpeg_algorithm():
    pass




def open_image(image_name: str):
    """
    Opens and returns the image.

    :param image_name: String containing the file name of the image.
    :return: None if failed, else the opened image
    """
    print(f"Trying to open the image {image_name}...")
    try:
        # Relative Path
        img = Image.open("images/IN/"+image_name)

        # If the image height and with isn't divisible by 8
        # if not width % 8 == 0 or not height % 8 == 0:
        #     print("Failed: Picture doesn't have the correct values")
        #     return None

        # return the opened image
        return img

    except IOError:
        # Return None if it couldn't open the picture
        print("Failed: Couldn't open the picture")
        return None

    if check_image(img) is None:
        return None

    print(f"Original amount of prixels: {width*height}")
    return img


def check_image(img):
    width, height = img.size
    if not width % 2 == 0 or not height % 2 == 0:
        return None
    return


def colour_space_conversion(img):
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
    width, height = img.size
    converted_image = np.ndarray([])

    # Convert each rgb value to the YCbCr colour space.
    for y in range(1, height):
        for x in range(1, width):
            pix_val = img.getpixel((y, x))
            converted_image[y, x] = rgb_to_ycbcr(pix_val)

    return converted_image


def rgb_to_ycbcr(rgb):
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
    :return: Tuple with 3 ndarrays containing the Luminance layer, and the two downsampled chrominance layers.
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


def test():
    # open_image("logitech_muis_1.png")
    # open_image("logitech_toetsenbord_1.png")

    pass



if __name__ == "__main__":
    # test_algorithm()
    test()