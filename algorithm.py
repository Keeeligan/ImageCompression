from PIL import Image
import colorsys
import numpy as np


def test_algorithm(image_name: str):
    img = open_image(image_name)


def jpeg_algorithm():
    pass




def open_image(image_name: str):
    """
    Opens and returns the image
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
    Using this conversion you can use the Luminance as a base layer, now if we downsample  the two Chrominance
    layers we're able to keep a good amount of detail/precision when we deleted quite a bit of information.

    :param img: The image that needs converting.
    :return:
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
    rgb = (204, 104, 45)
    r, g, b = [x-128 for x in rgb]


    # Create the YCbCr values based on the rgb values of the image.
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) / (2 * (1 - 0.114))
    cr = (r - y) / (2 * (1 - 0.299))
    # y, cb, cr = colorsys.rgb_to_ycbcr(r / 255, g / 255, b / 255)
    return y, cb, cr


def chrominance_downsample(img: tuple):
    pass


def test():
    # open_image("logitech_muis_1.png")
    # open_image("logitech_toetsenbord_1.png")

    pass



if __name__ == "__main__":
    # test_algorithm()
    test()