from PIL import Image
import algorithm_simple as alg_s
import algorithm_complex as alg_c
import numpy as np


def main():
    image_name = "logitech_mouse_1.png"
    img = open_image(image_name)
    if img is None:
        print("stopping compression")



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

        # Check if the image has the correct size.
        if check_image(img) is None:
            return None

        # Convert each rgb value to the YCbCr colour space.
        width, height = img.size
        img_tuple = np.ndarray([])

        for y in range(1, height):
            for x in range(1, width):
                img_tuple[y, x] = img.getpixel((y, x))

        # return the opened image
        return img_tuple

    except IOError:
        # Return None if it couldn't open the picture
        print("Failed: Couldn't open the picture")
        return None



    print(f"Original amount of pixels: {width*height}")
    return img

def check_image(img):
    width, height = img.size
    if not width % 2 == 0 or not height % 2 == 0:
        return None
    return



if __name__ == "__main__":
    main()