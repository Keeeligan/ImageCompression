from PIL import Image
import algorithm_simple as alg_s
import algorithm_complex as alg_c
import numpy as np
import pickle


def compress_alg_s(image_name:str = "logitech_mouse_1_ds.png" ):
    """
    Applies the simple algorithm's compression to an input image.

    Args:
        image_name (str, optional): The filename of the input image.
            Default is set to "logitech_mouse_1_ds.png".
    """
    img = open_image(image_name)
    save_compressed_data(img, "og_"+image_name)
    if img is None:
        print(img)
        print("stopping compression")
        return
    img = alg_s.run_simple_algorithm(img)
    image_name = image_name.strip(".png")
    save_compressed_data(img, f"new_{image_name}_simp")


def build_alg_s(image_name:str = "logitech_mouse_1_ds.png" ):
    """
    Builds an image using the simple algorithm .

    Args:
        image_name (str, optional): The filename of the input image.
            Default is set to "logitech_mouse_1_ds.png".
    """
    img = alg_s.build_image(image_name)

    print(f"img shape: {len(img)}x{len(img[0])}")
    image_name = image_name[:len(image_name)-len(".pickle")]
    save_image(img, f"{image_name}")


def compress_alg_c(image_name:str = "logitech_mouse_1_ds.png"):
    """
    Applies the complex algorithm's compression to an input image.

    Args:
        image_name (str, optional): The filename of the input image.
            Default is set to "logitech_mouse_1_ds.png".
    """
    img = open_image(image_name)
    save_compressed_data(img, "og_" + image_name)
    img = img.astype(int)
    if img is None:
        print(img)
        print("stopping compression")
        return
    img = alg_c.test_algorithm(img)
    image_name = image_name.strip(".png")
    save_compressed_data(img, f"new_{image_name}_comp")


def build_alg_c(image_name:str = "logitech_mouse_1_ds.png"):
    """
    Builds an image using the complex algorithm .

    Args:
        image_name (str, optional): The filename of the input image.
            Default is set to "logitech_mouse_1_ds.png".
    """
    img = alg_c.build_image(image_name)

    print(f"img shape: {len(img)}x{len(img[0])}")

    image_name = image_name[:len(image_name)-len(".pickle")]
    save_image(img, f"{image_name}")


def open_image(image_name: str):
    """
    Opens and returns the image.

    Args:
        image_name (str): String containing the file name of the image.

    Returns:
        Optional[np.ndarray]: None if the image failed to open, else the opened image as a NumPy array.

    """
    print(f"Trying to open the image {image_name}...")
    try:
        # Relative Path
        img = Image.open("images/IN/"+image_name)
        # Convert image to RGB color mode
        img = img.convert("RGB")

        width, height = img.size
        img_array = np.empty((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                img_array[y, x] = list(img.getpixel((x, y)))

        # return the opened image
        print("Returning opened image...")
        return img_array

    except IOError:
        # Return None if it couldn't open the picture
        print("Failed: Couldn't open the picture")
        return None

    print(f"Original amount of pixels: {width*height}")
    return img


def save_compressed_data(img, name: str, directory="images/STORE/"):
    """
    Saves compressed data (image) to a file.

    Args:
        img: The compressed data (image) to be saved.
        name (str): The name of the file to be saved (without extension).
        directory (str, optional): The directory path where the file will be saved.
            Default is set to "images/STORE/".
    """
    if type(img) == list:
        with open(f'{directory}{name}.pickle', 'wb') as file:
            pickle.dump(img, file)
        return

    np.save(f"{directory}{name}.npy", img)
    return


def save_image(img: np.ndarray, image_name, mode="RGB", directory="images/OUT/"):
    """
    Saves a NumPy array as an image file.

    Args:
        img (numpy.ndarray): The image to be saved.
        image_name (str): The name of the file to be saved (without extension).
        mode (str, optional): The mode of the image. Default is set to "RGB".
        directory (str, optional): The directory path where the file will be saved.
            Default is set to "images/OUT/".
    """
    new_img = Image.fromarray(img, mode=mode)
    new_img.save(f"{directory}{image_name}.png")


if __name__ == "__main__":
    compress_alg_c("white_test.png")
    build_alg_c("new_white_test_comp.pickle")
    pass
