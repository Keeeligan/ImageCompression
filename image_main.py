from PIL import Image
import algorithm_simple as alg_s
import algorithm_complex as alg_c
import numpy as np
import json
import pickle

from pprint import pprint

def run_simple_algorithm(image_name:str = "logitech_mouse_1_ds.png" ):
    img = open_image(image_name)
    save_compressed_data(img, "og_"+image_name)
    if img is None:
        print(img)
        print("stopping compression")
        return
    img = alg_s.test_algorithm(img)
    save_compressed_data(img, f"new_{image_name}")
    img = alg_s.build_image(image_name)

    print(type(img))
    print(img)

    with open(f'images/STORE/test_simpleout.json', 'w') as f:
        json.dump(img.tolist(), f)

    print(f"img shape: {len(img)}x{len(img[0])}")

    save_image(img, image_name)


def run_complex_algorithm(image_name:str = "logitech_mouse_1_ds.png"):
    img = open_image(image_name)
    save_compressed_data(img, "og_" + image_name)
    img = img.astype(int)
    if img is None:
        print(img)
        print("stopping compression")
        return
    img = alg_c.test_algorithm(img)

    save_compressed_data(img, f"new_{image_name}")

    img = alg_c.build_image(image_name)

    print(type(img))
    print(img)
    print(f"img shape: {len(img)}x{len(img[0])}")

    save_image(img, image_name)


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
    name = name.strip(".png")
    if type(img) == list:
        # Save with json
        with open(f'{directory}{name}.json', 'w') as f:
            json.dump(img, f)

        # Save with Pickle
        with open(f'{directory}{name}.pickle', 'wb') as file:
            pickle.dump(img, file)

        return

    np.save(f"{directory}{name}.npy", img)
    return


def save_image(img: np.ndarray, image_name, mode="RGB", directory="images/OUT/"):
    new_img = Image.fromarray(img, mode=mode)
    new_img.save(f"{directory}{image_name}")


if __name__ == "__main__":
    # run_simple_algorithm()
    run_complex_algorithm()
    # save_image(np.array(
    #     [[[255, 255, 255], [255, 255, 255]],
    #      [[255, 255, 255], [255, 255, 255]]], dtype=np.uint8), "test_small.png"
    # )
