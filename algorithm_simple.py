import numpy as np

def test_algorithm(img: np.ndarray):
    """

    :param img: np.ndarray with all the pixel values.
    """
    img = colour_compress(img)



    pass



def colour_compress(img: np.ndarray):
    """
    Compresses shifts all the pixel their RGB values down to the closest multiplication of four.
    This allows fow the Run Length Encoding (RLE) algorithm to work better in trade for the depth of colour.
    :param img:
    :return:
    """
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            rgb = []
            for i in range(len(img[y, x])):
                rgb.append(img[y, x][i] - (img[y, x][i] % 4))
            img[y, x] = tuple(rgb)
    return img





def run_length_enc(img: np.ndarray):
    """
    @TODO:
    :param img:
    :return:
    """
    img_rle = np.array

    for y in range(img.shape[0]):
        row = []
        count = 1
        for x in range(img.shape[1]-1):
            if img[y, x] == img[y, x + 1]:
                count += 1
            else:
                row.append((img[y, x - 1], count))
                count = 1
        img_rle = np.append(img_rle, row)

    return img_rle



def build_image():
    """
    @TODO
    Builds the image
    """
    pass




def save_image(img):

    # Build the image out of the array.

    # Save the image
    pass


if __name__ == "__main__":
    test_algorithm()