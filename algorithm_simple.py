import numpy as np

def test_algorithm(img: np.ndarray) -> np.ndarray:
    """

    :param img: np.ndarray with all the pixel values.
    """


    img = colour_compress(img)
    img = run_length_enc(img)

    return img



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
            img[y, x] = np.array(rgb)
    return img


def run_length_enc(img: np.ndarray) -> list:
    """
    @TODO:
    :param img:
    :return:
    """
    img_rle = []

    for y in range(img.shape[0]):
        row = []
        count = 1
        for x in range(img.shape[1]-1):
            if img[y, x][0] == img[y, x + 1][0] and img[y, x][1] == img[y, x + 1][1] and img[y, x][2] == img[y, x + 1][2]:
                count += 1
            else:
                row.append((img[y, x].tolist(), count))
                count = 1
        row.append((img[y, x].tolist(), count))
        img_rle.append(row)

    # return np.array(img_rle)
    return img_rle



def build_image():
    """
    @TODO
    Builds the image
    """
    pass



if __name__ == "__main__":
    test_algorithm()