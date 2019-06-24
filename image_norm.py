import numpy as np
import cv2


def greyWorld(image, median_blur=0, norm=1):
    """
    General grey-world algorithm

    TODO: variance-weighted GW

    Parameters:
        image: ndarray
            BGR/RGB image
        median_blur: int
            odd number
        norm: int

    Return: image
    """

    if (median_blur > 1):
        image = cv2.medianBlur(image, median_blur)

    image = np.float64(image)
    if (norm > 1):
        image_p = np.power(image, norm)
        B = np.mean(image_p[:, :, 0])**(1/norm)
        G = np.mean(image_p[:, :, 1])**(1/norm)
        R = np.mean(image_p[:, :, 2])**(1/norm)
    # Special cases:
    # p = 1 -> grey-world
    elif (norm == 1):
        B = np.mean(image[:, :, 0])
        G = np.mean(image[:, :, 1])
        R = np.mean(image[:, :, 2])
    # p = inf -> white patch (max-RGB)
    else:
        B = np.max(image[:, :, 0])
        G = np.max(image[:, :, 1])
        R = np.max(image[:, :, 2])

    average = np.sqrt((B**2 + G**2 + R**2)/3)
    k_B = average/B
    k_G = average/G
    k_R = average/R

    image[:, :, 0] = k_B * image[:, :, 0]
    image[:, :, 1] = k_G * image[:, :, 1]
    image[:, :, 2] = k_R * image[:, :, 2]

    image = np.uint8(image)

    return image


def main():
    pass


if __name__ == '__main__':
    main()
