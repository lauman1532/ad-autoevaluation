import random
import math

import numpy as np
import skimage
import cv2


def _zoomCrop(image, zoomed_image):
    m, n, _ = image.shape
    m_zoomed, n_zoomed, _ = zoomed_image.shape

    cropped_image = zoomed_image[
        math.floor((m_zoomed-m)/2):math.floor((m_zoomed+m)/2),
        math.floor((n_zoomed-n)/2):math.floor((n_zoomed+n)/2),
        :
    ]
    return cropped_image


def _rotateCrop(rotated_image, angle):
    m, n, _ = rotated_image.shape
    n_new, m_new = _largestRotatedRect(n, m, angle*math.pi/180)

    cropped_image = rotated_image[
        math.floor((m-m_new)/2):math.floor((m_new+m)/2),
        math.floor((n-n_new)/2):math.floor((n_new+n)/2),
        :
    ]
    return cropped_image


def _largestRotatedRect(w, h, angle):
    """
    Given a rectangle of size (w x h) that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def randomRotate(image, max_angle=20, crop=False, proba=1.0, skin_mask=None):
    assert max_angle > 0, "max_angle has to be positive."
    if(random.random() < proba):
        angle = random.randint(-max_angle, max_angle)

        if skin_mask is None:
            rotated_image = skimage.transform.rotate(
                image, angle, resize=False, preserve_range=True
            )

            if crop:
                rotated_image = _rotateCrop(np.uint8(rotated_image), angle)

                rotated_image = cv2.resize(
                    rotated_image, (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )

            rotated_image = np.uint8(rotated_image)
            return rotated_image
        else:
            rotated_image = skimage.transform.rotate(
                image, angle, resize=False, preserve_range=True
            )
            r_skin_mask = skimage.transform.rotate(
                skin_mask, angle, resize=False, preserve_range=True
            )

            if crop:
                rotated_image = _rotateCrop(np.uint8(rotated_image), angle)
                r_skin_mask = _rotateCrop(np.uint8(r_skin_mask), angle)
                rotated_image = cv2.resize(
                    rotated_image, (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )
                r_skin_mask = cv2.resize(
                    r_skin_mask, (skin_mask.shape[1], skin_mask.shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )

            rotated_image = np.uint8(rotated_image)
            r_skin_mask = np.uint8(r_skin_mask)
            return rotated_image, r_skin_mask
    else:
        if skin_mask is None:
            return image
        else:
            return image, skin_mask


def randomFlip(image, direction='both', proba=1.0, skin_mask=None):
    if(random.random() < proba):
        if skin_mask is None:
            if direction == 'top_down':
                augmented_image = cv2.flip(image, 0)
            elif direction == 'left_right':
                augmented_image = cv2.flip(image, +1)
            elif direction == 'both':
                augmented_image = cv2.flip(image, -1)
            return augmented_image
        else:
            if direction == 'top_down':
                augmented_image = cv2.flip(image, 0)
                skin_mask = cv2.flip(skin_mask, 0)
            elif direction == 'left_right':
                augmented_image = cv2.flip(image, +1)
                skin_mask = cv2.flip(skin_mask, +1)
            elif direction == 'both':
                augmented_image = cv2.flip(image, -1)
                skin_mask = cv2.flip(skin_mask, -1)
            return augmented_image, skin_mask
    else:
        if skin_mask is None:
            return image
        else:
            return image, skin_mask


def randomZoom(image, max_scale=1.1, proba=1.0, skin_mask=None):
    assert max_scale > 1, "max_scale has to be larger than 1."
    if(random.random() < proba):
        scale = random.uniform(1, max_scale)
        augmented_image = cv2.resize(image, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
        augmented_image = _zoomCrop(image, augmented_image)
        return augmented_image
    else:
        return image


def randomBrightnessContrast(image, a=[1.0, 1.0], b=[0.0, 0.0], proba=1.0):
    """
    Adjust the brightness and contrast of an image randomly, constrained by a
    and b.

    Parameters:
        image: ndarray
            BGR/RGB image
        a: [a_min, a_max], float
            Adjust constrast, a = [0, inf]
        b: [b_min, b_max], float
            Adjust brightness, b = [-inf, inf]

    Return: image
    """
    if(random.random() < proba):
        if a != [1, 1]:
            ar = random.uniform(a[0], a[1])
        if b != [0, 0]:
            br = random.uniform(b[0], b[1])
        augmented_image = np.clip(ar * image + br * np.mean(image), 0, 255)
        augmented_image = np.uint8(augmented_image)
        return augmented_image
    else:
        return image


'''
def random_gamma_adjust(image, gamma=1.0, proba=1.0):
    assert gamma < 0, "gamma has to be larger than 0."
    if(random.random() < proba):
'''
