import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import feature
import cv2

SKIN_MASK_RES_N = 512
SKIN_MASK_RES_M = 384

def skinSegment(image, skin_mask):
    skin_segment = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin_segment


def kmeans(image):
    k = 4
    N_ATTEMPTS = 20
    MAX_ITER = 300
    TOL = 0.0001

    km = cluster.KMeans(
        n_clusters=k, n_init=N_ATTEMPTS, max_iter=MAX_ITER, tol=TOL
    )

    image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_chr = image_LAB[:,:,1:2]
    #image = image[image != [0,0,0]]
    z = np.float32(image_chr.reshape(-1, 2))

    label = km.fit_predict(z)
    centre = km.cluster_centers_
    res = centre[label]
    lesion_mask = res.reshape((384, 512))

    return lesion_mask


def glcm(image, window_size):
    OFFSET = [1]
    THETA = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    N_GREY_LEVELS = 256
    
    patches = extract_patches_2d(image, (window_size, window_size))

    contrast = []
    ASM = []
    correlation = []
    
    for patch in patches:
        P = feature.greycomatrix(
            patch, OFFSET, THETA, levels=N_GREY_LEVELS, symmetric=True,
            normed=True
        )
        feature.greycoprops(P, prop='contrast')
        feature.greycoprops(P, prop='ASM')
        feature.greycoprops(P, prop='correlation')

    return contrast, ASM, correlation, P

def main():
    image = cv2.imread('images/test_02.jpg', cv2.IMREAD_COLOR)
    skin_mask = np.uint8(cv2.imread('images/skin_masks/02.jpg',
                         cv2.IMREAD_UNCHANGED)/255)

    image = cv2.resize(image, (512, 384))
    skin_mask = cv2.resize(skin_mask, (512, 384))

    image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    skin_segment = skinSegment(image, skin_mask)
    contrast, ASM, correlation, P = glcm(image_LAB[:,:,1], 7)

    skin_segment1 = cv2.cvtColor(skin_segment, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(skin_segment1)

    lesion_mask = kmeans(skin_segment)
    plt.figure()
    plt.imshow(lesion_mask)
    plt.show()


if __name__ == "__main__":
    main()
