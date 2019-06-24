import numpy as np
import cv2

from sklearn import svm
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Constants
SKIN_MASK_RES_M = 384
SKIN_MASK_RES_N = 512


def resample(image):
    """
    Apply Gaussian filter with kernel size (9,9) and standard deviation = 2.5
    and resample the image to a lower resolution (512, 384) for faster
    pre-processing.

    Parameters: image

    Return: image_ds -> downsampled image
    """

    # Apply Gaussian filter to reduce image noise
    image_blurred = cv2.GaussianBlur(image,  (9, 9), 2.5)
    # Resize to 512x384
    image_ds = cv2.resize(image_blurred, (SKIN_MASK_RES_N, SKIN_MASK_RES_M))

    return image_ds


def skinSegKovac(image):
    image_ds = resample(image)
    # allocate memory for skin mask
    skin_mask = np.empty((SKIN_MASK_RES_M, SKIN_MASK_RES_N), np.uint8)

    for m in range(0, SKIN_MASK_RES_M):
        for n in range(0, SKIN_MASK_RES_N):
            R = image_ds[m, n, 2]
            G = image_ds[m, n, 1]
            B = image_ds[m, n, 0]

            if (R > 95 and R > G > 40 and 20 < B < R and R - G > 15 and
                    R - min(B, G) > 15):
                skin_mask[m, n] = 1
            else:
                skin_mask[m, n] = 0

    return skin_mask


def skinSegCN(image):
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image_ds = resample(image_YCrCb)

    # allocate memory for skin mask
    skin_mask = np.empty((SKIN_MASK_RES_M, SKIN_MASK_RES_N), np.uint8)

    for m in range(0, SKIN_MASK_RES_M):
        for n in range(0, SKIN_MASK_RES_N):
            if (133 <= image_ds[m, n, 1] <= 173 and
                    77 <= image_ds[m, n, 2] <= 127):
                skin_mask[m, n] = 1
            else:
                skin_mask[m, n] = 0

    return skin_mask


def skinSegkmeans(image):
    k = 2
    N_ATTEMPTS = 20
    MAX_ITER = 50
    EXP_ACC = 0.01

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, MAX_ITER,
                EXP_ACC)

    image_ds = resample(image)

    image_LAB = cv2.cvtColor(image_ds, cv2.COLOR_BGR2LAB)
    image_chr = image_LAB[:, :, 1:2]
    z = np.float32(image_chr.reshape(-1, 2))

    _, label, centre = cv2.kmeans(z, k, None, criteria, N_ATTEMPTS,
                                  cv2.KMEANS_PP_CENTERS)

    centre = np.uint8(centre)
    if (centre[0])[0] > (centre[1])[0] and (centre[0])[1] > (centre[1])[1]:
        centre[0] = [1, 1]
        centre[1] = [0, 0]
    else:
        centre[0] = [0, 0]
        centre[1] = [1, 1]
    res = centre[label.ravel()]
    skin_mask = res.reshape((SKIN_MASK_RES_M, SKIN_MASK_RES_N))
    skin_mask = np.squeeze(skin_mask)

    return skin_mask


def skinMaskColour(image):
    skin_mask1 = cv2.medianBlur(skinSegKovac(image), 5)
    skin_mask2 = cv2.medianBlur(skinSegCN(image), 5)
    skin_mask3 = cv2.medianBlur(skinSegkmeans(image), 5)

    # Contour-based
    cnt1, _ = cv2.findContours(skin_mask1, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnt2, _ = cv2.findContours(skin_mask2, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnt3, _ = cv2.findContours(skin_mask3, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(skin_mask1, cnt1, -1, 1, -1)
    cv2.drawContours(skin_mask2, cnt2, -1, 1, -1)
    cv2.drawContours(skin_mask3, cnt3, -1, 1, -1)

    return skin_mask1, skin_mask2, skin_mask3


def gaborKernels():
    def gabor_fn(theta, Lambda, psi, gamma=0.5, bandwidth=2):
        slr = (1/np.pi) * np.sqrt(np.log(2)/2) * \
              (2**bandwidth+1)/(2**bandwidth-1)

        sigma = slr * Lambda
        sigma_x = sigma
        sigma_y = float(sigma)/gamma

        # Bounding box
        nstds = 4  # Number of standard deviation sigma
        xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y *
                                                             np.sin(theta)))
        xmax = np.ceil(max(1, xmax))
        ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y *
                                                             np.cos(theta)))
        ymax = np.ceil(max(1, ymax))
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1),
                             np.arange(xmin, xmax + 1))

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = np.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 /
                    sigma_y ** 2)) * np.exp(1j*(2*np.pi * x_theta/Lambda + psi))

        return gb, sigma

    gabor_kernels = []
    kernel_params = []
    psi = 0.5*np.pi

    for lambd in [1, 2, 4, 8, 16]:
        for theta in range(8):
            theta = theta/8 * np.pi  # eight orientations
            for b in [1, 1.5, 2]:
                for gamma in [0.1, 0.5, 0.9]:
                    kernel, sigma = gabor_fn(theta, lambd, psi=psi,
                                             gamma=gamma, bandwidth=b)
                    gabor_kernels.append(kernel)
                    kernel_params.append(sigma)

    return gabor_kernels, kernel_params


def loadModel(model_name):
    # print('\nLoading model %s...' % (model_name))
    model = joblib.load(model_name)
    # print('Finished loading.\n')
    return model


def skinMaskGabor(image, gabor_kernels, kernel_params):
    SKIN_MASK_RES_R_M = 192
    SKIN_MASK_RES_R_N = 256
    NUM_PIXELS_R = SKIN_MASK_RES_R_M*SKIN_MASK_RES_R_N

    image = cv2.resize(image, (SKIN_MASK_RES_R_N, SKIN_MASK_RES_R_M))
    image_r = image[:, :, 2]
    image_g = image[:, :, 1]
    image_b = image[:, :, 0]

    mag_list = []
    # compute Gabor images by convolving image with Gabor kernels
    for i, kernel in enumerate(gabor_kernels):
        sig = kernel_params[i]
        k_size = 4*int(sig)+1

        img_r_real = cv2.filter2D(image_r, -1, np.real(kernel))
        img_r_imag = cv2.filter2D(image_r, -1, np.imag(kernel))
        mag_r = np.sqrt(img_r_real**2 + img_r_imag**2)

        img_g_real = cv2.filter2D(image_g, -1, np.real(kernel))
        img_g_imag = cv2.filter2D(image_g, -1, np.imag(kernel))
        mag_g = np.sqrt(img_g_real**2 + img_g_imag**2)

        img_b_real = cv2.filter2D(image_b, -1, np.real(kernel))
        img_b_imag = cv2.filter2D(image_b, -1, np.imag(kernel))
        mag_b = np.sqrt(img_b_real**2 + img_b_imag**2)

        mag_r = cv2.GaussianBlur(mag_r.astype(np.float32), (k_size, k_size),
                                 sigmaX=sig)
        mag_g = cv2.GaussianBlur(mag_g.astype(np.float32), (k_size, k_size),
                                 sigmaX=sig)
        mag_b = cv2.GaussianBlur(mag_b.astype(np.float32), (k_size, k_size),
                                 sigmaX=sig)

        mag = np.dstack((mag_r, mag_g, mag_b))
        mag_list.append(mag)

    mag_list = np.array(mag_list)
    mag_list = np.swapaxes(mag_list, 0, 2)
    mag_list = np.swapaxes(mag_list, 0, 1)

    X = np.array(mag_list)
    X = X.reshape(NUM_PIXELS_R, 1080)
    X_scaled = StandardScaler().fit_transform(X)

    clf = loadModel('Models/linear_model.joblib')

    y_pred = clf.predict(X_scaled[0:NUM_PIXELS_R])
    y_pred = y_pred.reshape(SKIN_MASK_RES_R_M, SKIN_MASK_RES_R_N)

    return y_pred


def combineSkinMasks(skin_mask1, skin_mask2, skin_mask3, skin_mask_gabor):
    skin_mask_gabor = cv2.resize(skin_mask_gabor, (SKIN_MASK_RES_N,
                                                   SKIN_MASK_RES_M))
    X = np.dstack((skin_mask1, skin_mask2, skin_mask3, skin_mask_gabor))
    model = loadModel('Models/log_reg_model.joblib')

    y_pred = model.predict(X.reshape(-1, 4))
    y_pred = y_pred.reshape(SKIN_MASK_RES_M, SKIN_MASK_RES_N)

    return y_pred


def finalPostProcess(input_skin_mask):
    AS_SKIN_THRES = 0.95

    skin_mask = cv2.medianBlur(np.array(input_skin_mask, dtype=np.uint8), 7)
    new_skin_mask = np.zeros_like(skin_mask)

    # Find 4-way connected components
    labels, stats = cv2.connectedComponentsWithStats(skin_mask, 4)[1:3]
    # Igonre stats[0, *] becuase 0 is the label of background
    stats_argsorted = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    largest_label = 1 + stats_argsorted[0]
    new_skin_mask[labels == largest_label] = 1

    # Check if second largest connected component occupies similar number
    # of pixels compared to the largest connected component.
    # If true -> it is considered as skin
    if ((len(stats_argsorted) > 1) and
        (stats[stats_argsorted[1]+1, cv2.CC_STAT_AREA] >
            AS_SKIN_THRES*stats[stats_argsorted[0]+1, cv2.CC_STAT_AREA])):
        second_largest_label = 1 + stats_argsorted[1]
        new_skin_mask[labels == second_largest_label] = 1

    cnt, _ = cv2.findContours(new_skin_mask, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(new_skin_mask, cnt, -1, 1, -1)

    return new_skin_mask


def main():
    # Migrated to build_dataset.py
    pass


if __name__ == "__main__":
    main()
