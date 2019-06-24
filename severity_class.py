import os
import glob
import csv
from collections import OrderedDict

import numpy as np
import scipy
from matplotlib import pyplot as plt
import joblib

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.utils.multiclass import unique_labels

from skimage import feature
import cv2

import image_norm as imgnorm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)

IMAGE_DIR = os.path.join(BASE_DIR, 'Shared', 'swet_image_folders')
AUG_IMAGE_DIR = os.path.join(BASE_DIR, 'Shared', 'augmented_images', 'images')
SK_DIR = os.path.join(BASE_DIR, 'Shared', 'skin_masks')
AUG_SK_DIR = os.path.join(BASE_DIR, 'Shared', 'augmented_images', 'skin_masks')

DATASET_CSV = "dataset.csv"

IMG_RES_M = 512
IMG_RES_N = 512


def _writeDataset(image_files, sk_files, data_csv):
    try:
        with open(data_csv, 'r', newline='') as data_file, \
                open('dataset_complete.csv', 'w', newline='') as output_file:

            fieldnames = ['img_file', 'sk_file', 'ery_score']
            output_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            output_writer.writeheader()

            data_reader = csv.DictReader(data_file, delimiter=',')
            data_list = list(data_reader)

            for f, s in zip(list(image_files), list(sk_files)):
                path = os.path.normpath(f[0])
                ref_id = int(path.split(os.sep)[-2][1:])
                week = path.split(os.sep)[-1][2:4]

                if week == '00':
                    vis_id = 1
                elif week == '04':
                    vis_id = 2
                elif week == '12':
                    vis_id = 3
                elif week == '16':
                    vis_id = 4

                count = 0
                for i in range(0, len(data_list)):
                    if (ref_id == int(data_list[i]['ref_id']) and
                            vis_id == int(data_list[i]['vis_id'])):
                        for j in range(0, len(f)):
                            ery_score = data_list[i]['ery_score']
                            output_writer.writerow(
                                {
                                    'img_file': f[j],
                                    'sk_file': s[j],
                                    'ery_score': ery_score
                                }
                            )
                        count = count+1
                    elif (ref_id != int(data_list[i]['ref_id']) and count > 0):
                        break

    except FileNotFoundError:
        error_msg = "Can't find file {0}".format(data_csv)
        print(error_msg)

    except PermissionError:
        error_msg = "Lack of permission to read/write file(s)"
        print(error_msg)


def _skinSegment(image, skin_mask):
    skin_segment = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin_segment


def kmeansQuantise(image, quantisation_levels=64):
    """Re-quantisation of image using K-means algorithm
    """
    K = quantisation_levels
    N_ATTEMPTS = 10
    MAX_ITER = 300
    TOL = 0.0001

    km = KMeans(
        n_clusters=K, n_init=N_ATTEMPTS, max_iter=MAX_ITER, tol=TOL,
        n_jobs=-1
    )

    z = np.float32(image.reshape(-1, 3))

    labels = km.fit_predict(z)
    centres = km.cluster_centers_

    res = centres[labels]/(256/K)
    res = res.astype(np.uint8)
    image_q = res.reshape((IMG_RES_M, IMG_RES_N, 3))

    return image_q


def uniformQuantise(image, quantisation_levels=64):
    num_bins = 256/quantisation_levels
    # image = np.uint8(np.floor((image/(256/quantisation_levels))))
    image = np.uint8(np.digitize(image, np.arange(0, 256, num_bins))) - 1
    return image


def glcm(image, skin_mask, grey_levels=64, window_size=None):
    OFFSET = [1]
    THETA = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    GREY_LEVELS = grey_levels

    if window_size is None:
        P = feature.greycomatrix(
            image, OFFSET, THETA, levels=GREY_LEVELS, symmetric=True,
            normed=True
        )
        contrast = glcmProps(P, prop='contrast')
        ASM = glcmProps(P, prop='ASM')
        correlation = glcmProps(P, prop='correlation')
        glcm_mean = glcmProps(P, prop='mean')

        return contrast, ASM, correlation, glcm_mean
    else:
        k = np.uint8(np.floor(window_size/2))
        skin_mask_r = skin_mask[k:skin_mask.shape[0]-k,
                                k:skin_mask.shape[1]-k].ravel()

        patches = extract_patches_2d(image, (window_size, window_size))

        contrast = np.zeros((len(patches), 4))
        ASM = np.zeros((len(patches), 4))
        correlation = np.zeros((len(patches), 4))
        glcm_mean = np.zeros((len(patches), 4))

        for i, patch in enumerate(patches):
            if skin_mask_r[i] != 0:
                P = feature.greycomatrix(
                    patch, OFFSET, THETA, levels=GREY_LEVELS, symmetric=True,
                    normed=True
                )
            contrast[i, :] = glcmProps(P, prop='contrast')
            ASM[i, :] = glcmProps(P, prop='ASM')
            correlation[i, :] = glcmProps(P, prop='correlation')
            glcm_mean[i, :] = glcmProps(P, prop='mean')

        return contrast, ASM, correlation, glcm_mean


def glcm1(image, skin_mask, window_size):
    """Depreciated, do not use
    """
    OFFSET = [1]
    THETA = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    N_GREY_LEVELS = 64

    k = np.uint8(np.floor(window_size/2))
    skin_mask_r = skin_mask[k:skin_mask.shape[0]-k,
                            k:skin_mask.shape[1]-k].ravel()

    patches = extract_patches_2d(image, (window_size, window_size))

    memmap_folder = './joblib_memmap'

    contrast_filename_memmap = os.path.join(memmap_folder, 'constrast_memmap')
    ASM_filename_memmap = os.path.join(memmap_folder, 'ASM_memmap')
    correlation_filename_memmap = os.path.join(memmap_folder, 'correlation_memmap')
    glcm_mean_filename_memmap = os.path.join(memmap_folder, 'glcm_mean_memmap')

    contrast = np.memmap(contrast_filename_memmap, dtype=np.float32,
                         shape=(len(patches), 4), mode='w+')
    ASM = np.memmap(ASM_filename_memmap, dtype=np.float64,
                         shape=(len(patches), 4), mode='w+')
    correlation = np.memmap(correlation_filename_memmap, dtype=np.float32,
                         shape=(len(patches), 4), mode='w+')
    glcm_mean = np.memmap(glcm_mean_filename_memmap, dtype=np.float32,
                         shape=(len(patches), 4), mode='w+')

    Parallel(n_jobs=1, verbose=1)(
        delayed(glcmCal)(i, patch, window_size, skin_mask_r,
                         OFFSET, THETA, N_GREY_LEVELS,
                         contrast, ASM, correlation, glcm_mean)
        for i, patch in enumerate(patches)
    )

    return contrast, ASM, correlation, glcm_mean


def glcmCal(i, patch, window_size, skin_mask_r,
            OFFSET, THETA, N_GREY_LEVELS,
            contrast, ASM, correlation, glcm_mean):

    if skin_mask_r[i] != 0:
        P = feature.greycomatrix(
            patch, OFFSET, THETA, levels=N_GREY_LEVELS, symmetric=True,
            normed=True
        )
        contrast[i, :] = glcmProps(P, prop='contrast')
        ASM[i, :] = glcmProps(P, prop='ASM')
        correlation[i, :] = glcmProps(P, prop='correlation')
        glcm_mean[i, :] = glcmProps(P, prop='mean')


def glcmProps(P, prop='contrast'):
    """Calculate texture properties of a GLCM.

    This is a modified function from scikit-image:
        feature.greycoprops

    Modifications include: minus checking dimensions,
    the normalisation of GLCM to boost performance,
    addition of calculation of GLCM mean.
    """

    (num_level, num_level2, num_dist, num_angle) = P.shape
    assert num_level == num_level2
    assert num_dist > 0
    assert num_angle > 0

    # create weights for specified property
    I, J = np.ogrid[0:num_level, 0:num_level]
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop in ['ASM', 'energy', 'correlation']:
        pass
    elif prop == 'mean':
        weights, _ = np.mgrid[0:num_level, 0:num_level]
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1. / (1. + (I - J) ** 2)
    else:
        raise ValueError('%s is an invalid property' % (prop))

    # compute property for each GLCM
    if prop == 'energy':
        asm = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]
        results = np.sqrt(asm)
    elif prop == 'ASM':
        results = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]
    elif prop == 'correlation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

        std_i = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_i) ** 2),
                                           axes=(0, 1))[0, 0])
        std_j = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_j) ** 2),
                                           axes=(0, 1))[0, 0])
        cov = np.apply_over_axes(np.sum, (P * (diff_i * diff_j)),
                                 axes=(0, 1))[0, 0]

        # handle the special case of standard deviations near zero
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1

        # handle the standard case
        mask_1 = mask_0 == False
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
    elif prop in ['contrast', 'dissimilarity', 'homogeneity', 'mean']:
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]

    return results


def glcmPadding(x, window_size):
    _REDUCED_RES_M = IMG_RES_M - (window_size-1)
    _REDUCED_RES_N = IMG_RES_N - (window_size-1)
    _PAD_SIZE = np.uint8(np.floor(window_size/2))

    _con = x[0]
    _asm = x[1]
    _cor = x[2]
    _gm = x[3]

    con = np.pad(_con.reshape(_REDUCED_RES_M, _REDUCED_RES_N, -1),
                 [(_PAD_SIZE, _PAD_SIZE), (_PAD_SIZE, _PAD_SIZE), (0, 0)],
                 'edge')
    asm = np.pad(_asm.reshape(_REDUCED_RES_M, _REDUCED_RES_N, -1),
                 [(_PAD_SIZE, _PAD_SIZE), (_PAD_SIZE, _PAD_SIZE), (0, 0)],
                 'edge')
    cor = np.pad(_cor.reshape(_REDUCED_RES_M, _REDUCED_RES_N, -1),
                 [(_PAD_SIZE, _PAD_SIZE), (_PAD_SIZE, _PAD_SIZE), (0, 0)],
                 'edge')
    gm = np.pad(_gm.reshape(_REDUCED_RES_M, _REDUCED_RES_N, -1),
                [(_PAD_SIZE, _PAD_SIZE), (_PAD_SIZE, _PAD_SIZE), (0, 0)],
                'edge')

    res = np.array([con, asm, cor, gm])

    return res


def colourMoments(image, skin_mask, window_size=None):
    if image.ndim > 2:
        n_colour_channels = image.shape[2]
    else:
        n_colour_channels = 1

    if window_size is None:
        # One colour moment per image channel
        c_moment_mean = np.zeros((1, n_colour_channels))
        c_moment_var = np.zeros((1, n_colour_channels))
        c_moment_skew = np.zeros((1, n_colour_channels))
        c_moment_kurtosis = np.zeros((1, n_colour_channels))

        # Compute colour moments over the skin region
        for j in range(0, n_colour_channels):
            # Mean
            c_moment_mean[1, j] = np.mean(image[skin_mask != 0, j])

            # Standard deviation
            c_moment_var[1, j] = np.std(image[skin_mask != 0, j])

            # Skewness
            c_moment_skew[1, j] = \
                scipy.stats.skew(image[skin_mask != 0, j].ravel())

            # Kurosis
            c_moment_kurtosis[1, j] = \
                scipy.stats.kurtosis(image[skin_mask != 0, j].ravel())

        return c_moment_mean, c_moment_var, c_moment_skew, c_moment_kurtosis
    else:
        # One colour moment per pixel per channel
        REDUCED_RES_M = IMG_RES_M - (window_size-1)
        REDUCED_RES_N = IMG_RES_N - (window_size-1)
        REDUCED_PIXEL_COUNT = REDUCED_RES_M*REDUCED_RES_N
        PAD_SIZE = np.uint8(np.floor(window_size/2))

        k = np.uint8(np.floor(window_size/2))
        skin_mask_r = skin_mask[k:skin_mask.shape[0]-k,
                                k:skin_mask.shape[1]-k].ravel()

        patches = extract_patches_2d(image, (window_size, window_size))

        if patches.ndim < 4:
            patches = patches[:, :, :, np.newaxis]

        c_moment_mean = np.zeros((REDUCED_PIXEL_COUNT, n_colour_channels))
        c_moment_var = np.zeros((REDUCED_PIXEL_COUNT, n_colour_channels))
        c_moment_skew = np.zeros((REDUCED_PIXEL_COUNT, n_colour_channels))
        c_moment_kurtosis = np.zeros((REDUCED_PIXEL_COUNT, n_colour_channels))

        # Compute colour moments
        for i in range(0, len(skin_mask_r)):
            if skin_mask_r[i] != 0:
                for j in range(0, n_colour_channels):
                    # Mean
                    c_moment_mean[i, j] = np.mean(patches[i, :, :, j])

                    # Standard deviation
                    c_moment_var[i, j] = np.std(patches[i, :, :, j])

                    # Skewness
                    c_moment_skew[i, j] = \
                        scipy.stats.skew(patches[i, :, :, j].ravel())

                    # Kurosis
                    c_moment_kurtosis[1, j] = \
                        scipy.stats.kurtosis(patches[i, :, :, j].ravel())

        # Padding
        c_moment_mean = c_moment_mean.reshape(
            (REDUCED_RES_M, REDUCED_RES_N, n_colour_channels)
        )
        c_moment_mean = np.pad(
            c_moment_mean,
            [(PAD_SIZE, PAD_SIZE), (PAD_SIZE, PAD_SIZE), (0, 0)],
            'edge'
        )
        c_moment_var = c_moment_var.reshape(
            (REDUCED_RES_M, REDUCED_RES_N, n_colour_channels)
        )
        c_moment_var = np.pad(
            c_moment_var,
            [(PAD_SIZE, PAD_SIZE), (PAD_SIZE, PAD_SIZE), (0, 0)],
            'edge'
        )
        c_moment_skew = c_moment_skew.reshape(
            (REDUCED_RES_M, REDUCED_RES_N, n_colour_channels)
        )
        c_moment_skew = np.pad(
            c_moment_skew,
            [(PAD_SIZE, PAD_SIZE), (PAD_SIZE, PAD_SIZE), (0, 0)],
            'edge'
        )
        c_moment_kurtosis = c_moment_kurtosis.reshape(
            (REDUCED_RES_M, REDUCED_RES_N, n_colour_channels)
        )
        c_moment_kurtosis = np.pad(
            c_moment_kurtosis,
            [(PAD_SIZE, PAD_SIZE), (PAD_SIZE, PAD_SIZE), (0, 0)],
            'edge'
        )

        return c_moment_mean, c_moment_var, c_moment_skew, c_moment_kurtosis


def kmeans(X, skin_mask, k=3):
    """Apply K-means clustering to image.

    Keyword arguments:
    ------------------
    X : pixelwise features
    skin_mask : the skin mask corresponding to the skin segment
    k : number of clusters (default = 3)

    """
    _N_ATTEMPTS = 20
    _MAX_ITER = 300
    _TOL = 0.0001

    km = KMeans(
        n_clusters=k, n_init=_N_ATTEMPTS, max_iter=_MAX_ITER, tol=_TOL,
        n_jobs=-1
    )

    # Standardise data to [-1, 1]
    X_scaler = preprocessing.StandardScaler()
    if X.ndim > 1:
        X = np.float32(X.reshape(-1, X.shape[1]))
        X_scaled = X_scaler.fit_transform(X)

        labels = km.fit_predict(X_scaled)
    else:
        X_scaled = X_scaler.fit_transform(X.reshape(-1, 1))
        labels = km.fit_predict(X_scaled)

    centre = km.cluster_centers_

    largest_centre_idx = np.argmax(centre)
    for i in range(0, len(labels)):
        if labels[i] == largest_centre_idx:
            labels[i] = 1
        else:
            labels[i] = 0

    _lm = np.copy(skin_mask)
    _lm[_lm == 0] = 0
    _lm[_lm == 1] = labels
    lesion_mask = _lm

    return lesion_mask


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def main():
    image_files = glob.glob(IMAGE_DIR + '/*/**.[jJ][pP][gG]')
    image_files.sort()
    aug_image_files = glob.glob(AUG_IMAGE_DIR + '/*/**.[jJ][pP][gG]')
    aug_image_files.sort()

    image_files = zip(image_files, aug_image_files)

    sk_files = glob.glob(SK_DIR + '/*/**.png')
    sk_files.sort()
    aug_sk_files = glob.glob(AUG_SK_DIR + '/*/**.png')
    aug_sk_files.sort()

    sk_files = zip(sk_files, aug_sk_files)

    _writeDataset(image_files, sk_files, DATASET_CSV)

    X = []

    with open(DATASET_CSV, 'r', newline='') as data_file:
        data_reader = csv.DictReader(data_file, delimiter=',')
        data_list = list(data_reader)

    Y = data_list['ery_score']

    for i in range(0, len(data_list)):
        image = cv2.imread(data_list[i]['img_file'], cv2.IMREAD_COLOR)
        skin_mask = cv2.imread(
            data_list[i]['sk_file'],
            cv2.IMREAD_UNCHANGED
        )
        skin_mask = np.uint8(skin_mask/255)
        image_m, image_n, _ = image.shape

        if(image_m != 512 or image_n != 512):
            image = cv2.resize(image, (512, 512))
            skin_mask = cv2.resize(skin_mask, (512, 512))

        # Colour constancy
        image = imgnorm.greyWorld(image, median_blur=3, norm=6)

        image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Colour moments
        c1, c2, c3, c4 = colourMoments(image_LAB[:, :, :], skin_mask)

        # Re-quantisation of A channels
        image_A_q = uniformQuantise(image_LAB[:, :, 1], 64)
        # plt.imshow(image_A_q)
        # plt.colorbar()
        skin_segment = _skinSegment(image_A_q, skin_mask)
        lesion_mask = kmeans(image_LAB[skin_mask != 0, 1], skin_mask)
        n_lesion_pixels = np.count_nonzeros(lesion_mask)
        n_skin_pixels = np.count_nonzeros(skin_segment)
        coverage = n_lesion_pixels/n_skin_pixels

        X_temp = np.concatenate([c1, c2, c3, c4, coverage], axis=None)

        # Extract 3 random patches from the images
        good_patch_count = 0
        count = 0
        patches = extract_patches_2d(
            skin_segment,
            (32, 32), max_patches=100, random_state=1
        )
        for patch in patches:
            if good_patch_count == 4:
                break
            if count < 50:
                if (32*32-np.count_nonzero(patch) < 103):
                    good_patch_count = good_patch_count + 1
                    count = count + 1
                    g1, g2, g3, g4 = glcm(patch, skin_mask, 64)
                    X_temp = np.concatenate(
                        (X_temp, [g1, g2, g3, g4]), axis=None
                    )
            if count > 50:
                if (32*32-np.count_nonzero(patch) < 206):
                    good_patch_count = good_patch_count + 1
                    count = count + 1
                    g1, g2, g3, g4 = glcm(patch, skin_mask, 64)
                    X_temp = np.concatenate(
                        (X_temp, [g1, g2, g3, g4]), axis=None
                    )

        X.append(X_temp)

    joblib.dump(np.array(X), 'X_data.joblib', 3)

    # X = joblib.load('X_data.joblib')
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=1, shuffle=True, stratify=Y
    )
    kNN_clf = KNeighborsClassifier(
        n_neighbors=5, weights='uniform', algorithm='auto',
        leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=10
    )
    kNN_clf.fit(X_train, Y_train)
    y = kNN_clf.predict(X_test)
    bc_ACC = balanced_accuracy_score(Y_test, y)
    print(bc_ACC)

if __name__ == "__main__":
    main()
