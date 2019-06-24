import os
import glob
import csv

import cv2

import skin_seg as skinseg
import image_augmentation as imgaug


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
IMAGE_DIR = os.path.join(BASE_DIR, 'Shared', 'swet_image_folders')
AUG_IMAGE_DIR = os.path.join(BASE_DIR, 'Shared', 'augmented_images')

DATA_CSV = "label data.csv"
OUTPUT_CSV = "dataset.csv"

IMG_RES_M = 512
IMG_RES_N = 512


def rebuildCSV(data_csv, output_csv):
    ref_id_set = set()
    num_unique_id = 0

    try:
        with open(data_csv, 'r', newline='') as data_file, \
             open(output_csv, 'w', newline='') as output_file:
            data_reader = csv.DictReader(data_file, delimiter=',')

            fieldnames = ['ref_id', 'vis_id', 'ery_score']
            output_writer = csv.DictWriter(output_file, fieldnames=fieldnames)

            output_writer.writeheader()
            for row in data_reader:
                if row['photo'] == 'yes':
                    ref_id = row['refno']  # reference number
                    vis_id = row['visno']  # photo id
                    # representative site's erythema severity
                    ery_score = row['rs_ery']

                    output_writer.writerow(
                        {
                            'ref_id': ref_id,
                            'vis_id': vis_id,
                            'ery_score': ery_score
                        }
                    )

                    ref_id_set.add(ref_id)

            num_unique_id = len(ref_id_set)
            print("Number of unique ID: %d" % (num_unique_id))

    except FileNotFoundError:
        error_msg = "Can't find file {0}".format(data_csv)
        print(error_msg)

    except PermissionError:
        error_msg = "Lack of permission to read/write file(s)"
        print(error_msg)


def augmentImage(image, skin_mask):
    image = cv2.resize(image, (IMG_RES_N, IMG_RES_M))
    skin_mask = cv2.resize(skin_mask, (IMG_RES_N, IMG_RES_M))

    image, skin_mask = imgaug.randomFlip(
        image, 'left_right', proba=0.5, skin_mask=skin_mask
    )
    image, skin_mask = imgaug.randomFlip(
        image, 'top_down', proba=0.5, skin_mask=skin_mask
    )
    image, skin_mask = imgaug.randomRotate(
        image, max_angle=30, proba=1.0, skin_mask=skin_mask
    )
    image = imgaug.randomBrightnessContrast(
        image, [0.8, 1.2], [-0.1, 0.1]
    )
    return image, skin_mask


def main():
    # rebuildCSV(DATA_CSV, OUTPUT_CSV)

    # print(PROJECT_ROOT)
    # print(os.path.dirname(BASE_DIR))
    # print(IMAGE_DIR)

    image_files = glob.glob(IMAGE_DIR + '/*/**.[jJ][pP][gG]')
    image_files.sort()
    num_images = len(image_files)
    print("Number of unique images: %d" % (num_images))

    gabor_kernels, kernel_params = skinseg.gaborKernels()

    image_counter = 0
    print("\n==== Skin segmenting images ====")

    # For each image, do skin segmentation
    for f in image_files:
        file_dir = os.path.dirname(f)
        patient_id = os.path.basename(file_dir)[1:]
        filename = os.path.splitext(os.path.basename(f))[0]

        skin_mask_dir = os.path.join(BASE_DIR, 'Shared', 'skin_masks',
                                     patient_id)
        skin_mask_file = os.path.join(skin_mask_dir, filename + '.png')

        if not os.path.isfile(skin_mask_file):
            image = cv2.imread(f, cv2.IMREAD_COLOR)
            image_ds = skinseg.resample(image)
            skin_mask1, skin_mask2, skin_mask3 = \
                skinseg.skinMaskColour(image_ds)
            skin_mask_gabor = skinseg.skinMaskGabor(
                image,
                gabor_kernels,
                kernel_params
            )

            if (image_counter % 10 == 0):
                print("Processed %4d out of %4d images"
                      % (image_counter, num_images))
            # print(patient_id)
            # print(filename)

            # create folder for storing skin masks if previously not exists
            if not os.path.exists(skin_mask_dir):
                os.makedirs(skin_mask_dir)

            # save skin masks
            '''
            cv2.imwrite(os.path.join(skin_mask_dir, 'sm1.png'), skin_mask1)
            cv2.imwrite(os.path.join(skin_mask_dir, 'sm2.png'), skin_mask2)
            cv2.imwrite(os.path.join(skin_mask_dir, 'sm3.png'), skin_mask3)
            cv2.imwrite(os.path.join(skin_mask_dir, 'sm4.png'), skin_mask_gabor)
            '''

            # combine skin masks
            pre_f_skin_mask = skinseg.combineSkinMasks(
                skin_mask1, skin_mask2,
                skin_mask3, skin_mask_gabor
            )

            # post-processing of skin mask
            final_skin_mask = skinseg.finalPostProcess(pre_f_skin_mask)

            # save final binary skin mask
            # note that skin mask is binary, so it may not be visible under
            # system viewer (all black)
            cv2.imwrite(os.path.join(skin_mask_dir, filename + '.png'),
                        final_skin_mask*255)

            image_counter = image_counter + 1
    print("\nCompleted skin segmentation")

    print("\n==== Image Augmentation ====")
    image_counter = 0
    for f in image_files:
        file_dir = os.path.dirname(f)
        patient_id = os.path.basename(file_dir)[1:]
        filename = os.path.splitext(os.path.basename(f))[0]

        skin_mask_dir = os.path.join(BASE_DIR, 'Shared', 'skin_masks',
                                     patient_id)
        skin_mask_file = os.path.join(skin_mask_dir, filename + '.png')

        aug_image_dir = os.path.join(AUG_IMAGE_DIR, 'images',
                                     patient_id)
        aug_sk_dir = os.path.join(AUG_IMAGE_DIR, 'skin_masks',
                                  patient_id)

        if (image_counter % 20 == 0):
            print("Processed %4d out of %4d images"
                  % (image_counter, num_images))

        # create folders for storing images and skin masks if previously not
        # exist
        if not os.path.exists(aug_image_dir):
            os.makedirs(aug_image_dir)
        if not os.path.exists(aug_sk_dir):
            os.makedirs(aug_sk_dir)

        image = cv2.imread(f, cv2.IMREAD_COLOR)
        skin_mask = cv2.imread(skin_mask_file, cv2.IMREAD_GRAYSCALE)

        image, skin_mask = augmentImage(image, skin_mask)

        # Please note all augmented images are resized to 512x512
        cv2.imwrite(os.path.join(aug_image_dir, filename + '.jpg'),
                    image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(os.path.join(aug_sk_dir, filename + '.png'),
                    skin_mask)

        image_counter = image_counter + 1
    print("\nCompleted image augmentation")


if __name__ == "__main__":
    main()
