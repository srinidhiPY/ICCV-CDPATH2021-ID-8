import csv
import glob
import os
import random
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import scipy.stats.stats as st
from skimage.measure import label
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from skimage.measure import regionprops
import utils as utils
from skimage import color
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation, opening, closing, area_closing, disk

N_FEATURES = 32
MAX, MEAN, VARIANCE, SKEWNESS, KURTOSIS = 0, 1, 2, 3, 4

##################################

###
def get_image_mask(wsi_path, level_used):

    try:
        wsi_image = OpenSlide(wsi_path)

        if wsi_image.level_count == 3:  # MSK
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used, wsi_image.level_dimensions[level_used]))

        elif wsi_image.level_count == 2:  # MSK
            level_used = level_used - 1  # MSK
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used, wsi_image.level_dimensions[level_used]))

        elif wsi_image.level_count == 4:
            level_used = level_used + 1   # MSK
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used, wsi_image.level_dimensions[level_used]))

        else:
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used, wsi_image.level_dimensions[level_used]))

        wsi_image.close()
    except OpenSlideUnsupportedFormatError:
        raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)

    # Mask generation
    img_RGB = np.asarray(rgb_image)
    img_HSV = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)

    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > 50
    min_G = img_RGB[:, :, 1] > 50
    min_B = img_RGB[:, :, 2] > 50

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask

###
def get_region_props(heatmap_threshold_2d, heatmap_prob_2d):
    labeled_img = label(heatmap_threshold_2d)
    return regionprops(labeled_img, intensity_image=heatmap_prob_2d)

###
def draw_bbox(heatmap_threshold, region_props, threshold_label='t90'):
    n_regions = len(region_props)
    print('No of regions(%s): %d' % (threshold_label, n_regions))
    for index in range(n_regions):
        region = region_props[index]
        print('area: ', region['area'])
        print('bbox: ', region['bbox'])
        print('centroid: ', region['centroid'])
        print('convex_area: ', region['convex_area'])
        print('eccentricity: ', region['eccentricity'])
        print('extent: ', region['extent'])
        print('major_axis_length: ', region['major_axis_length'])
        print('minor_axis_length: ', region['minor_axis_length'])
        print('orientation: ', region['orientation'])
        print('perimeter: ', region['perimeter'])
        print('solidity: ', region['solidity'])

        cv2.rectangle(heatmap_threshold, (region['bbox'][1], region['bbox'][0]),
                      (region['bbox'][3], region['bbox'][2]), color=(0, 255, 0), thickness=1)
        cv2.ellipse(heatmap_threshold, (int(region['centroid'][1]), int(region['centroid'][0])),
                    (int(region['major_axis_length'] / 2), int(region['minor_axis_length'] / 2)),
                    region['orientation'] * 90, 0, 360, color=(0, 0, 255), thickness=2)

    cv2.imshow('bbox_%s' % threshold_label, heatmap_threshold)

###
def get_largest_tumor_index(region_props):
    largest_tumor_index = -1

    largest_tumor_area = -1

    n_regions = len(region_props)
    for index in range(n_regions):
        if region_props[index]['area'] > largest_tumor_area:
            largest_tumor_area = region_props[index]['area']
            largest_tumor_index = index

    return largest_tumor_index

###
def get_longest_axis_in_largest_tumor_region(region_props, largest_tumor_region_index):
    largest_tumor_region = region_props[largest_tumor_region_index]
    return max(largest_tumor_region['major_axis_length'], largest_tumor_region['minor_axis_length'])


###
def get_tumor_region_to_tissue_ratio(region_props, image_open):
    tissue_area = cv2.countNonZero(np.float32(image_open))
    tumor_area = 0

    n_regions = len(region_props)
    for index in range(n_regions):
        tumor_area += region_props[index]['area']

    return float(tumor_area) / tissue_area


###
def get_tumor_region_to_bbox_ratio(region_props):
    # for all regions or largest region
    print()


###
def get_feature(region_props, n_region, feature_name):
    feature = [0] * 5
    if n_region > 0:
        feature_values = [region[feature_name] for region in region_props]
        feature[MAX] = utils.format_2f(np.max(feature_values))
        feature[MEAN] = utils.format_2f(np.mean(feature_values))
        feature[VARIANCE] = utils.format_2f(np.var(feature_values))
        feature[SKEWNESS] = utils.format_2f(st.skew(np.array(feature_values)))
        feature[KURTOSIS] = utils.format_2f(st.kurtosis(np.array(feature_values)))

    return feature

###
def get_average_prediction_across_tumor_regions(region_props):
    # close 255
    region_mean_intensity = [region.mean_intensity for region in region_props]
    return np.mean(region_mean_intensity)


#############################################################
def extract_features(heatmap_prob, mask):

    """
        Feature list:
        -> (01) given t = 0.90, total number of tumor regions
        -> (02) given t = 0.90, percentage of tumor region over the whole tissue region
        -> (03) given t = 0.50, the area of largest tumor region
        -> (04) given t = 0.90/0.50, the longest axis in the largest tumor region
        -> (05) given t = 0.90, total number pixels with probability greater than 0.90
        -> (06) given t = 0.90, average prediction across tumor region
        -> (07-11) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'area'
        -> (12-16) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'perimeter'
        -> (17-21) given t = 0.90, max, mean, variance, skewness, and kurtosis of  'compactness(eccentricity[?])'
        -> (22-26) given t = 0.90, max, mean, variance, skewness, and kurtosis of  'rectangularity(extent)'
        -> (27-31) given t = 0.90, max, mean, variance, skewness, and kurtosis of 'solidity'

    :param heatmap_prob:
    :param image_open:
    :return:
    """

    heatmap_threshold_t90 = np.array(heatmap_prob)
    heatmap_threshold_t50 = np.array(heatmap_prob)
    heatmap_threshold_t90[heatmap_threshold_t90 < int(0.95 * 255)] = 0              # 0.95 (best)
    heatmap_threshold_t90[heatmap_threshold_t90 >= int(0.95 * 255)] = 255           # 0.95 (best)
    heatmap_threshold_t50[heatmap_threshold_t50 <= int(0.5 * 255)] = 0              # 0.50 (best)
    heatmap_threshold_t50[heatmap_threshold_t50 > int(0.5 * 255)] = 255             # 0.50 (best)

    heatmap_threshold_t90_2d = np.reshape(heatmap_threshold_t90[:, :, :1],
                                          (heatmap_threshold_t90.shape[0], heatmap_threshold_t90.shape[1]))
    heatmap_threshold_t50_2d = np.reshape(heatmap_threshold_t50[:, :, :1],
                                          (heatmap_threshold_t50.shape[0], heatmap_threshold_t50.shape[1]))
    heatmap_prob_2d = np.reshape(heatmap_prob[:, :, :1],
                                 (heatmap_prob.shape[0], heatmap_prob.shape[1]))

    # Apply post-processing
    struct_elem = disk(3)   # default = 3     Dilation is always better, but erosion removes small tumor cells
    heatmap_threshold_t90_2d = dilation(heatmap_threshold_t90_2d, struct_elem)
    heatmap_threshold_t50_2d = dilation(heatmap_threshold_t50_2d, struct_elem)

    heatmap_threshold_t90_2d = closing(heatmap_threshold_t90_2d, struct_elem)
    heatmap_threshold_t50_2d = closing(heatmap_threshold_t50_2d, struct_elem)

    struct_elem = disk(0)   # default = 0
    heatmap_threshold_t90_2d = erosion(heatmap_threshold_t90_2d, struct_elem)
    heatmap_threshold_t50_2d = erosion(heatmap_threshold_t50_2d, struct_elem)
    # Image.fromarray(np.uint8(heatmap_threshold_t90_2d)).show()
    # Image.fromarray(np.uint8(heatmap_threshold_t50_2d)).show()

    'Calculate region properties using region props and labels for those regions'
    region_props_t90 = get_region_props(np.array(heatmap_threshold_t90_2d), heatmap_prob_2d)
    region_props_t50 = get_region_props(np.array(heatmap_threshold_t50_2d), heatmap_prob_2d)

    features = []

    f_count_tumor_region = len(region_props_t90)
    if f_count_tumor_region == 0:
        return [0.00] * N_FEATURES

    features.append(utils.format_2f(f_count_tumor_region))

    ###############################  New  Features  #########################
    f_percentage_tumor_over_tissue_region = get_tumor_region_to_tissue_ratio(region_props_t90, mask)
    features.append(utils.format_2f(f_percentage_tumor_over_tissue_region))

    largest_tumor_region_index_t90 = get_largest_tumor_index(region_props_t90)
    largest_tumor_region_index_t50 = get_largest_tumor_index(region_props_t50)
    f_area_largest_tumor_region_t50 = region_props_t50[largest_tumor_region_index_t50].area
    features.append(utils.format_2f(f_area_largest_tumor_region_t50))

    f_longest_axis_largest_tumor_region_t50 = get_longest_axis_in_largest_tumor_region(region_props_t50, largest_tumor_region_index_t50)
    features.append(utils.format_2f(f_longest_axis_largest_tumor_region_t50))

    f_longest_axis_largest_tumor_region_t90 = get_longest_axis_in_largest_tumor_region(region_props_t90, largest_tumor_region_index_t90)
    features.append(utils.format_2f(f_longest_axis_largest_tumor_region_t90))

    f_pixels_count_prob_gt_90 = cv2.countNonZero(heatmap_threshold_t90_2d)
    features.append(utils.format_2f(f_pixels_count_prob_gt_90))

    f_avg_prediction_across_tumor_regions = get_average_prediction_across_tumor_regions(region_props_t90)
    features.append(utils.format_2f(f_avg_prediction_across_tumor_regions))

    f_area = get_feature(region_props_t90, f_count_tumor_region, 'area')
    features += f_area

    f_perimeter = get_feature(region_props_t90, f_count_tumor_region, 'perimeter')
    features += f_perimeter

    f_eccentricity = get_feature(region_props_t90, f_count_tumor_region, 'eccentricity')
    features += f_eccentricity

    f_extent_t90 = get_feature(region_props_t90, len(region_props_t90), 'extent')
    features += f_extent_t90

    f_solidity = get_feature(region_props_t90, f_count_tumor_region, 'solidity')
    features += f_solidity

    return features

################################# Test ##########################################
def extract_features_test(heatmap_prob_name_postfix, f_test):

    test_wsi_paths = glob.glob(os.path.join(utils.TEST_WSI_PATH, '*.svs'))
    test_wsi_paths.sort()

    test_GT = pd.read_csv(utils.TEST_CSV_GT, header=None)

    features_file_test = open(f_test, 'w')

    wr_test = csv.writer(features_file_test, quoting=csv.QUOTE_NONNUMERIC)
    wr_test.writerow(utils.heatmap_feature_names)

    index = 0
    for wsi_path in test_wsi_paths:
        wsi_name = utils.get_filename_from_path(wsi_path)
        wsi_name = wsi_name[0].upper() + wsi_name[1:]
        print('extracting features for: %s' % wsi_name)
        heatmap_prob_path = glob.glob(os.path.join(utils.TEST_HEAT_MAP_DIR, '%s%s' % (wsi_name, heatmap_prob_name_postfix)))
        mask = get_image_mask(wsi_path, utils.Level_used_heatmaps_gen)
        heatmap_prob = cv2.imread(heatmap_prob_path[0])

        test_label = test_GT.loc[test_GT[0] == wsi_name][1]

        if test_label.iloc[0] == 'Tumor':
           test_label = 1
        elif test_label.iloc[0] == 'Normal':
            test_label = 0
        else:
            print('unknown label found')
            break

        test_features = extract_features(heatmap_prob, mask)         # Features
        test_features += str(test_label)                             # Labels
        print(test_features)
        wr_test.writerow(test_features)
        index += 1


###################################### Train_Validation ##########################
def extract_features_train_validation(heatmap_prob_name_postfix, f_train, f_validation):

    # Read training images
    tumor_train_wsi_paths = glob.glob(os.path.join(utils.TRAIN_WSI_PATH + 'tumor_train/', '*.tif'))
    tumor_train_wsi_paths.sort()
    normal_train_wsi_paths = glob.glob(os.path.join(utils.TRAIN_WSI_PATH + 'normal_train/', '*.tif'))
    normal_train_wsi_paths.sort()

    train_wsi_paths = tumor_train_wsi_paths + normal_train_wsi_paths
    print('number of training samples: %d' %len(train_wsi_paths))

    # Read validation images
    tumor_val_wsi_paths = glob.glob(os.path.join(utils.VAL_WSI_PATH + 'tumor_val/', '*.tif'))
    tumor_val_wsi_paths.sort()
    normal_val_wsi_paths = glob.glob(os.path.join(utils.VAL_WSI_PATH + 'normal_val/', '*.tif'))
    normal_val_wsi_paths.sort()

    val_wsi_paths = tumor_val_wsi_paths + normal_val_wsi_paths
    print('number of validation samples: %d' %len(val_wsi_paths))

    # Extract features
    features_file_train = open(f_train, 'w')
    features_file_validation = open(f_validation, 'w')

    train_GT = pd.read_csv(utils.TRAIN_CSV_GT, header=None)
    val_GT = pd.read_csv(utils.VAL_CSV_GT, header=None)

    wr_train = csv.writer(features_file_train, quoting=csv.QUOTE_NONNUMERIC)
    wr_validation = csv.writer(features_file_validation, quoting=csv.QUOTE_NONNUMERIC)
    wr_train.writerow(utils.heatmap_feature_names)
    wr_validation.writerow(utils.heatmap_feature_names)

    ### Training
    index = 0
    for wsi_path in train_wsi_paths:
        wsi_name = utils.get_filename_from_path(wsi_path)
        print('extracting features for training: %s' % wsi_name)
        heatmap_prob_path = glob.glob(os.path.join(utils.TRAIN_HEAT_MAP_DIR, '*%s*%s' % (wsi_name, heatmap_prob_name_postfix)))
        print(heatmap_prob_path)
        train_mask = get_image_mask(wsi_path, utils.Level_used_heatmaps_gen)
        train_heatmap_prob = cv2.imread(heatmap_prob_path[0])

        wsi_name = wsi_name.capitalize()
        train_label = train_GT.loc[train_GT[0] == wsi_name][1]

        if train_label.iloc[0] == 1:     # Tumor
           train_label = 1
        elif train_label.iloc[0] == 0:   # Normal
            train_label = 0
        else:
            print('unknown label found')
            break

        train_features = extract_features(train_heatmap_prob, train_mask)       # Features
        print(train_features)
        train_features += str(train_label)                                      # Labels
        wr_train.writerow(train_features)

        index += 1

    ### Validation
    index = 0
    for wsi_path in val_wsi_paths:
        wsi_name = utils.get_filename_from_path(wsi_path)
        print('extracting features for validation: %s' % wsi_name)
        heatmap_prob_path = glob.glob(os.path.join(utils.VAL_HEAT_MAP_DIR, '*%s*%s' % (wsi_name, heatmap_prob_name_postfix)))
        print(heatmap_prob_path)
        val_mask = get_image_mask(wsi_path, utils.Level_used_heatmaps_gen)
        val_heatmap_prob = cv2.imread(heatmap_prob_path[0])

        wsi_name = wsi_name.capitalize()
        val_label = val_GT.loc[val_GT[0] == wsi_name][1]

        if val_label.iloc[0] == 1:     # Tumor
           val_label = 1
        elif val_label.iloc[0] == 0:   # Normal
            val_label = 0
        else:
            print('unknown label found')
            break

        val_features = extract_features(val_heatmap_prob, val_mask)            # Features
        print(val_features)
        val_features += str(val_label)                                         # Labels
        wr_validation.writerow(val_features)

        index += 1


############## Main #####################
def extract_features_heatmap():

    # extract_features_train_validation('.png', utils.TRAIN_HEATMAP_FEATURE_CSV, utils.VAL_HEATMAP_FEATURE_CSV)   # train_val_set features (format, path to save features.csv)
    extract_features_test('.png', utils.TEST_HEATMAP_FEATURE_CSV)   # test_set features (format, path to save features.csv)


if __name__ == '__main__':

    ## Extract features from heatmap
    extract_features_heatmap()

